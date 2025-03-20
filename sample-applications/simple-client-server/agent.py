import boto3
import json
import time
import zipfile
import logging
import uuid
import traceback
import pprint
import threading
import schedule
import re
from io import BytesIO
from collections import defaultdict
from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode
from opentelemetry.trace import get_tracer
from opentelemetry.trace.span import Span
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider

iam_client = boto3.client('iam')
sts_client = boto3.client('sts')
session = boto3.session.Session()
region = session.region_name
account_id = sts_client.get_caller_identity()["Account"]
dynamodb_client = boto3.client('dynamodb')
dynamodb_resource = boto3.resource('dynamodb')
lambda_client = boto3.client('lambda')
bedrock_agent_client = boto3.client('bedrock-agent')
bedrock_agent_runtime_client = boto3.client('bedrock-agent-runtime')
logging.basicConfig(format='[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache for event stream data by session_id
event_stream_cache = defaultdict(list)
scheduler_started = False

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from opentelemetry.propagate import inject, extract
from opentelemetry.trace import SpanContext

tracer = get_tracer(
            __name__,
            "test_version",
            None,
            schema_url="https://opentelemetry.io/schemas/1.11.0",
        )
@dataclass
class ModelInvocationData:
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    prompt_input: Optional[str] = None
    output_content: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None

@dataclass
class AgentInvocationData:
    type: str
    action_group_name: Optional[str] = None
    execution_type: Optional[str] = None
    function: Optional[str] = None
    invocation_type: Optional[str] = None
    knowledge_base_id: Optional[str] = None
    text: Optional[str] = None

def create_dynamodb(table_name):
    try:
        table = dynamodb_resource.create_table(
            TableName=table_name,
            KeySchema=[
                {
                    'AttributeName': 'booking_id',
                    'KeyType': 'HASH'
                }
            ],
            AttributeDefinitions=[
                {
                    'AttributeName': 'booking_id',
                    'AttributeType': 'S'
                }
            ],
            BillingMode='PAY_PER_REQUEST'  # Use on-demand capacity mode
        )

        # Wait for the table to be created
        print(f'Creating table {table_name}...')
        table.wait_until_exists()
        print(f'Table {table_name} created successfully!')
    except dynamodb_client.exceptions.ResourceInUseException:
        print(f'Table {table_name} already exists, skipping table creation step')


def create_lambda(lambda_function_name, lambda_iam_role):
    # add to function

    # Package up the lambda function code
    s = BytesIO()
    z = zipfile.ZipFile(s, 'w')
    z.write("lambda_function.py")
    z.close()
    zip_content = s.getvalue()
    try:
        # Create Lambda Function
        lambda_function = lambda_client.create_function(
            FunctionName=lambda_function_name,
            Runtime='python3.12',
            Timeout=60,
            Role=lambda_iam_role['Role']['Arn'],
            Code={'ZipFile': zip_content},
            Handler='lambda_function.lambda_handler'
        )
    except lambda_client.exceptions.ResourceConflictException:
        print("Lambda function already exists, retrieving it")
        lambda_function = lambda_client.get_function(
            FunctionName=lambda_function_name
        )
        lambda_function = lambda_function['Configuration']

    return lambda_function


def create_lambda_role(agent_name, dynamodb_table_name):
    lambda_function_role = f'{agent_name}-lambda-role'
    dynamodb_access_policy_name = f'{agent_name}-dynamodb-policy'
    # Create IAM Role for the Lambda function
    try:
        assume_role_policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {
                        "Service": "lambda.amazonaws.com"
                    },
                    "Action": "sts:AssumeRole"
                }
            ]
        }

        assume_role_policy_document_json = json.dumps(assume_role_policy_document)

        lambda_iam_role = iam_client.create_role(
            RoleName=lambda_function_role,
            AssumeRolePolicyDocument=assume_role_policy_document_json
        )

        # Pause to make sure role is created
        time.sleep(10)
    except iam_client.exceptions.EntityAlreadyExistsException:
        lambda_iam_role = iam_client.get_role(RoleName=lambda_function_role)

    # Attach the AWSLambdaBasicExecutionRole policy
    iam_client.attach_role_policy(
        RoleName=lambda_function_role,
        PolicyArn='arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole'
    )

    # Create a policy to grant access to the DynamoDB table
    dynamodb_access_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "dynamodb:GetItem",
                    "dynamodb:PutItem",
                    "dynamodb:DeleteItem"
                ],
                "Resource": "arn:aws:dynamodb:{}:{}:table/{}".format(
                    region, account_id, dynamodb_table_name
                )
            }
        ]
    }

    # Create the policy
    dynamodb_access_policy_json = json.dumps(dynamodb_access_policy)
    try:
        dynamodb_access_policy = iam_client.create_policy(
            PolicyName=dynamodb_access_policy_name,
            PolicyDocument=dynamodb_access_policy_json
        )
    except iam_client.exceptions.EntityAlreadyExistsException:
        dynamodb_access_policy = iam_client.get_policy(
            PolicyArn=f"arn:aws:iam::{account_id}:policy/{dynamodb_access_policy_name}"
        )

    # Attach the policy to the Lambda function's role
    iam_client.attach_role_policy(
        RoleName=lambda_function_role,
        PolicyArn=dynamodb_access_policy['Policy']['Arn']
    )
    return lambda_iam_role


def invoke_agent_h(query, session_id, agent_id, alias_id, enable_trace=False, session_state=None):
    """
    Invoke the agent and cache the entire event stream for later processing
    """
    end_session: bool = False
    if not session_state:
        session_state = {}

    # invoke the agent API
    agent_response = bedrock_agent_runtime_client.invoke_agent(
        inputText=query,
        agentId=agent_id,
        agentAliasId=alias_id,
        sessionId=session_id,
        enableTrace=enable_trace,
        endSession=end_session,
        sessionState=session_state
    )

    print(f"print agent id: {agent_id}")

    # Cache the entire event_stream completion object
    event_stream = agent_response['completion']
    
    # Convert the event_stream to a list of events
    events = list(event_stream)
    
    # Start the scheduler if not already started
    global scheduler_started
    if not scheduler_started:
        start_trace_processing_scheduler()
        scheduler_started = True
    
    agent_answer = None
    
    print(f"event size is {len(events)}")
    try:
        # Process events to get the answer
        for event in events:
            if 'chunk' in event:
                data = event['chunk']['bytes']
                agent_answer = data.decode('utf8')
                print(f"Final answer in chunk is {agent_answer}")
            elif 'trace' in event:
                logger.info(json.dumps(event['trace'], indent=2))
                trace_event = event.get('trace', {}).get('trace', {}).get('orchestrationTrace', {})
                if 'observation' in trace_event:
                    observation_data = trace_event.get("observation", {})
                    if observation_data.get("finalResponse"):
                        final_resp = observation_data.get("finalResponse", {}).get("text")
                        print(f"Final response in trace is {final_resp}")
                        if not agent_answer:
                            agent_answer = final_resp
            else:
                raise Exception("unexpected event.", event)
        print(f"event size is at the end {len(events)} for {session_id}")
        # Store the entire event stream in cache
        current_span = trace.get_current_span()
        if current_span:
            span_context = current_span.get_span_context()
            print(f"Trace ID: {span_context.trace_id:x}")
            print(f"Span ID: {span_context.span_id:x}")
            headers = {}
            inject(headers)
            print("Serialized SpanContext into headers:", headers)
        event_stream_cache[session_id] = {
            "headers": headers,
            "events": events
        }
        return agent_answer
    except Exception as e:
        raise Exception("unexpected event.", e)

    
def process_cached_traces():
    """
    Process and log all events from the cached event_stream data
    """
    logger.info("Processing cached event streams...")
    
    # Start a server span named "AWS Bedrock Agent"
    if len(event_stream_cache) > 0:
        try:
            for session_id, session_data in event_stream_cache.items():
                span_context = extract(session_data["headers"])  # Restore SpanContext
                events = session_data["events"]
                logger.info(f"Processing event stream for session: {session_id} : {len(events)}")
                with tracer.start_as_current_span(
                    name="AWS Bedrock Agent",
                    kind=SpanKind.SERVER,
                    context=span_context,
                    attributes={
                        "session_id": session_id,
                        "aws.local.service": "BookingAgent",
                        "aws.local.operation": "InvokeAgent"
                    }
                ) as span:
                    span.set_attribute("aws.local.service", "BookingAgent")
                _process_event_stream(span, events)
        except Exception as e:
            logger.error("Error processing event stream: %s", str(e))
            logger.error("Stack Trace: %s", traceback.format_exc())
            
            # Swallow the exception to not interrupt the application
            # but still preserve the original events
        
        
    # for session_id, events in event_stream_cache.items():
    #     logger.info(f"Processing event stream for session: {session_id}")
    #     trace_count = 0
    #     chunk_count = 0
    #     other_count = 0
        
    #     for event in events:
    #         if 'trace' in event:
    #             trace_count += 1
    #             trace_data = event.get('trace', {})
    #             logger.info(f"Trace {trace_count} for session {session_id}: {json.dumps(trace_data, indent=2)}")
    #         elif 'chunk' in event:
    #             chunk_count += 1
    #             try:
    #                 chunk_data = event['chunk']['bytes'].decode('utf8')
    #                 logger.info(f"Chunk {chunk_count} for session {session_id}: {chunk_data}")
    #             except Exception as e:
    #                 logger.warning(f"Could not decode chunk {chunk_count} for session {session_id}: {str(e)}")
    #         else:
    #             other_count += 1
    #             logger.info(f"Other event type for session {session_id}: {json.dumps(event, indent=2)}")
        
        # logger.info(f"Session {session_id} summary: Processed {trace_count} traces, {chunk_count} chunks, and {other_count} other events")
    
    # Clear the entire event stream cache after processing
    if event_stream_cache:
        logger.info(f"Clearing event stream cache for {len(event_stream_cache)} sessions")
        event_stream_cache.clear()
    else:
        logger.info("No event streams to process")

def _process_event_stream(span: Span, events: List[Dict[str, Any]]):
    """Process the event stream and create appropriate spans."""
    model_data = None
    agent_data = None

    for event in events:
        if 'trace' not in event:
            continue

        trace_event = (
            event.get('trace', {})
            .get('trace', {})
            .get('orchestrationTrace', {}) or
            event.get('trace', {})
            .get('trace', {})
            .get('guardrailTrace', {})
        )

        if not trace_event:
            continue

        if 'modelInvocationInput' in trace_event:
            model_data = _handle_model_invocation_input(trace_event)
        elif 'modelInvocationOutput' in trace_event and model_data:
            _handle_model_invocation_output(span, trace_event, model_data)
            model_data = None
        elif 'invocationInput' in trace_event:
            agent_data = _handle_invocation_input(trace_event)
        elif 'observation' in trace_event:
            _handle_observation(span, trace_event, agent_data)
            agent_data = None
        elif 'inputAssessments' in trace_event:
            _handle_guardrail(span, trace_event)
        elif 'rationale' in trace_event:
            _handle_reasoning(span, trace_event)

def _handle_model_invocation_input(trace_event: Dict[str, Any]) -> ModelInvocationData:
        """Process model invocation input and return structured data."""
        model_input = trace_event.get("modelInvocationInput", {})
        inference_config = model_input.get("inferenceConfiguration", {})
        prompt_json_str = model_input.get("text", {})

        cleaned_str = re.sub(r'(?<!\\)\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', prompt_json_str)
        data = json.loads(cleaned_str)

        message_content = None
        for message in reversed(data.get("messages", [])):
            if message.get("role") == "user":
                message_content = message.get("content")
                break

        return ModelInvocationData(
            temperature=inference_config.get("temperature"),
            top_p=inference_config.get("topP"),
            prompt_input=message_content
        )

def _handle_model_invocation_output(span: Span, trace_event: Dict[str, Any], model_data: ModelInvocationData):
    """Process model invocation output and create a span."""
    model_output = trace_event.get("modelInvocationOutput", {})
    usage = model_output.get("metadata", {}).get("usage", {})
    content = model_output.get("rawResponse", {}).get("content")

    model_data.output_content = content
    model_data.input_tokens = usage.get("inputTokens")
    model_data.output_tokens = usage.get("outputTokens")

    with tracer.start_as_current_span(
        "InvokeLlmModel",
        context=trace.set_span_in_context(span),
        kind=SpanKind.CLIENT
    ) as child_span:
        child_span.set_attribute("aws.local.service", "BookingAgent")
        child_span.set_attribute("aws.local.operation", "InvokeAgent")
        child_span.set_attribute("aws.remote.service", "ClaudeModel")
        child_span.set_attribute("aws.remote.operation", "InvokeModel")
        
        if model_data.temperature is not None:
            child_span.set_attribute("gen_ai.request.temperature", model_data.temperature)
        if model_data.top_p is not None:
            child_span.set_attribute("gen_ai.request.top_p", model_data.top_p)
        if model_data.prompt_input is not None:
            child_span.set_attribute("gen_ai.request.prompt", model_data.prompt_input)
            child_span.add_event("gen_ai.user.message", {"content": model_data.prompt_input})
        if model_data.output_content is not None:
            child_span.set_attribute("gen_ai.request.output", model_data.output_content)
            child_span.add_event("gen_ai.choice", {
                "index": 0,
                "finish_reason": "stop",
                "message": {"content": model_data.output_content}
            })
        if model_data.input_tokens is not None:
            child_span.set_attribute("gen_ai.usage.input_tokens", model_data.input_tokens)
        if model_data.output_tokens is not None:
            child_span.set_attribute("gen_ai.usage.output_tokens", model_data.output_tokens)

def _handle_invocation_input(trace_event: Dict[str, Any]) -> Optional[AgentInvocationData]:
    """Process invocation input and return structured data."""
    invocation_data = trace_event.get("invocationInput", {})
    action_group_data = invocation_data.get("actionGroupInvocationInput", {})
    knowledge_base_data = invocation_data.get("knowledgeBaseLookupInput", {})

    if action_group_data:
        return AgentInvocationData(
            type="action_group",
            action_group_name=action_group_data.get("actionGroupName"),
            execution_type=action_group_data.get("executionType"),
            function=action_group_data.get("function")
        )
    elif knowledge_base_data:
        return AgentInvocationData(
            type="knowledge_base",
            invocation_type=invocation_data.get("invocationType"),
            knowledge_base_id=knowledge_base_data.get("knowledgeBaseId"),
            text=knowledge_base_data.get("text")
        )
    return None

def _handle_observation(span: Span, trace_event: Dict[str, Any], agent_data: Optional[AgentInvocationData]):
    """Process observation data and create appropriate spans."""
    observation_data = trace_event.get("observation", {})

    if observation_data.get("finalResponse"):
        _create_final_response_span(span, observation_data)
    elif agent_data:
        _create_agent_action_span(span, agent_data, observation_data)

def _create_final_response_span(span: Span, observation_data: Dict[str, Any]):
    """Create a span for final response."""
    with tracer.start_as_current_span(
        "FinalResponse",
        context=trace.set_span_in_context(span),
        kind=SpanKind.CLIENT
    ) as child_span:
        child_span.set_attribute("aws.local.service", "BookingAgent")
        child_span.set_attribute("aws.local.operation", "InvokeAgent")
        child_span.set_attribute("aws.remote.service", "ClaudeModel")
        child_span.set_attribute("aws.remote.operation", "InvokeModel")
        final_resp = observation_data.get("finalResponse", {})
        if final_resp.get("text"):
            child_span.set_attribute("gen_ai.agent.finalResponse", final_resp["text"])

def _create_agent_action_span(span: Span, agent_data: AgentInvocationData, observation_data: Dict[str, Any]):
    """Create a span for agent action."""
    with tracer.start_as_current_span(
        "InvokeActionFunction",
        context=trace.set_span_in_context(span),
        kind=SpanKind.CLIENT
    ) as child_span:
        child_span.set_attribute("aws.local.service", "BookingAgent")
        child_span.set_attribute("aws.local.operation", "InvokeAgent")

        if agent_data.type == "action_group":
            _set_action_group_attributes(child_span, agent_data, observation_data)
        elif agent_data.type == "knowledge_base":
            _set_knowledge_base_attributes(child_span, agent_data, observation_data)

def _set_action_group_attributes(span: Span, agent_data: AgentInvocationData, observation_data: Dict[str, Any]):
    """Set attributes for action group span."""
    span.set_attribute("aws.local.operation", "funcInvocation")
    if agent_data.action_group_name:
        span.set_attribute("gen_ai.agent.action_group.name", agent_data.action_group_name)
    if agent_data.execution_type:
        span.set_attribute("gen_ai.agent.action_group.execution_type", agent_data.execution_type)
    if agent_data.function:
        span.set_attribute("gen_ai.agent.action_group.function", agent_data.function)
        span.set_attribute("aws.remote.service", agent_data.function)
        span.set_attribute("aws.remote.operation", "invokeLambda")

    action_group_output = observation_data.get("actionGroupInvocationOutput", {})
    if action_group_output.get("text"):
        span.set_attribute("gen_ai.agent.action_group.output", action_group_output["text"])

def _set_knowledge_base_attributes(span: Span, agent_data: AgentInvocationData, observation_data: Dict[str, Any]):
    """Set attributes for knowledge base span."""
    span.set_attribute("aws.local.operation", "kbQuery")
    if agent_data.invocation_type:
        span.set_attribute("gen_ai.agent.knowledge_base.invocation_type", agent_data.invocation_type)
    if agent_data.knowledge_base_id:
        span.set_attribute("gen_ai.agent.knowledge_base.id", agent_data.knowledge_base_id)
        span.set_attribute("aws.remote.service", "KnowledgeBase")
        span.set_attribute("aws.remote.operation", "QueryKB")
    if agent_data.text:
        span.set_attribute("gen_ai.agent.knowledge_base.output", agent_data.text)

    knowledge_base_output = observation_data.get("knowledgeBaseLookupOutput", {})
    retrieved_references = knowledge_base_output.get("retrievedReferences", [])
    span.set_attribute("gen_ai.retrievedReferences.count", len(retrieved_references))

def _handle_guardrail(span: Span, trace_event: Dict[str, Any]):
    """Process guardrail data and create a span."""
    with tracer.start_as_current_span(
        "InvokeGuardrail",
        context=trace.set_span_in_context(span),
        kind=SpanKind.CLIENT
    ) as child_span:
        child_span.set_attribute("aws.local.service", "BookingAgent")
        child_span.set_attribute("aws.local.operation", "InvokeAgent")
        child_span.set_attribute("aws.remote.service", "Guardrail")
        child_span.set_attribute("aws.remote.operation", "QueryGaurdrail")
        assessments = trace_event.get("inputAssessments", [])
        if assessments and "topicPolicy" in assessments[0]:
            topics = assessments[0]["topicPolicy"].get("topics", [])
            if topics:
                child_span.set_attribute("gen_ai.guardrails.action", topics[0].get("action"))
                child_span.set_attribute("gen_ai.guardrails.name", topics[0].get("name"))
                child_span.set_attribute("gen_ai.guardrails.type", topics[0].get("type"))

def _handle_reasoning(span: Span, trace_event: Dict[str, Any]):
    """Process reasoning data and create a span."""
    with tracer.start_as_current_span(
        "LlmModelReasoning",
        context=trace.set_span_in_context(span),
        kind=SpanKind.CLIENT
    ) as child_span:
        child_span.set_attribute("aws.local.service", "BookingAgent")
        child_span.set_attribute("aws.local.operation", "InvokeAgent")
        rationale_text = trace_event.get("rationale", {}).get("text")
        if rationale_text:
            child_span.set_attribute("gen_ai.agent.reasoning.rationale", rationale_text)
        child_span.set_attribute("aws.remote.service", "ClaudeModel")
        child_span.set_attribute("aws.remote.operation", "Reasoning")


def start_trace_processing_scheduler():
    """
    Start a scheduler to periodically process traces
    """
    logger.info("Starting trace processing scheduler")
    
    # Schedule the trace processing to run every minute
    schedule.every(3).seconds.do(process_cached_traces)
    
    # Run the scheduler in a separate thread
    def run_scheduler():
        while True:
            schedule.run_pending()
            time.sleep(1)
    
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()


def create_agent_role(agent_name, agent_foundation_model, kb_id=None):
    agent_bedrock_allow_policy_name = f"{agent_name}-ba"
    agent_role_name = f'AmazonBedrockExecutionRoleForAgents_{agent_name}'
    # Create IAM policies for agent
    statements = [
        {
            "Sid": "AmazonBedrockAgentBedrockFoundationModelPolicy",
            "Effect": "Allow",
            "Action": "bedrock:InvokeModel",
            "Resource": [
                f"arn:aws:bedrock:{region}::foundation-model/{agent_foundation_model}"
            ]
        }
    ]
    # add Knowledge Base retrieve and retrieve and generate permissions if agent has KB attached to it
    if kb_id:
        statements.append(
            {
                "Sid": "QueryKB",
                "Effect": "Allow",
                "Action": [
                    "bedrock:Retrieve",
                    "bedrock:RetrieveAndGenerate"
                ],
                "Resource": [
                    f"arn:aws:bedrock:{region}:{account_id}:knowledge-base/{kb_id}"
                ]
            }
        )

    bedrock_agent_bedrock_allow_policy_statement = {
        "Version": "2012-10-17",
        "Statement": statements
    }

    bedrock_policy_json = json.dumps(bedrock_agent_bedrock_allow_policy_statement)
    try:
        agent_bedrock_policy = iam_client.create_policy(
            PolicyName=agent_bedrock_allow_policy_name,
            PolicyDocument=bedrock_policy_json
        )
    except iam_client.exceptions.EntityAlreadyExistsException:
        agent_bedrock_policy = iam_client.get_policy(
            PolicyArn=f"arn:aws:iam::{account_id}:policy/{agent_bedrock_allow_policy_name}"
        )
                    
    # Create IAM Role for the agent and attach IAM policies
    assume_role_policy_document = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {
                "Service": "bedrock.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }]
    }

    assume_role_policy_document_json = json.dumps(assume_role_policy_document)
    try:
        agent_role = iam_client.create_role(
            RoleName=agent_role_name,
            AssumeRolePolicyDocument=assume_role_policy_document_json
        )

        # Pause to make sure role is created
        time.sleep(10)
    except iam_client.exceptions.EntityAlreadyExistsException:
        agent_role = iam_client.get_role(
            RoleName=agent_role_name,
        )

    iam_client.attach_role_policy(
        RoleName=agent_role_name,
        PolicyArn=agent_bedrock_policy['Policy']['Arn']
    )
    return agent_role


def delete_agent_roles_and_policies(agent_name, kb_policy_name):
    agent_bedrock_allow_policy_name = f"{agent_name}-ba"
    agent_role_name = f'AmazonBedrockExecutionRoleForAgents_{agent_name}'
    dynamodb_access_policy_name = f'{agent_name}-dynamodb-policy'
    lambda_function_role = f'{agent_name}-lambda-role'

    for policy in [agent_bedrock_allow_policy_name, kb_policy_name]:
        try:
            iam_client.detach_role_policy(
                RoleName=agent_role_name,
                PolicyArn=f'arn:aws:iam::{account_id}:policy/{policy}'
            )
        except Exception as e:
            print(f"Could not detach {policy} from {agent_role_name}")
            print(e)

    for policy in [dynamodb_access_policy_name]:
        try:
            iam_client.detach_role_policy(
                RoleName=lambda_function_role,
                PolicyArn=f'arn:aws:iam::{account_id}:policy/{policy}'
            )
        except Exception as e:
            print(f"Could not detach {policy} from {lambda_function_role}")
            print(e)

    try:
        iam_client.detach_role_policy(
            RoleName=lambda_function_role,
            PolicyArn='arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole'
        )
    except Exception as e:
        print(f"Could not detach AWSLambdaBasicExecutionRole from {lambda_function_role}")
        print(e)

    for role_name in [agent_role_name, lambda_function_role]:
        try:
            iam_client.delete_role(
                RoleName=role_name
            )
        except Exception as e:
            print(f"Could not delete role {role_name}")
            print(e)

    for policy in [agent_bedrock_allow_policy_name, kb_policy_name, dynamodb_access_policy_name]:
        try:
            iam_client.delete_policy(
                PolicyArn=f'arn:aws:iam::{account_id}:policy/{policy}'
            )
        except Exception as e:
            print(f"Could not delete policy {policy}")
            print(e)


def clean_up_resources(
        table_name, lambda_function, lambda_function_name, agent_action_group_response, agent_functions,
        agent_id, kb_id, alias_id
):
    action_group_id = agent_action_group_response['agentActionGroup']['actionGroupId']
    action_group_name = agent_action_group_response['agentActionGroup']['actionGroupName']
    # Delete Agent Action Group, Agent Alias, and Agent
    try:
        bedrock_agent_client.update_agent_action_group(
            agentId=agent_id,
            agentVersion='DRAFT',
            actionGroupId= action_group_id,
            actionGroupName=action_group_name,
            actionGroupExecutor={
                'lambda': lambda_function['FunctionArn']
            },
            functionSchema={
                'functions': agent_functions
            },
            actionGroupState='DISABLED',
        )
        bedrock_agent_client.disassociate_agent_knowledge_base(
            agentId=agent_id,
            agentVersion='DRAFT',
            knowledgeBaseId=kb_id
        )
        bedrock_agent_client.delete_agent_action_group(
            agentId=agent_id,
            agentVersion='DRAFT',
            actionGroupId=action_group_id
        )
        bedrock_agent_client.delete_agent_alias(
            agentAliasId=alias_id,
            agentId=agent_id
        )
        bedrock_agent_client.delete_agent(agentId=agent_id)
        print(f"Agent {agent_id}, Agent Alias {alias_id}, and Action Group have been deleted.")
    except Exception as e:
        print(f"Error deleting Agent resources: {e}")

    # Delete Lambda function
    try:
        lambda_client.delete_function(FunctionName=lambda_function_name)
        print(f"Lambda function {lambda_function_name} has been deleted.")
    except Exception as e:
        print(f"Error deleting Lambda function {lambda_function_name}: {e}")

    # Delete DynamoDB table
    try:
        dynamodb_client.delete_table(TableName=table_name)
        print(f"Table {table_name} is being deleted...")
        waiter = dynamodb_client.get_waiter('table_not_exists')
        waiter.wait(TableName=table_name)
        print(f"Table {table_name} has been deleted.")
    except Exception as e:
        print(f"Error deleting table {table_name}: {e}")


# Force trace processing when needed (can be called manually)
def force_process_traces():
    process_cached_traces()


if __name__ == "__main__":
    session_id: str = str(uuid.uuid1())
    query = "this is a test"
    response = invoke_agent_h(query, session_id, 'R8BU8WVB8S', 'TSTALIASID', True)
    print(f"the final response is {response}")
    
    # Wait for a moment and then process traces
    time.sleep(5)
    process_cached_traces()
