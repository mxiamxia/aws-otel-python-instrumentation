# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import abc
import inspect
import io
import json
import logging
import math
import re
from typing import Any, Dict, Optional
from opentelemetry import trace
from opentelemetry.trace import get_tracer

from botocore.response import StreamingBody

from amazon.opentelemetry.distro._aws_attribute_keys import (
    AWS_BEDROCK_AGENT_ID,
    AWS_BEDROCK_DATA_SOURCE_ID,
    AWS_BEDROCK_GUARDRAIL_ARN,
    AWS_BEDROCK_GUARDRAIL_ID,
    AWS_BEDROCK_KNOWLEDGE_BASE_ID,
)
from amazon.opentelemetry.distro._aws_span_processing_util import (
    GEN_AI_REQUEST_MAX_TOKENS,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_REQUEST_TEMPERATURE,
    GEN_AI_REQUEST_TOP_P,
    GEN_AI_RESPONSE_FINISH_REASONS,
    GEN_AI_SYSTEM,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
)
from opentelemetry.instrumentation.botocore.extensions.types import (
    _AttributeMapT,
    _AwsSdkCallContext,
    _AwsSdkExtension,
    _BotoResultT,
)
from opentelemetry.trace.span import Span

_AGENT_ID: str = "agentId"
_KNOWLEDGE_BASE_ID: str = "knowledgeBaseId"
_DATA_SOURCE_ID: str = "dataSourceId"
_GUARDRAIL_ID: str = "guardrailId"
_GUARDRAIL_ARN: str = "guardrailArn"
_MODEL_ID: str = "modelId"
_AWS_BEDROCK_SYSTEM: str = "aws.bedrock"

_logger = logging.getLogger(__name__)
# Set logger level to DEBUG
_logger.setLevel(logging.DEBUG)


class _BedrockAgentOperation(abc.ABC):
    """
    We use subclasses and operation names to handle specific Bedrock Agent operations.
    - Only operations involving Agent, DataSource, or KnowledgeBase resources are supported.
    - Operations without these specified resources are not covered.
    - When an operation involves multiple resources (e.g., AssociateAgentKnowledgeBase),
      we map it to one resource based on some judgement classification of rules.

    For detailed API documentation on Bedrock Agent operations, visit:
    https://docs.aws.amazon.com/bedrock/latest/APIReference/API_Operations_Agents_for_Amazon_Bedrock.html
    """

    request_attributes: Optional[Dict[str, str]] = None
    response_attributes: Optional[Dict[str, str]] = None

    @classmethod
    @abc.abstractmethod
    def operation_names(cls):
        pass


class _AgentOperation(_BedrockAgentOperation):
    """
    This class covers BedrockAgent API related to <a
    href="https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent_Agent.html">Agents</a>,
    and extracts agent-related attributes.
    """

    request_attributes = {
        AWS_BEDROCK_AGENT_ID: _AGENT_ID,
    }
    response_attributes = {
        AWS_BEDROCK_AGENT_ID: _AGENT_ID,
    }

    @classmethod
    def operation_names(cls):
        return [
            "CreateAgentActionGroup",
            "CreateAgentAlias",
            "DeleteAgentActionGroup",
            "DeleteAgentAlias",
            "DeleteAgent",
            "DeleteAgentVersion",
            "GetAgentActionGroup",
            "GetAgentAlias",
            "GetAgent",
            "GetAgentVersion",
            "ListAgentActionGroups",
            "ListAgentAliases",
            "ListAgentKnowledgeBases",
            "ListAgentVersions",
            "PrepareAgent",
            "UpdateAgentActionGroup",
            "UpdateAgentAlias",
            "UpdateAgent",
        ]


class _KnowledgeBaseOperation(_BedrockAgentOperation):
    """
    This class covers BedrockAgent API related to <a
    href="https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent_KnowledgeBase.html">KnowledgeBases</a>,
    and extracts knowledge base-related attributes.

    Note: The 'CreateDataSource' operation does not have a 'dataSourceId' in the context,
    but it always comes with a 'knowledgeBaseId'. Therefore, we categorize it under 'knowledgeBaseId' operations.
    """

    request_attributes = {
        AWS_BEDROCK_KNOWLEDGE_BASE_ID: _KNOWLEDGE_BASE_ID,
    }
    response_attributes = {
        AWS_BEDROCK_KNOWLEDGE_BASE_ID: _KNOWLEDGE_BASE_ID,
    }

    @classmethod
    def operation_names(cls):
        return [
            "AssociateAgentKnowledgeBase",
            "CreateDataSource",
            "DeleteKnowledgeBase",
            "DisassociateAgentKnowledgeBase",
            "GetAgentKnowledgeBase",
            "GetKnowledgeBase",
            "ListDataSources",
            "UpdateAgentKnowledgeBase",
        ]


class _DataSourceOperation(_BedrockAgentOperation):
    """
    This class covers BedrockAgent API related to <a
    href="https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent_DataSource.html">DataSources</a>,
    and extracts data source-related attributes.
    """

    request_attributes = {
        AWS_BEDROCK_KNOWLEDGE_BASE_ID: _KNOWLEDGE_BASE_ID,
        AWS_BEDROCK_DATA_SOURCE_ID: _DATA_SOURCE_ID,
    }
    response_attributes = {
        AWS_BEDROCK_DATA_SOURCE_ID: _DATA_SOURCE_ID,
    }

    @classmethod
    def operation_names(cls):
        return ["DeleteDataSource", "GetDataSource", "UpdateDataSource"]


# _OPERATION_NAME_TO_CLASS_MAPPING maps operation names to their corresponding classes
# by iterating over all subclasses of _BedrockAgentOperation and extract operations
# by calling operation_names() function.
_OPERATION_NAME_TO_CLASS_MAPPING = {
    op_name: op_class
    for op_class in [_KnowledgeBaseOperation, _DataSourceOperation, _AgentOperation]
    for op_name in op_class.operation_names()
    if inspect.isclass(op_class) and issubclass(op_class, _BedrockAgentOperation) and not inspect.isabstract(op_class)
}


class _BedrockAgentExtension(_AwsSdkExtension):
    """
    This class is an extension for <a
    href="https://docs.aws.amazon.com/bedrock/latest/APIReference/API_Operations_Agents_for_Amazon_Bedrock.html">
    Agents for Amazon Bedrock</a>.

    This class primarily identify three types of resource based operations: _AgentOperation, _KnowledgeBaseOperation,
    and _DataSourceOperation. We only support operations that are related to the resource
    and where the context contains the resource ID.
    """

    def __init__(self, call_context: _AwsSdkCallContext):
        super().__init__(call_context)
        self._operation_class = _OPERATION_NAME_TO_CLASS_MAPPING.get(call_context.operation)

    def extract_attributes(self, attributes: _AttributeMapT):
        if self._operation_class is None:
            return
        for attribute_key, request_param_key in self._operation_class.request_attributes.items():
            request_param_value = self._call_context.params.get(request_param_key)
            if request_param_value:
                attributes[attribute_key] = request_param_value

    def on_success(self, span: Span, result: _BotoResultT):
        if self._operation_class is None:
            return

        for attribute_key, response_param_key in self._operation_class.response_attributes.items():
            response_param_value = result.get(response_param_key)
            if response_param_value:
                span.set_attribute(
                    attribute_key,
                    response_param_value,
                )


from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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

class _BedrockAgentRuntimeExtension(_AwsSdkExtension):
    """
    This class is an extension for <a
    href="https://docs.aws.amazon.com/bedrock/latest/APIReference/API_Operations_Agents_for_Amazon_Bedrock_Runtime.html">
    Agents for Amazon Bedrock Runtime</a>.
    """

    def __init__(self, call_context: _AwsSdkCallContext):
        super().__init__(call_context)
        self._logger = logging.getLogger(__name__)
        self._tracer = get_tracer(
            __name__,
            "test_version",
            None,
            schema_url="https://opentelemetry.io/schemas/1.11.0",
        )

    def extract_attributes(self, attributes: _AttributeMapT):
        agent_id = self._call_context.params.get(_AGENT_ID)
        if agent_id:
            attributes[AWS_BEDROCK_AGENT_ID] = agent_id

        knowledge_base_id = self._call_context.params.get(_KNOWLEDGE_BASE_ID)
        if knowledge_base_id:
            attributes[AWS_BEDROCK_KNOWLEDGE_BASE_ID] = knowledge_base_id

    def on_success(self, span: Span, result: _BotoResultT):
        """Process the success response from Bedrock Agent Runtime."""
        if self._call_context.span_name != "Bedrock Agent Runtime.InvokeAgent":
            return

        event_stream = result.get('completion')
        if not event_stream or not any(event_stream):
            return

        buffered_events = [event for event in event_stream]
        try:
            self._process_event_stream(span, buffered_events)
            result['completion'] = buffered_events
        except Exception as e:
            self._logger.error("Error processing event stream: %s", str(e))
            # Swallow the exception to not interrupt the application
            # but still preserve the original events
            result['completion'] = buffered_events

    def _process_event_stream(self, span: Span, events: List[Dict[str, Any]]):
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
                model_data = self._handle_model_invocation_input(trace_event)
            elif 'modelInvocationOutput' in trace_event and model_data:
                self._handle_model_invocation_output(span, trace_event, model_data)
                model_data = None
            elif 'invocationInput' in trace_event:
                agent_data = self._handle_invocation_input(trace_event)
            elif 'observation' in trace_event:
                self._handle_observation(span, trace_event, agent_data)
                agent_data = None
            elif 'inputAssessments' in trace_event:
                self._handle_guardrail(span, trace_event)
            elif 'rationale' in trace_event:
                self._handle_reasoning(span, trace_event)

    def _handle_model_invocation_input(self, trace_event: Dict[str, Any]) -> ModelInvocationData:
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

    def _handle_model_invocation_output(self, span: Span, trace_event: Dict[str, Any], model_data: ModelInvocationData):
        """Process model invocation output and create a span."""
        model_output = trace_event.get("modelInvocationOutput", {})
        usage = model_output.get("metadata", {}).get("usage", {})
        content = model_output.get("rawResponse", {}).get("content")

        model_data.output_content = content
        model_data.input_tokens = usage.get("inputTokens")
        model_data.output_tokens = usage.get("outputTokens")

        with self._tracer.start_as_current_span(
            "InvokeLlmModel",
            context=trace.set_span_in_context(span)
        ) as child_span:
            child_span.set_attribute("aws.local.service", "ai_agent")
            child_span.set_attribute("aws.local.operation", "invokeModel")

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

    def _handle_invocation_input(self, trace_event: Dict[str, Any]) -> Optional[AgentInvocationData]:
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

    def _handle_observation(self, span: Span, trace_event: Dict[str, Any], agent_data: Optional[AgentInvocationData]):
        """Process observation data and create appropriate spans."""
        observation_data = trace_event.get("observation", {})

        if observation_data.get("finalResponse"):
            self._create_final_response_span(span, observation_data)
        elif agent_data:
            self._create_agent_action_span(span, agent_data, observation_data)

    def _create_final_response_span(self, span: Span, observation_data: Dict[str, Any]):
        """Create a span for final response."""
        with self._tracer.start_as_current_span(
            "FinalResponse",
            context=trace.set_span_in_context(span)
        ) as child_span:
            child_span.set_attribute("aws.local.service", "ai_agent")
            child_span.set_attribute("aws.local.operation", "finalResponse")
            final_resp = observation_data.get("finalResponse", {})
            if final_resp.get("text"):
                child_span.set_attribute("gen_ai.agent.finalResponse", final_resp["text"])

    def _create_agent_action_span(self, span: Span, agent_data: AgentInvocationData, observation_data: Dict[str, Any]):
        """Create a span for agent action."""
        with self._tracer.start_as_current_span(
            "InvokeActionFunction",
            context=trace.set_span_in_context(span)
        ) as child_span:
            child_span.set_attribute("aws.local.service", "ai_agent")

            if agent_data.type == "action_group":
                self._set_action_group_attributes(child_span, agent_data, observation_data)
            elif agent_data.type == "knowledge_base":
                self._set_knowledge_base_attributes(child_span, agent_data, observation_data)

    def _set_action_group_attributes(self, span: Span, agent_data: AgentInvocationData, observation_data: Dict[str, Any]):
        """Set attributes for action group span."""
        span.set_attribute("aws.local.operation", "funcInvocation")
        if agent_data.action_group_name:
            span.set_attribute("gen_ai.agent.action_group.name", agent_data.action_group_name)
        if agent_data.execution_type:
            span.set_attribute("gen_ai.agent.action_group.execution_type", agent_data.execution_type)
        if agent_data.function:
            span.set_attribute("gen_ai.agent.action_group.function", agent_data.function)

        action_group_output = observation_data.get("actionGroupInvocationOutput", {})
        if action_group_output.get("text"):
            span.set_attribute("gen_ai.agent.action_group.output", action_group_output["text"])

    def _set_knowledge_base_attributes(self, span: Span, agent_data: AgentInvocationData, observation_data: Dict[str, Any]):
        """Set attributes for knowledge base span."""
        span.set_attribute("aws.local.operation", "kbQuery")
        if agent_data.invocation_type:
            span.set_attribute("gen_ai.agent.knowledge_base.invocation_type", agent_data.invocation_type)
        if agent_data.knowledge_base_id:
            span.set_attribute("gen_ai.agent.knowledge_base.id", agent_data.knowledge_base_id)
        if agent_data.text:
            span.set_attribute("gen_ai.agent.knowledge_base.output", agent_data.text)

        knowledge_base_output = observation_data.get("knowledgeBaseLookupOutput", {})
        retrieved_references = knowledge_base_output.get("retrievedReferences", [])
        span.set_attribute("gen_ai.retrievedReferences.count", len(retrieved_references))

    def _handle_guardrail(self, span: Span, trace_event: Dict[str, Any]):
        """Process guardrail data and create a span."""
        with self._tracer.start_as_current_span(
            "InvokeGuardrail",
            context=trace.set_span_in_context(span)
        ) as child_span:
            child_span.set_attribute("aws.local.service", "ai_agent")
            assessments = trace_event.get("inputAssessments", [])
            if assessments and "topicPolicy" in assessments[0]:
                topics = assessments[0]["topicPolicy"].get("topics", [])
                if topics:
                    child_span.set_attribute("gen_ai.guardrails.action", topics[0].get("action"))
                    child_span.set_attribute("gen_ai.guardrails.name", topics[0].get("name"))
                    child_span.set_attribute("gen_ai.guardrails.type", topics[0].get("type"))

    def _handle_reasoning(self, span: Span, trace_event: Dict[str, Any]):
        """Process reasoning data and create a span."""
        with self._tracer.start_as_current_span(
            "LlmModelReasoning",
            context=trace.set_span_in_context(span)
        ) as child_span:
            child_span.set_attribute("aws.local.service", "ai_agent")
            child_span.set_attribute("aws.local.operation", "rationale")
            rationale_text = trace_event.get("rationale", {}).get("text")
            if rationale_text:
                child_span.set_attribute("gen_ai.agent.reasoning.rationale", rationale_text)


class _BedrockExtension(_AwsSdkExtension):
    """
    This class is an extension for <a
    href="https://docs.aws.amazon.com/bedrock/latest/APIReference/API_Operations_Amazon_Bedrock.html">Bedrock</a>.
    """

    # pylint: disable=no-self-use
    def on_success(self, span: Span, result: _BotoResultT):
        # _GUARDRAIL_ID can only be retrieved from the response, not from the request
        guardrail_id = result.get(_GUARDRAIL_ID)
        if guardrail_id:
            span.set_attribute(
                AWS_BEDROCK_GUARDRAIL_ID,
                guardrail_id,
            )

        guardrail_arn = result.get(_GUARDRAIL_ARN)
        if guardrail_arn:
            span.set_attribute(
                AWS_BEDROCK_GUARDRAIL_ARN,
                guardrail_arn,
            )


class _BedrockRuntimeExtension(_AwsSdkExtension):
    """
    This class is an extension for <a
    href="https://docs.aws.amazon.com/bedrock/latest/APIReference/API_Operations_Amazon_Bedrock_Runtime.html">
    Amazon Bedrock Runtime</a>.
    """

    def extract_attributes(self, attributes: _AttributeMapT):
        attributes[GEN_AI_SYSTEM] = _AWS_BEDROCK_SYSTEM

        model_id = self._call_context.params.get(_MODEL_ID)
        if model_id:
            attributes[GEN_AI_REQUEST_MODEL] = model_id

            # Get the request body if it exists
            body = self._call_context.params.get("body")
            if body:
                try:
                    request_body = json.loads(body)

                    if "amazon.titan" in model_id:
                        self._extract_titan_attributes(attributes, request_body)
                    if "amazon.nova" in model_id:
                        self._extract_nova_attributes(attributes, request_body)
                    elif "anthropic.claude" in model_id:
                        self._extract_claude_attributes(attributes, request_body)
                    elif "meta.llama" in model_id:
                        self._extract_llama_attributes(attributes, request_body)
                    elif "cohere.command" in model_id:
                        self._extract_cohere_attributes(attributes, request_body)
                    elif "ai21.jamba" in model_id:
                        self._extract_ai21_attributes(attributes, request_body)
                    elif "mistral" in model_id:
                        self._extract_mistral_attributes(attributes, request_body)

                except json.JSONDecodeError:
                    _logger.debug("Error: Unable to parse the body as JSON")

    def _extract_titan_attributes(self, attributes, request_body):
        config = request_body.get("textGenerationConfig", {})
        self._set_if_not_none(attributes, GEN_AI_REQUEST_TEMPERATURE, config.get("temperature"))
        self._set_if_not_none(attributes, GEN_AI_REQUEST_TOP_P, config.get("topP"))
        self._set_if_not_none(attributes, GEN_AI_REQUEST_MAX_TOKENS, config.get("maxTokenCount"))

    def _extract_nova_attributes(self, attributes, request_body):
        config = request_body.get("inferenceConfig", {})
        self._set_if_not_none(attributes, GEN_AI_REQUEST_TEMPERATURE, config.get("temperature"))
        self._set_if_not_none(attributes, GEN_AI_REQUEST_TOP_P, config.get("top_p"))
        self._set_if_not_none(attributes, GEN_AI_REQUEST_MAX_TOKENS, config.get("max_new_tokens"))

    def _extract_claude_attributes(self, attributes, request_body):
        self._set_if_not_none(attributes, GEN_AI_REQUEST_MAX_TOKENS, request_body.get("max_tokens"))
        self._set_if_not_none(attributes, GEN_AI_REQUEST_TEMPERATURE, request_body.get("temperature"))
        self._set_if_not_none(attributes, GEN_AI_REQUEST_TOP_P, request_body.get("top_p"))

    def _extract_cohere_attributes(self, attributes, request_body):
        prompt = request_body.get("message")
        if prompt:
            attributes[GEN_AI_USAGE_INPUT_TOKENS] = math.ceil(len(prompt) / 6)
        self._set_if_not_none(attributes, GEN_AI_REQUEST_MAX_TOKENS, request_body.get("max_tokens"))
        self._set_if_not_none(attributes, GEN_AI_REQUEST_TEMPERATURE, request_body.get("temperature"))
        self._set_if_not_none(attributes, GEN_AI_REQUEST_TOP_P, request_body.get("p"))

    def _extract_ai21_attributes(self, attributes, request_body):
        self._set_if_not_none(attributes, GEN_AI_REQUEST_MAX_TOKENS, request_body.get("max_tokens"))
        self._set_if_not_none(attributes, GEN_AI_REQUEST_TEMPERATURE, request_body.get("temperature"))
        self._set_if_not_none(attributes, GEN_AI_REQUEST_TOP_P, request_body.get("top_p"))

    def _extract_llama_attributes(self, attributes, request_body):
        self._set_if_not_none(attributes, GEN_AI_REQUEST_MAX_TOKENS, request_body.get("max_gen_len"))
        self._set_if_not_none(attributes, GEN_AI_REQUEST_TEMPERATURE, request_body.get("temperature"))
        self._set_if_not_none(attributes, GEN_AI_REQUEST_TOP_P, request_body.get("top_p"))

    def _extract_mistral_attributes(self, attributes, request_body):
        prompt = request_body.get("prompt")
        if prompt:
            attributes[GEN_AI_USAGE_INPUT_TOKENS] = math.ceil(len(prompt) / 6)
        self._set_if_not_none(attributes, GEN_AI_REQUEST_MAX_TOKENS, request_body.get("max_tokens"))
        self._set_if_not_none(attributes, GEN_AI_REQUEST_TEMPERATURE, request_body.get("temperature"))
        self._set_if_not_none(attributes, GEN_AI_REQUEST_TOP_P, request_body.get("top_p"))

    @staticmethod
    def _set_if_not_none(attributes, key, value):
        if value is not None:
            attributes[key] = value

    # pylint: disable=too-many-branches
    def on_success(self, span: Span, result: Dict[str, Any]):
        model_id = self._call_context.params.get(_MODEL_ID)

        if not model_id:
            return

        if "body" in result and isinstance(result["body"], StreamingBody):
            original_body = None
            try:
                original_body = result["body"]
                body_content = original_body.read()

                # Use one stream for telemetry
                stream = io.BytesIO(body_content)
                telemetry_content = stream.read()
                response_body = json.loads(telemetry_content.decode("utf-8"))
                if "amazon.titan" in model_id:
                    self._handle_amazon_titan_response(span, response_body)
                if "amazon.nova" in model_id:
                    self._handle_amazon_nova_response(span, response_body)
                elif "anthropic.claude" in model_id:
                    self._handle_anthropic_claude_response(span, response_body)
                elif "meta.llama" in model_id:
                    self._handle_meta_llama_response(span, response_body)
                elif "cohere.command" in model_id:
                    self._handle_cohere_command_response(span, response_body)
                elif "ai21.jamba" in model_id:
                    self._handle_ai21_jamba_response(span, response_body)
                elif "mistral" in model_id:
                    self._handle_mistral_mistral_response(span, response_body)
                # Replenish stream for downstream application use
                new_stream = io.BytesIO(body_content)
                result["body"] = StreamingBody(new_stream, len(body_content))

            except json.JSONDecodeError:
                _logger.debug("Error: Unable to parse the response body as JSON")
            except Exception as e:  # pylint: disable=broad-exception-caught, invalid-name
                _logger.debug("Error processing response: %s", e)
            finally:
                if original_body is not None:
                    original_body.close()

    # pylint: disable=no-self-use
    def _handle_amazon_titan_response(self, span: Span, response_body: Dict[str, Any]):
        if "inputTextTokenCount" in response_body:
            span.set_attribute(GEN_AI_USAGE_INPUT_TOKENS, response_body["inputTextTokenCount"])
            if "results" in response_body and response_body["results"]:
                result = response_body["results"][0]
                if "tokenCount" in result:
                    span.set_attribute(GEN_AI_USAGE_OUTPUT_TOKENS, result["tokenCount"])
                if "completionReason" in result:
                    span.set_attribute(GEN_AI_RESPONSE_FINISH_REASONS, [result["completionReason"]])

    # pylint: disable=no-self-use
    def _handle_amazon_nova_response(self, span: Span, response_body: Dict[str, Any]):
        if "usage" in response_body:
            usage = response_body["usage"]
            if "inputTokens" in usage:
                span.set_attribute(GEN_AI_USAGE_INPUT_TOKENS, usage["inputTokens"])
            if "outputTokens" in usage:
                span.set_attribute(GEN_AI_USAGE_OUTPUT_TOKENS, usage["outputTokens"])
        if "stopReason" in response_body:
            span.set_attribute(GEN_AI_RESPONSE_FINISH_REASONS, [response_body["stopReason"]])

    # pylint: disable=no-self-use
    def _handle_anthropic_claude_response(self, span: Span, response_body: Dict[str, Any]):
        if "usage" in response_body:
            usage = response_body["usage"]
            if "input_tokens" in usage:
                span.set_attribute(GEN_AI_USAGE_INPUT_TOKENS, usage["input_tokens"])
            if "output_tokens" in usage:
                span.set_attribute(GEN_AI_USAGE_OUTPUT_TOKENS, usage["output_tokens"])
        if "stop_reason" in response_body:
            span.set_attribute(GEN_AI_RESPONSE_FINISH_REASONS, [response_body["stop_reason"]])

    # pylint: disable=no-self-use
    def _handle_cohere_command_response(self, span: Span, response_body: Dict[str, Any]):
        # Output tokens: Approximate from the response text
        if "text" in response_body:
            span.set_attribute(GEN_AI_USAGE_OUTPUT_TOKENS, math.ceil(len(response_body["text"]) / 6))
        if "finish_reason" in response_body:
            span.set_attribute(GEN_AI_RESPONSE_FINISH_REASONS, [response_body["finish_reason"]])

    # pylint: disable=no-self-use
    def _handle_ai21_jamba_response(self, span: Span, response_body: Dict[str, Any]):
        if "usage" in response_body:
            usage = response_body["usage"]
            if "prompt_tokens" in usage:
                span.set_attribute(GEN_AI_USAGE_INPUT_TOKENS, usage["prompt_tokens"])
            if "completion_tokens" in usage:
                span.set_attribute(GEN_AI_USAGE_OUTPUT_TOKENS, usage["completion_tokens"])
        if "choices" in response_body:
            choices = response_body["choices"][0]
            if "finish_reason" in choices:
                span.set_attribute(GEN_AI_RESPONSE_FINISH_REASONS, [choices["finish_reason"]])

    # pylint: disable=no-self-use
    def _handle_meta_llama_response(self, span: Span, response_body: Dict[str, Any]):
        if "prompt_token_count" in response_body:
            span.set_attribute(GEN_AI_USAGE_INPUT_TOKENS, response_body["prompt_token_count"])
        if "generation_token_count" in response_body:
            span.set_attribute(GEN_AI_USAGE_OUTPUT_TOKENS, response_body["generation_token_count"])
        if "stop_reason" in response_body:
            span.set_attribute(GEN_AI_RESPONSE_FINISH_REASONS, [response_body["stop_reason"]])

    # pylint: disable=no-self-use
    def _handle_mistral_mistral_response(self, span: Span, response_body: Dict[str, Any]):
        if "outputs" in response_body:
            outputs = response_body["outputs"][0]
            if "text" in outputs:
                span.set_attribute(GEN_AI_USAGE_OUTPUT_TOKENS, math.ceil(len(outputs["text"]) / 6))
        if "stop_reason" in outputs:
            span.set_attribute(GEN_AI_RESPONSE_FINISH_REASONS, [outputs["stop_reason"]])
