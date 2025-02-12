# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import uuid
import time

from agent import invoke_agent_h
import boto3
from flask import Flask, request, jsonify, render_template

# Let's use Amazon S3
s3 = boto3.resource("s3")

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route("/test")
def test():
    return "test"

@app.route("/invoke_agent", methods=['POST'])
def invoke_agent():
    start_time = time.time()

    data = request.json
    param = data.get('message', '')
    session_id: str = str(uuid.uuid1())
    # query = "Can you show me my reservation? I am Min"
    response = invoke_agent_h(param, session_id, 'R8BU8WVB8S', 'TSTALIASID', True)
    time_spent = time.time() - start_time

    print(f'agent response: {response}')

    # Assuming response is a string. If it's not, adjust accordingly.
    return jsonify({
        "response": response,
        "time_spent": f"{time_spent:.2f} seconds"
    })

@app.route("/server_request")
def server_request():
    print(request.args.get("param"))
    for bucket in s3.buckets.all():
        print(bucket.name)
    return "served"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
