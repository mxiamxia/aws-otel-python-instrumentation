# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=no-self-use

import logging
import time
import uuid
from typing import Any, Dict, Optional

import botocore.session
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class CloudWatchLogClient:
    """
    CloudWatch Logs client for batching and sending log events.

    This class handles the batching logic and CloudWatch Logs API interactions
    for sending EMF logs efficiently while respecting CloudWatch Logs constraints.
    """

    # Constants for CloudWatch Logs limits
    # http://docs.aws.amazon.com/AmazonCloudWatch/latest/logs/cloudwatch_limits_cwl.html
    CW_MAX_EVENT_PAYLOAD_BYTES = 256 * 1024  # 256KB
    CW_MAX_REQUEST_EVENT_COUNT = 10000
    CW_PER_EVENT_HEADER_BYTES = 26
    BATCH_FLUSH_INTERVAL = 60 * 1000
    CW_MAX_REQUEST_PAYLOAD_BYTES = 1 * 1024 * 1024  # 1MB
    CW_TRUNCATED_SUFFIX = "[Truncated...]"
    # None of the log events in the batch can be older than 14 days
    CW_EVENT_TIMESTAMP_LIMIT_PAST = 14 * 24 * 60 * 60 * 1000
    # None of the log events in the batch can be more than 2 hours in the future.
    CW_EVENT_TIMESTAMP_LIMIT_FUTURE = 2 * 60 * 60 * 1000

    def __init__(
        self,
        log_group_name: str,
        log_stream_name: Optional[str] = None,
        aws_region: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the CloudWatch Logs client.

        Args:
            log_group_name: CloudWatch log group name
            log_stream_name: CloudWatch log stream name (auto-generated if None)
            aws_region: AWS region (auto-detected if None)
            **kwargs: Additional arguments passed to botocore client
        """
        self.log_group_name = log_group_name
        self.log_stream_name = log_stream_name or self._generate_log_stream_name()

        session = botocore.session.Session()
        self.logs_client = session.create_client("logs", region_name=aws_region, **kwargs)

        # Event batch to store logs before sending to CloudWatch
        self._event_batch = None

    def _generate_log_stream_name(self) -> str:
        """Generate a unique log stream name."""
        unique_id = str(uuid.uuid4())[:8]
        return f"otel-python-{unique_id}"

    def _create_log_group_if_needed(self):
        """Create log group if it doesn't exist."""
        try:
            self.logs_client.create_log_group(logGroupName=self.log_group_name)
            logger.info("Created log group: %s", self.log_group_name)
        except ClientError as error:
            if error.response.get("Error", {}).get("Code") == "ResourceAlreadyExistsException":
                logger.debug("Log group %s already exists", self.log_group_name)
            else:
                logger.error("Failed to create log group %s : %s", self.log_group_name, error)
                raise

    def _create_log_stream_if_needed(self):
        """Create log stream if it doesn't exist."""
        try:
            self.logs_client.create_log_stream(logGroupName=self.log_group_name, logStreamName=self.log_stream_name)
            logger.info("Created log stream: %s", self.log_stream_name)
        except ClientError as error:
            if error.response.get("Error", {}).get("Code") == "ResourceAlreadyExistsException":
                logger.debug("Log stream %s already exists", self.log_stream_name)
            else:
                logger.error("Failed to create log stream %s : %s", self.log_stream_name, error)
                raise

    def _validate_log_event(self, log_event: Dict) -> bool:
        """
        Validate the log event according to CloudWatch Logs constraints.
        Implements the same validation logic as the Go version.

        Args:
            log_event: The log event to validate

        Returns:
            bool: True if valid, False otherwise
        """

        # Check empty message
        if not log_event.get("message"):
            logger.error("Empty log event message")
            return False

        message = log_event.get("message", "")
        timestamp = log_event.get("timestamp", 0)

        # Check message size
        message_size = len(message) + self.CW_PER_EVENT_HEADER_BYTES
        if message_size > self.CW_MAX_EVENT_PAYLOAD_BYTES:
            logger.warning(
                "Log event size %s exceeds maximum allowed size %s. Truncating.",
                message_size,
                self.CW_MAX_EVENT_PAYLOAD_BYTES,
            )
            max_message_size = (
                self.CW_MAX_EVENT_PAYLOAD_BYTES - self.CW_PER_EVENT_HEADER_BYTES - len(self.CW_TRUNCATED_SUFFIX)
            )
            log_event["message"] = message[:max_message_size] + self.CW_TRUNCATED_SUFFIX

        # Check timestamp constraints
        current_time = int(time.time() * 1000)  # Current time in milliseconds
        event_time = timestamp

        # Calculate the time difference
        time_diff = current_time - event_time

        # Check if too old or too far in the future
        if time_diff > self.CW_EVENT_TIMESTAMP_LIMIT_PAST or time_diff < -self.CW_EVENT_TIMESTAMP_LIMIT_FUTURE:
            logger.error(
                "Log event timestamp %s is either older than 14 days or more than 2 hours in the future. "
                "Current time: %s",
                event_time,
                current_time,
            )
            return False

        return True

    def _create_event_batch(self) -> Dict:
        """
        Create a new log event batch.

        Returns:
            Dict: A new event batch
        """
        return {
            "logEvents": [],
            "byteTotal": 0,
            "minTimestampMs": 0,
            "maxTimestampMs": 0,
            "createdTimestampMs": int(time.time() * 1000),
        }

    def _event_batch_exceeds_limit(self, batch: Dict, next_event_size: int) -> bool:
        """
        Check if adding the next event would exceed CloudWatch Logs limits.

        Args:
            batch: The current batch
            next_event_size: Size of the next event in bytes

        Returns:
            bool: True if adding the next event would exceed limits
        """
        return (
            len(batch["logEvents"]) >= self.CW_MAX_REQUEST_EVENT_COUNT
            or batch["byteTotal"] + next_event_size > self.CW_MAX_REQUEST_PAYLOAD_BYTES
        )

    def _is_batch_active(self, batch: Dict, target_timestamp_ms: int) -> bool:
        """
        Check if the event batch spans more than 24 hours.

        Args:
            batch: The event batch
            target_timestamp_ms: The timestamp of the event to add

        Returns:
            bool: True if the batch is active and can accept the event
        """
        # New log event batch
        if batch["minTimestampMs"] == 0 or batch["maxTimestampMs"] == 0:
            return True

        # Check if adding the event would make the batch span more than 24 hours
        if target_timestamp_ms - batch["minTimestampMs"] > 24 * 3600 * 1000:
            return False

        if batch["maxTimestampMs"] - target_timestamp_ms > 24 * 3600 * 1000:
            return False

        # flush the event batch when reached 60s interval
        current_time = int(time.time() * 1000)
        if current_time - batch["createdTimestampMs"] >= self.BATCH_FLUSH_INTERVAL:
            return False

        return True

    def _append_to_batch(self, batch: Dict, log_event: Dict, event_size: int) -> None:
        """
        Append a log event to the batch.

        Args:
            batch: The event batch
            log_event: The log event to append
            event_size: Size of the event in bytes
        """
        batch["logEvents"].append(log_event)
        batch["byteTotal"] += event_size

        timestamp = log_event["timestamp"]
        if batch["minTimestampMs"] == 0 or batch["minTimestampMs"] > timestamp:
            batch["minTimestampMs"] = timestamp

        if batch["maxTimestampMs"] == 0 or batch["maxTimestampMs"] < timestamp:
            batch["maxTimestampMs"] = timestamp

    def _sort_log_events(self, batch: Dict) -> None:
        """
        Sort log events in the batch by timestamp.

        Args:
            batch: The event batch
        """
        batch["logEvents"] = sorted(batch["logEvents"], key=lambda x: x["timestamp"])

    def _send_log_batch(self, batch: Dict) -> None:
        """
        Send a batch of log events to CloudWatch Logs.
        Creates log group and stream lazily if they don't exist.

        Args:
            batch: The event batch
        """
        if not batch["logEvents"]:
            return None

        # Sort log events by timestamp
        self._sort_log_events(batch)

        # Prepare the PutLogEvents request
        put_log_events_input = {
            "logGroupName": self.log_group_name,
            "logStreamName": self.log_stream_name,
            "logEvents": batch["logEvents"],
        }

        start_time = time.time()

        try:
            # Make the PutLogEvents call
            response = self.logs_client.put_log_events(**put_log_events_input)

            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.debug(
                "Successfully sent %s log events (%s KB) in %s ms",
                len(batch["logEvents"]),
                batch["byteTotal"] / 1024,
                elapsed_ms,
            )

            return response

        except ClientError as error:
            # Handle resource not found errors by creating log group/stream
            error_code = error.response.get("Error", {}).get("Code")
            if error_code == "ResourceNotFoundException":
                logger.info("Log group or stream not found, creating resources and retrying")

                try:
                    # Create log group first
                    self._create_log_group_if_needed()
                    # Then create log stream
                    self._create_log_stream_if_needed()

                    # Retry the PutLogEvents call
                    response = self.logs_client.put_log_events(**put_log_events_input)

                    elapsed_ms = int((time.time() - start_time) * 1000)
                    logger.debug(
                        "Successfully sent %s log events (%s KB) in %s ms after creating resources",
                        len(batch["logEvents"]),
                        batch["byteTotal"] / 1024,
                        elapsed_ms,
                    )

                    return response

                except ClientError as retry_error:
                    logger.error("Failed to send log events after creating resources: %s", retry_error)
                    raise
            else:
                logger.error("Failed to send log events: %s", error)
                raise

    def send_log_event(self, log_event: Dict[str, Any]):
        """
        Send a log event to CloudWatch Logs.

        This function implements the same logic as the Go version in the OTel Collector.
        It batches log events according to CloudWatch Logs constraints and sends them
        when the batch is full or spans more than 24 hours.

        Args:
            log_event: The log event to send
        """
        try:
            # Validate the log event
            if not self._validate_log_event(log_event):
                return

            # Calculate event size
            event_size = len(log_event["message"]) + self.CW_PER_EVENT_HEADER_BYTES

            # Initialize event batch if needed
            if self._event_batch is None:
                self._event_batch = self._create_event_batch()

            # Check if we need to send the current batch and create a new one
            current_batch = self._event_batch
            if self._event_batch_exceeds_limit(current_batch, event_size) or not self._is_batch_active(
                current_batch, log_event["timestamp"]
            ):
                # Send the current batch
                self._send_log_batch(current_batch)
                # Create a new batch
                self._event_batch = self._create_event_batch()
                current_batch = self._event_batch

            # Add the log event to the batch
            self._append_to_batch(current_batch, log_event, event_size)

        except Exception as error:
            logger.error("Failed to process log event: %s", error)
            raise

    def flush_pending_events(self) -> bool:
        """
        Flush any pending log events.

        Returns:
            True if successful, False otherwise
        """
        if self._event_batch is not None and len(self._event_batch["logEvents"]) > 0:
            current_batch = self._event_batch
            self._send_log_batch(current_batch)
            self._event_batch = self._create_event_batch()
        logger.debug("CloudWatchLogClient flushed the buffered log events")
        return True
