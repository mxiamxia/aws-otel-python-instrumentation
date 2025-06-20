# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=no-self-use

import json
import logging
import time
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import botocore.session
from botocore.exceptions import ClientError

from opentelemetry.sdk.metrics import (
    Counter,
    ObservableCounter,
    ObservableGauge,
    ObservableUpDownCounter,
    UpDownCounter,
)
from opentelemetry.sdk.metrics._internal.point import Metric
from opentelemetry.sdk.metrics.export import (
    AggregationTemporality,
    Gauge,
    Sum,
    Histogram,
    ExponentialHistogram,
    MetricExporter,
    MetricExportResult,
    MetricsData,
    NumberDataPoint,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.util.types import Attributes

logger = logging.getLogger(__name__)


class MetricRecord:
    """The metric data unified representation of all OTel metrics for OTel to CW EMF conversion."""

    def __init__(self, metric_name: str, metric_unit: str, metric_description: str):
        """
        Initialize metric record.

        Args:
            metric_name: Name of the metric
            metric_unit: Unit of the metric
            metric_description: Description of the metric
        """
        # Instrument metadata
        self.name = metric_name
        self.unit = metric_unit
        self.description = metric_description

        # Will be set by conversion methods
        self.timestamp: Optional[int] = None
        self.attributes: Attributes = {}

        # Different metric type data - only one will be set per record
        self.value: Optional[float] = None
        self.sum_data: Optional[Any] = None
        self.histogram_data: Optional[Any] = None
        self.exp_histogram_data: Optional[Any] = None


class AwsCloudWatchEmfExporter(MetricExporter):
    """
    OpenTelemetry metrics exporter for CloudWatch EMF format.

    This exporter converts OTel metrics into CloudWatch EMF logs which are then
    sent to CloudWatch Logs. CloudWatch Logs automatically extracts the metrics
    from the EMF logs.

    https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/CloudWatch_Embedded_Metric_Format_Specification.html

    """

    # CloudWatch EMF supported units
    # Ref: https://docs.aws.amazon.com/AmazonCloudWatch/latest/APIReference/API_MetricDatum.html
    EMF_SUPPORTED_UNITS = {
        "Seconds",
        "Microseconds",
        "Milliseconds",
        "Bytes",
        "Kilobytes",
        "Megabytes",
        "Gigabytes",
        "Terabytes",
        "Bits",
        "Kilobits",
        "Megabits",
        "Gigabits",
        "Terabits",
        "Percent",
        "Count",
        "Bytes/Second",
        "Kilobytes/Second",
        "Megabytes/Second",
        "Gigabytes/Second",
        "Terabytes/Second",
        "Bits/Second",
        "Kilobits/Second",
        "Megabits/Second",
        "Gigabits/Second",
        "Terabits/Second",
        "Count/Second",
        "None",
    }

    # OTel to CloudWatch unit mapping
    # Ref: opentelemetry-collector-contrib/blob/main/exporter/awsemfexporter/grouped_metric.go#L188
    UNIT_MAPPING = {
        "1": "",
        "ns": "",
        "ms": "Milliseconds",
        "s": "Seconds",
        "us": "Microseconds",
        "By": "Bytes",
        "bit": "Bits",
    }

    def __init__(
        self,
        namespace: str = "default",
        log_group_name: str = None,
        log_stream_name: Optional[str] = None,
        aws_region: Optional[str] = None,
        preferred_temporality: Optional[Dict[type, AggregationTemporality]] = None,
        **kwargs,
    ):
        """
        Initialize the CloudWatch EMF exporter.

        Args:
            namespace: CloudWatch namespace for metrics
            log_group_name: CloudWatch log group name
            log_stream_name: CloudWatch log stream name (auto-generated if None)
            aws_region: AWS region (auto-detected if None)
            preferred_temporality: Optional dictionary mapping instrument types to aggregation temporality
            **kwargs: Additional arguments passed to botocore client
        """
        # Set up temporality preference default to DELTA if customers not set
        if preferred_temporality is None:
            preferred_temporality = {
                Counter: AggregationTemporality.DELTA,
                Histogram: AggregationTemporality.DELTA,
                ObservableCounter: AggregationTemporality.DELTA,
                ObservableGauge: AggregationTemporality.DELTA,
                ObservableUpDownCounter: AggregationTemporality.DELTA,
                UpDownCounter: AggregationTemporality.DELTA,
            }

        super().__init__(preferred_temporality)

        self.namespace = namespace
        self.log_group_name = log_group_name
        self.log_stream_name = log_stream_name or self._generate_log_stream_name()

        session = botocore.session.Session()
        self.logs_client = session.create_client("logs", region_name=aws_region, **kwargs)

        # Ensure log group exists
        self._ensure_log_group_exists()

        # Ensure log stream exists
        self._ensure_log_stream_exists()

    # Default to unique log stream name matching OTel Collector
    # EMF Exporter behavior with language for source identification
    def _generate_log_stream_name(self) -> str:
        """Generate a unique log stream name."""

        unique_id = str(uuid.uuid4())[:8]
        return f"otel-python-{unique_id}"

    def _ensure_log_group_exists(self):
        """Ensure the log group exists, create if it doesn't."""
        try:
            self.logs_client.create_log_group(logGroupName=self.log_group_name)
            logger.info("Created log group: %s", self.log_group_name)
        except ClientError as error:
            if error.response.get("Error", {}).get("Code") == "ResourceAlreadyExistsException":
                logger.debug("Log group %s already exists", self.log_group_name)
            else:
                logger.error("Failed to create log group %s : %s", self.log_group_name, error)
                raise

    def _ensure_log_stream_exists(self):
        try:
            self.logs_client.create_log_stream(logGroupName=self.log_group_name, logStreamName=self.log_stream_name)
            logger.info("Created log stream: %s", self.log_stream_name)
        except ClientError as error:
            if error.response.get("Error", {}).get("Code") == "ResourceAlreadyExistsException":
                logger.debug("Log stream %s already exists", self.log_stream_name)
            else:
                logger.error("Failed to create log stream %s : %s", self.log_group_name, error)
                raise

    def _get_metric_name(self, record: MetricRecord) -> Optional[str]:
        """Get the metric name from the metric record or data point."""

        try:
            if record.name:
                return record.name
        except AttributeError:
            pass
        # Return None if no valid metric name found
        return None

    def _get_unit(self, record: MetricRecord) -> Optional[str]:
        """Get CloudWatch unit from MetricRecord unit."""
        unit = record.unit

        if not unit:
            return None

        # First check if unit is already a supported EMF unit
        if unit in self.EMF_SUPPORTED_UNITS:
            return unit

        # Map from OTel unit to CloudWatch unit
        mapped_unit = self.UNIT_MAPPING.get(unit)

        return mapped_unit

    def _get_dimension_names(self, attributes: Attributes) -> List[str]:
        """Extract dimension names from attributes."""
        # Implement dimension selection logic
        # For now, use all attributes as dimensions
        return list(attributes.keys())

    def _get_attributes_key(self, attributes: Attributes) -> str:
        """
        Create a hashable key from attributes for grouping metrics.

        Args:
            attributes: The attributes dictionary

        Returns:
            A string representation of sorted attributes key-value pairs
        """
        # Sort the attributes to ensure consistent keys
        sorted_attrs = sorted(attributes.items())
        # Create a string representation of the attributes
        return str(sorted_attrs)

    def _normalize_timestamp(self, timestamp_ns: int) -> int:
        """
        Normalize a nanosecond timestamp to milliseconds for CloudWatch.

        Args:
            timestamp_ns: Timestamp in nanoseconds

        Returns:
            Timestamp in milliseconds
        """
        # Convert from nanoseconds to milliseconds
        return timestamp_ns // 1_000_000

    def _create_metric_record(self, metric_name: str, metric_unit: str, metric_description: str) -> MetricRecord:
        """
        Creates the intermediate metric data structure that standardizes different otel metric representation
        and will be used to generate EMF events. The base record
        establishes the instrument schema (name/unit/description) that will be populated
        with dimensions, timestamps, and values during metric processing.

        Args:
            metric_name: Name of the metric
            metric_unit: Unit of the metric
            metric_description: Description of the metric

        Returns:
            A MetricRecord object
        """
        return MetricRecord(metric_name, metric_unit, metric_description)

    def _convert_gauge(self, metric: Metric, data_point: NumberDataPoint) -> MetricRecord:
        """Convert a Gauge metric datapoint to a metric record.

        Args:
            metric: The metric object
            data_point: The datapoint to convert

        Returns:
            MetricRecord with populated timestamp, attributes, and value
        """
        # Create base record
        record = self._create_metric_record(metric.name, metric.unit, metric.description)

        # Set timestamp
        try:
            timestamp_ms = (
                self._normalize_timestamp(data_point.time_unix_nano)
                if data_point.time_unix_nano is not None
                else int(time.time() * 1000)
            )
        except AttributeError:
            # data_point doesn't have time_unix_nano attribute
            timestamp_ms = int(time.time() * 1000)
        record.timestamp = timestamp_ms

        # Set attributes
        try:
            record.attributes = data_point.attributes
        except AttributeError:
            # data_point doesn't have attributes
            record.attributes = {}

        # For Gauge, set the value directly
        try:
            record.value = data_point.value
        except AttributeError:
            # data_point doesn't have value
            record.value = None

        return record

    def _convert_sum(self, metric: Metric, data_point: NumberDataPoint) -> MetricRecord:
        """Convert a Sum metric datapoint to a metric record.

        Args:
            metric: The metric object
            data_point: The datapoint to convert

        Returns:
            MetricRecord with populated timestamp, attributes, and sum_data
        """
        # Create base record
        record = self._create_metric_record(metric.name, metric.unit, metric.description)

        # Set timestamp
        timestamp_ms = (
            self._normalize_timestamp(data_point.time_unix_nano) if hasattr(data_point, "time_unix_nano") else int(time.time() * 1000)
        )
        record.timestamp = timestamp_ms

        # Set attributes
        record.attributes = data_point.attributes

        # For Sum, set the sum_data
        record.sum_data = type('SumData', (), {})()
        record.sum_data.value = data_point.value

        return record

    def _convert_histogram(self, metric: Metric, data_point: Any) -> MetricRecord:
        """Convert a Histogram metric datapoint to a metric record.

        https://github.com/mircohacker/opentelemetry-collector-contrib/blob/main/exporter/awsemfexporter/datapoint.go#L148

        Args:
            metric: The metric object
            data_point: The datapoint to convert

        Returns:
            MetricRecord with populated timestamp, attributes, and histogram_data
        """
        # Create base record
        record = self._create_metric_record(metric.name, metric.unit, metric.description)

        # Set timestamp
        timestamp_ms = (
            self._normalize_timestamp(data_point.time_unix_nano) if hasattr(data_point, "time_unix_nano") else int(time.time() * 1000)
        )
        record.timestamp = timestamp_ms

        # Set attributes
        record.attributes = data_point.attributes

        # For Histogram, set the histogram_data
        record.histogram_data = type('Histogram', (), {})()
        record.histogram_data.value = {
            "Count": data_point.count,
            "Sum": data_point.sum,
            "Min": data_point.min,
            "Max": data_point.max
        }
        return record

    def _convert_exp_histogram(self, metric: Metric, data_point: Any) -> MetricRecord:
        """
        Convert an ExponentialHistogram metric datapoint to a metric record.

        This function follows the logic of CalculateDeltaDatapoints in the Go implementation,
        converting exponential buckets to their midpoint values.

        Ref:
            https://github.com/open-telemetry/opentelemetry-collector-contrib/issues/22626

        Args:
            metric: The metric object
            data_point: The datapoint to convert

        Returns:
            MetricRecord with populated timestamp, attributes, and exp_histogram_data
        """
        import math

        # Create base record
        record = self._create_metric_record(metric.name, metric.unit, metric.description)

        # Set timestamp
        timestamp_ms = (
            self._normalize_timestamp(data_point.time_unix_nano) if hasattr(data_point, "time_unix_nano") else int(time.time() * 1000)
        )
        record.timestamp = timestamp_ms

        # Set attributes
        record.attributes = data_point.attributes

        # Initialize arrays for values and counts
        array_values = []
        array_counts = []

        # Get scale
        scale = data_point.scale
        # Calculate base using the formula: 2^(2^(-scale))
        base = math.pow(2, math.pow(2, float(-scale)))

        # Process positive buckets
        if hasattr(data_point, "positive") and hasattr(data_point.positive, "bucket_counts") and data_point.positive.bucket_counts:
            positive_offset = getattr(data_point.positive, "offset", 0)
            positive_bucket_counts = data_point.positive.bucket_counts

            bucket_begin = 0
            bucket_end = 0

            for i, count in enumerate(positive_bucket_counts):
                index = i + positive_offset

                if bucket_begin == 0:
                    bucket_begin = math.pow(base, float(index))
                else:
                    bucket_begin = bucket_end

                bucket_end = math.pow(base, float(index + 1))

                # Calculate midpoint value of the bucket
                metric_val = (bucket_begin + bucket_end) / 2

                # Only include buckets with positive counts
                if count > 0:
                    array_values.append(metric_val)
                    array_counts.append(float(count))

        # Process zero bucket
        zero_count = getattr(data_point, "zero_count", 0)
        if zero_count > 0:
            array_values.append(0)
            array_counts.append(float(zero_count))

        # Process negative buckets
        if hasattr(data_point, "negative") and hasattr(data_point.negative, "bucket_counts") and data_point.negative.bucket_counts:
            negative_offset = getattr(data_point.negative, "offset", 0)
            negative_bucket_counts = data_point.negative.bucket_counts

            bucket_begin = 0
            bucket_end = 0

            for i, count in enumerate(negative_bucket_counts):
                index = i + negative_offset

                if bucket_end == 0:
                    bucket_end = -math.pow(base, float(index))
                else:
                    bucket_end = bucket_begin

                bucket_begin = -math.pow(base, float(index + 1))

                # Calculate midpoint value of the bucket
                metric_val = (bucket_begin + bucket_end) / 2

                # Only include buckets with positive counts
                if count > 0:
                    array_values.append(metric_val)
                    array_counts.append(float(count))

        # Set the histogram data in the format expected by CloudWatch EMF
        record.exp_histogram_data = type('ExpHistogram', (), {})()
        record.exp_histogram_data.value = {
            "Values": array_values,
            "Counts": array_counts,
            "Count": data_point.count,
            "Sum": data_point.sum,
            "Max": data_point.max,
            "Min": data_point.min
        }

        return record

    # Constants for CloudWatch Logs limits
    CW_MAX_EVENT_PAYLOAD_BYTES = 256 * 1024  # 256KB
    CW_MAX_REQUEST_EVENT_COUNT = 10000
    CW_PER_EVENT_HEADER_BYTES = 26
    BATCH_FLUSH_INTERVAL = 60 * 1000
    CW_MAX_REQUEST_PAYLOAD_BYTES = 1 * 1024 * 1024  # 1MB
    CW_TRUNCATED_SUFFIX = "[Truncated...]"
    CW_EVENT_TIMESTAMP_LIMIT_PAST = 14 * 24 * 60 * 60 * 1000  # 14 days in milliseconds
    CW_EVENT_TIMESTAMP_LIMIT_FUTURE = 2 * 60 * 60 * 1000  # 2 hours in milliseconds
    
    def _validate_log_event(self, log_event: Dict) -> bool:
        """
        Validate the log event according to CloudWatch Logs constraints.
        Implements the same validation logic as the Go version.

        Args:
            log_event: The log event to validate

        Returns:
            bool: True if valid, False otherwise
        """
        message = log_event.get("message", "")
        timestamp = log_event.get("timestamp", 0)

        # Check message size
        message_size = len(message) + self.CW_PER_EVENT_HEADER_BYTES
        if message_size > self.CW_MAX_EVENT_PAYLOAD_BYTES:
            logger.warning(
                "Log event size %s exceeds maximum allowed size %s. Truncating.",
                message_size, self.CW_MAX_EVENT_PAYLOAD_BYTES
            )
            max_message_size = (
                self.CW_MAX_EVENT_PAYLOAD_BYTES - self.CW_PER_EVENT_HEADER_BYTES - len(self.CW_TRUNCATED_SUFFIX)
            )
            log_event["message"] = message[:max_message_size] + self.CW_TRUNCATED_SUFFIX

        # Check empty message
        if not log_event.get("message"):
            logger.error("Empty log event message")
            return False

        # Check timestamp constraints
        current_time = int(time.time() * 1000)  # Current time in milliseconds
        event_time = timestamp

        # Calculate the time difference
        time_diff = current_time - event_time

        # Check if too old or too far in the future
        if time_diff > self.CW_EVENT_TIMESTAMP_LIMIT_PAST or time_diff < -self.CW_EVENT_TIMESTAMP_LIMIT_FUTURE:
            logger.error(
                "Log event timestamp %s is either older than 14 days or more than 2 hours in the future. "
                "Current time: %s", event_time, current_time
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

        Args:
            batch: The event batch
        """
        if not batch["logEvents"]:
            return

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
                len(batch['logEvents']), batch['byteTotal'] / 1024, elapsed_ms
            )

            return response

        except ClientError as error:
            logger.error("Failed to send log events: %s", error)
            raise

    # Event batch to store logs before sending to CloudWatch
    _event_batch = None

    def _group_by_attributes_and_timestamp(self, record: MetricRecord) -> Tuple[str, int]:
        """Group metric record by attributes and timestamp.

        Args:
            record: The metric record

        Returns:
            A tuple key for grouping
        """
        # Create a key for grouping based on attributes
        attrs_key = self._get_attributes_key(record.attributes)
        return (attrs_key, record.timestamp)

    def _create_emf_log(
        self, metric_records: List[MetricRecord], resource: Resource, timestamp: Optional[int] = None
    ) -> Dict:
        """
        Create EMF log dictionary from metric records.

        Since metric_records is already grouped by attributes, this function
        creates a single EMF log for all records.
        """
        # Start with base structure
        emf_log = {"_aws": {"Timestamp": timestamp or int(time.time() * 1000), "CloudWatchMetrics": []}}

        # Set with latest EMF version schema
        # opentelemetry-collector-contrib/blob/main/exporter/awsemfexporter/metric_translator.go#L414
        emf_log["Version"] = "1"

        # Add resource attributes to EMF log but not as dimensions
        # OTel collector EMF Exporter has a resource_to_telemetry_conversion flag that will convert resource attributes
        # as regular metric attributes(potential dimensions). However, for this SDK EMF implementation,
        # we align with the OpenTelemetry concept that all metric attributes are treated as dimensions.
        # And have resource attributes as just additional metadata in EMF, added otel.resource as prefix to distinguish.
        if resource and resource.attributes:
            for key, value in resource.attributes.items():
                emf_log[f"otel.resource.{key}"] = str(value)

        # Initialize collections for dimensions and metrics
        metric_definitions = []
        # Collect attributes from all records (they should be the same for all records in the group)
        # Only collect once from the first record and apply to all records
        all_attributes = metric_records[0].attributes if metric_records and metric_records[0].attributes else {}

        # Process each metric record
        for record in metric_records:

            metric_name = self._get_metric_name(record)

            # Skip processing if metric name is None or empty
            if not metric_name:
                continue

            # Create metric data dict
            metric_data = {"Name": metric_name}

            unit = self._get_unit(record)
            if unit:
                metric_data["Unit"] = unit

            # Process different types of aggregations
            if hasattr(record, 'exp_histogram_data') and record.exp_histogram_data:
                # Base2 Exponential Histogram
                exp_histogram = record.exp_histogram_data
                # Store value directly in emf_log
                emf_log[metric_name] = exp_histogram.value
            elif hasattr(record, 'histogram_data') and record.histogram_data:
                # Regular Histogram metrics
                histogram_data = record.histogram_data
                # Store value directly in emf_log
                emf_log[metric_name] = histogram_data.value
            elif hasattr(record, 'sum_data') and record.sum_data:
                # Counter/UpDownCounter
                sum_data = record.sum_data
                # Store value directly in emf_log
                emf_log[metric_name] = sum_data.value
            elif record.value is not None:
                # Other aggregations (e.g., LastValue/Gauge)
                emf_log[metric_name] = record.value
            else:
                logger.debug("Skipping metric %s as it does not have valid metric value", metric_name)
                continue

            # Add to metric definitions list
            metric_definitions.append(metric_data)

        # Get dimension names from collected attributes
        dimension_names = self._get_dimension_names(all_attributes)

        # Add attribute values to the root of the EMF log
        for name, value in all_attributes.items():
            emf_log[name] = str(value)

        # Add the single dimension set to CloudWatch Metrics if we have dimensions and metrics
        if dimension_names and metric_definitions:
            emf_log["_aws"]["CloudWatchMetrics"].append(
                {"Namespace": self.namespace, "Dimensions": [dimension_names], "Metrics": metric_definitions}
            )

        return emf_log

    def _send_log_event(self, log_event: Dict[str, Any]):
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

    # pylint: disable=too-many-nested-blocks,unused-argument
    def export(
        self, metrics_data: MetricsData, timeout_millis: Optional[int] = None, **_kwargs: Any
    ) -> MetricExportResult:
        """
        Export metrics as EMF logs to CloudWatch.

        Groups metrics by attributes and timestamp before creating EMF logs.

        Args:
            metrics_data: MetricsData containing resource metrics and scope metrics
            timeout_millis: Optional timeout in milliseconds
            **kwargs: Additional keyword arguments

        Returns:
            MetricExportResult indicating success or failure
        """
        try:
            if not metrics_data.resource_metrics:
                return MetricExportResult.SUCCESS

            # Process all metrics from all resource metrics and scope metrics
            for resource_metrics in metrics_data.resource_metrics:
                for scope_metrics in resource_metrics.scope_metrics:
                    # Dictionary to group metrics by attributes and timestamp
                    grouped_metrics = defaultdict(list)

                    # Process all metrics in this scope
                    for metric in scope_metrics.metrics:
                        # Skip if metric.data is None or no data_points exists
                        try:
                            if not (metric.data and metric.data.data_points):
                                continue
                        except AttributeError:
                            # Metric doesn't have data or data_points attribute
                            continue

                        # Process metrics based on type
                        metric_type = type(metric.data)
                        if metric_type == Gauge:
                            for dp in metric.data.data_points:
                                record = self._convert_gauge(metric, dp)
                                grouped_metrics[self._group_by_attributes_and_timestamp(record)].append(record)
                        elif metric_type == Sum:
                            for dp in metric.data.data_points:
                                record = self._convert_sum(metric, dp)
                                grouped_metrics[self._group_by_attributes_and_timestamp(record)].append(record)
                        elif metric_type == Histogram:
                            for dp in metric.data.data_points:
                                record = self._convert_histogram(metric, dp)
                                grouped_metrics[self._group_by_attributes_and_timestamp(record)].append(record)
                        elif metric_type == ExponentialHistogram:
                            for dp in metric.data.data_points:
                                record = self._convert_exp_histogram(metric, dp)
                                grouped_metrics[self._group_by_attributes_and_timestamp(record)].append(record)
                        else:
                            logger.debug("Unsupported Metric Type: %s", metric_type)

                    # Now process each group separately to create one EMF log per group
                    for (_, timestamp_ms), metric_records in grouped_metrics.items():
                        if not metric_records:
                            continue

                        # Create and send EMF log for this batch of metrics
                        self._send_log_event(
                            {
                                "message": json.dumps(
                                    self._create_emf_log(metric_records, resource_metrics.resource, timestamp_ms)
                                ),
                                "timestamp": timestamp_ms,
                            }
                        )

            return MetricExportResult.SUCCESS
        # pylint: disable=broad-exception-caught
        # capture all types of exceptions to not interrupt the instrumented services
        except Exception as error:
            logger.error("Failed to export metrics: %s", error)
            return MetricExportResult.FAILURE

    def force_flush(self, timeout_millis: int = 10000) -> bool:  # pylint: disable=unused-argument
        """
        Force flush any pending metrics.

        Args:
            timeout_millis: Timeout in milliseconds

        Returns:
            True if successful, False otherwise
        """
        if self._event_batch is not None and len(self._event_batch["logEvents"]) > 0:
            current_batch = self._event_batch
            self._send_log_batch(current_batch)
            self._event_batch = self._create_event_batch()
        logger.debug("AwsCloudWatchEMFExporter force flushes the buffered metrics")
        return True

    def shutdown(self, timeout_millis: Optional[int] = None, **_kwargs: Any) -> bool:
        """
        Shutdown the exporter.
        Override to handle timeout and other keyword arguments, but do nothing.

        Args:
            timeout_millis: Ignored timeout in milliseconds
            **kwargs: Ignored additional keyword arguments
        """
        # Force flush any remaining batched events
        self.force_flush(timeout_millis)
        logger.debug("AwsCloudWatchEmfExporter shutdown called with timeout_millis=%s", timeout_millis)
        return True