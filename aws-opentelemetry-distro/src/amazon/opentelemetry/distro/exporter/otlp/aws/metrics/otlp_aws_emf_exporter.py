"""
OpenTelemetry EMF (Embedded Metric Format) Exporter for CloudWatch.
This exporter converts OTel metrics into CloudWatch EMF format.
"""

import json
import os
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

import boto3
from opentelemetry.sdk.metrics import (
    Counter,
    Histogram,
    ObservableCounter,
    ObservableGauge,
    ObservableUpDownCounter,
    UpDownCounter,
)
from opentelemetry.sdk.metrics.export import (
    MetricExporter,
    MetricExportResult,
    MetricsData,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.metrics import Instrument
from opentelemetry.sdk.metrics.export import (
    AggregationTemporality,
)

logger = logging.getLogger(__name__)


class CloudWatchEMFExporter(MetricExporter):
    """
    OpenTelemetry metrics exporter for CloudWatch EMF format.

    This exporter converts OTel metrics into CloudWatch EMF logs which are then
    sent to CloudWatch Logs. CloudWatch Logs automatically extracts the metrics
    from the EMF logs.
    """

    # OTel to CloudWatch unit mapping
    UNIT_MAPPING = {
        "ms": "Milliseconds",
        "s": "Seconds",
        "us": "Microseconds",
        "ns": "Nanoseconds",
        "By": "Bytes",
        "KiBy": "Kilobytes",
        "MiBy": "Megabytes",
        "GiBy": "Gigabytes",
        "TiBy": "Terabytes",
        "Bi": "Bits",
        "KiBi": "Kilobits",
        "MiBi": "Megabits",
        "GiBi": "Gigabits",
        "TiBi": "Terabits",
        "%": "Percent",
        "1": "Count",
        "{count}": "Count",
    }

    def __init__(
        self,
        namespace: str = "default",
        log_group_name: str = None,
        log_stream_name: Optional[str] = None,
        aws_region: Optional[str] = None,
        metric_declarations: Optional[List[Dict]] = None,
        parse_json_encoded_attr_values: bool = True,
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
            metric_declarations: Optional metric declarations for filtering
            parse_json_encoded_attr_values: Whether to parse JSON-encoded attribute values
            preferred_temporality: Optional dictionary mapping instrument types to aggregation temporality
            **kwargs: Additional arguments passed to boto3 client
        """
        super().__init__(preferred_temporality)

        self.namespace = namespace
        self.log_group_name = log_group_name
        self.log_stream_name = log_stream_name or self._generate_log_stream_name()
        self.metric_declarations = metric_declarations or []
        self.parse_json_encoded_attr_values = parse_json_encoded_attr_values

        # Initialize CloudWatch Logs client
        # If aws_region is not provided, boto3 will check environment variables AWS_REGION or AWS_DEFAULT_REGION
        # Make sure we have a region specified somehow
        if aws_region is None:
            aws_region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")

        session = boto3.Session(region_name=aws_region)
        self.logs_client = session.client("logs", **kwargs)

        # Ensure log group exists
        self._ensure_log_group_exists()

    def _generate_log_stream_name(self) -> str:
        """Generate a unique log stream name."""
        import uuid

        unique_id = str(uuid.uuid4())[:8]
        return f"otel-python-{unique_id}"

    def _ensure_log_group_exists(self):
        """Ensure the log group exists, create if it doesn't."""
        try:
            self.logs_client.describe_log_groups(logGroupNamePrefix=self.log_group_name, limit=1)
        except Exception:
            try:
                self.logs_client.create_log_group(logGroupName=self.log_group_name)
                logger.info(f"Created log group: {self.log_group_name}")
            except Exception as e:
                logger.error(f"Failed to create log group {self.log_group_name}: {e}")
                raise

    def _get_metric_name(self, record) -> str:
        """Get the metric name from the metric record or data point."""
        # For metrics in MetricsData format
        if hasattr(record, "name"):
            return record.name
        # For compatibility with older record format
        elif hasattr(record, "instrument") and hasattr(record.instrument, "name"):
            return record.instrument.name
        else:
            # Fallback with generic name
            return "unknown_metric"

    def _get_unit(self, instrument_or_metric) -> Optional[str]:
        """Get CloudWatch unit from OTel instrument or metric unit."""
        # Check if we have an Instrument object or a metric with unit attribute
        if isinstance(instrument_or_metric, Instrument):
            unit = instrument_or_metric.unit
        else:
            unit = getattr(instrument_or_metric, "unit", None)

        if not unit:
            return None

        return self.UNIT_MAPPING.get(unit, unit)

    def _get_dimension_names(self, attributes: Dict[str, Any]) -> List[str]:
        """Extract dimension names from attributes."""
        # Implement dimension selection logic
        # For now, use all attributes as dimensions
        return list(attributes.keys())

    def _get_attributes_key(self, attributes: Dict[str, Any]) -> str:
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

    def _create_metric_record(self, metric_name: str, metric_unit: str, metric_description: str) -> Any:
        """Create a base metric record with instrument information.

        Args:
            metric_name: Name of the metric
            metric_unit: Unit of the metric
            metric_description: Description of the metric

        Returns:
            A base metric record object
        """
        record = type("MetricRecord", (), {})()
        record.instrument = type("Instrument", (), {})()
        record.instrument.name = metric_name
        record.instrument.unit = metric_unit
        record.instrument.description = metric_description

        return record

    def _convert_gauge(self, metric, dp) -> Tuple[Any, int]:
        """Convert a Gauge metric datapoint to a metric record.

        Args:
            metric: The metric object
            dp: The datapoint to convert

        Returns:
            Tuple of (metric record, timestamp in ms)
        """
        # Create base record
        record = self._create_metric_record(metric.name, metric.unit, metric.description)

        # Set timestamp
        timestamp_ms = (
            self._normalize_timestamp(dp.time_unix_nano) if hasattr(dp, "time_unix_nano") else int(time.time() * 1000)
        )
        record.timestamp = timestamp_ms

        # Set attributes
        record.attributes = dp.attributes

        # For Gauge, set the value directly
        record.value = dp.value

        return record, timestamp_ms

    def _convert_sum(self, metric, dp) -> Tuple[Any, int]:
        """Convert a Sum metric datapoint to a metric record.

        Args:
            metric: The metric object
            dp: The datapoint to convert

        Returns:
            Tuple of (metric record, timestamp in ms)
        """
        # Create base record
        record = self._create_metric_record(metric.name, metric.unit, metric.description)

        # Set timestamp
        timestamp_ms = (
            self._normalize_timestamp(dp.time_unix_nano) if hasattr(dp, "time_unix_nano") else int(time.time() * 1000)
        )
        record.timestamp = timestamp_ms

        # Set attributes
        record.attributes = dp.attributes

        # For Sum, set the sum_data
        record.sum_data = type('SumData', (), {})()
        record.sum_data.value = dp.value

        return record, timestamp_ms

    def _group_by_attributes_and_timestamp(self, record, timestamp_ms) -> Tuple[str, int]:
        """Group metric record by attributes and timestamp.

        Args:
            record: The metric record
            timestamp_ms: The timestamp in milliseconds

        Returns:
            A tuple key for grouping
        """
        # Create a key for grouping based on attributes
        attrs_key = self._get_attributes_key(record.attributes)
        return (attrs_key, timestamp_ms)

    def _create_emf_log(self, metric_records, resource: Resource, timestamp: Optional[int] = None) -> Dict:
        """
        Create EMF log dictionary from metric records.

        Since metric_records is already grouped by attributes, this function
        creates a single EMF log for all records.
        """
        # Start with base structure
        emf_log = {"_aws": {"Timestamp": timestamp or int(time.time() * 1000), "CloudWatchMetrics": []}}
        
        # Set with latest EMF version schema
        emf_log["Version"] = "1"

        # Add resource attributes to EMF log but not as dimensions
        if resource and resource.attributes:
            for key, value in resource.attributes.items():
                emf_log[f"resource.{key}"] = str(value)

        # Initialize collections for dimensions and metrics
        all_attributes = {}
        metric_definitions = []

        # Process each metric record
        for record in metric_records:
            # Collect attributes from all records (they should be the same for all records in the group)
            if hasattr(record, "attributes") and record.attributes:
                for key, value in record.attributes.items():
                    all_attributes[key] = value

            metric_name = self._get_metric_name(record)
            unit = self._get_unit(record.instrument)

            # Create metric data dict
            metric_data = {}
            if unit:
                metric_data["Unit"] = unit

            # Process different types of aggregations
            if hasattr(record, 'sum_data'):
                # Counter/UpDownCounter
                sum_data = record.sum_data
                # Store value directly in emf_log
                emf_log[metric_name] = sum_data.value
            else:
                # Other aggregations (e.g., LastValue)
                if hasattr(record, 'value'):
                    # Store value directly in emf_log
                    emf_log[metric_name] = record.value

            # Add to metric definitions list
            metric_definitions.append({"Name": metric_name, **metric_data})

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

    def export(self, metrics_data: MetricsData, timeout_millis: Optional[int] = None, **kwargs) -> MetricExportResult:
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
            # Import metric data types within the method to avoid circular imports
            from opentelemetry.sdk.metrics.export import Gauge, Sum

            if not metrics_data.resource_metrics:
                return MetricExportResult.SUCCESS

            # Process all metrics from all resource metrics and scope metrics
            for resource_metrics in metrics_data.resource_metrics:
                # The resource is now part of each resource_metrics object
                resource = resource_metrics.resource

                for scope_metrics in resource_metrics.scope_metrics:
                    # Dictionary to group metrics by attributes and timestamp
                    # Key: (attributes_key, timestamp_ms)
                    # Value: list of metric records
                    grouped_metrics = defaultdict(list)

                    # Process all metrics in this scope
                    for metric in scope_metrics.metrics:
                        # Convert metrics to a format compatible with _create_emf_log
                        # Access data points through metric.data.data_points
                        if hasattr(metric, "data") and hasattr(metric.data, "data_points"):
                            # Process different metric types
                            if isinstance(metric.data, Gauge):
                                for dp in metric.data.data_points:
                                    record, timestamp_ms = self._convert_gauge(metric, dp)
                                    group_key = self._group_by_attributes_and_timestamp(record, timestamp_ms)
                                    grouped_metrics[group_key].append(record)

                            elif isinstance(metric.data, Sum):
                                for dp in metric.data.data_points:
                                    record, timestamp_ms = self._convert_sum(metric, dp)
                                    group_key = self._group_by_attributes_and_timestamp(record, timestamp_ms)
                                    grouped_metrics[group_key].append(record)

                            else:
                                logger.warning("Unsupported Metric Type: %s", type(metric.data))
                                continue  # Skip this metric but continue processing others

                    # Now process each group separately to create one EMF log per group
                    for (attrs_key, timestamp_ms), metric_records in grouped_metrics.items():
                        if metric_records:
                            logger.debug(
                                f"Creating EMF log for group with {len(metric_records)} metrics. "
                                f"Timestamp: {timestamp_ms}, Attributes: {attrs_key[:100]}..."
                            )

                            # Create EMF log for this batch of metrics with the group's timestamp
                            emf_log_dict = self._create_emf_log(metric_records, resource, timestamp_ms)

                            # Convert to JSON
                            log_event = {"message": json.dumps(emf_log_dict), "timestamp": timestamp_ms}

                            # TODO: Send to CloudWatch Logs (will be implemented in PR 3)
                            logger.debug(f"Would send EMF log: {log_event}")

            return MetricExportResult.SUCCESS

        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return MetricExportResult.FAILURE

    def force_flush(self, timeout_millis: int = 10000) -> bool:
        """
        Force flush any pending metrics.

        Args:
            timeout_millis: Timeout in milliseconds

        Returns:
            True if successful, False otherwise
        """
        # TODO: Implement in PR 3
        logger.debug("CloudWatchEMFExporter force flushes the bufferred metrics")
        return True

    def shutdown(self, timeout_millis=None, **kwargs):
        """
        Shutdown the exporter.
        Override to handle timeout and other keyword arguments, but do nothing.

        Args:
            timeout_millis: Ignored timeout in milliseconds
            **kwargs: Ignored additional keyword arguments
        """
        # TODO: Implement in PR 3
        self.force_flush(timeout_millis)
        logger.debug(f"CloudWatchEMFExporter shutdown called with timeout_millis={timeout_millis}")
        return True


def create_emf_exporter(
    namespace: str = "OTelPython",
    log_group_name: str = "/aws/otel/python",
    log_stream_name: Optional[str] = None,
    aws_region: Optional[str] = None,
    debug: bool = False,
    **kwargs,
) -> CloudWatchEMFExporter:
    """
    Convenience function to create a CloudWatch EMF exporter with DELTA temporality.

    Args:
        namespace: CloudWatch namespace for metrics
        log_group_name: CloudWatch log group name
        log_stream_name: CloudWatch log stream name (auto-generated if None)
        aws_region: AWS region (auto-detected if None)
        debug: Whether to enable debug printing of EMF logs
        **kwargs: Additional arguments passed to the CloudWatchEMFExporter

    Returns:
        Configured CloudWatchEMFExporter instance
    """

    # Set up temporality preference - always use DELTA for CloudWatch
    temporality_dict = {
        Counter: AggregationTemporality.DELTA,
        ObservableCounter: AggregationTemporality.DELTA,
        ObservableGauge: AggregationTemporality.DELTA,
        ObservableUpDownCounter: AggregationTemporality.DELTA,
        UpDownCounter: AggregationTemporality.DELTA,
    }

    # Configure logging if debug is enabled
    if debug:
        logging.basicConfig(level=logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    # Create and return the exporter
    return CloudWatchEMFExporter(
        namespace=namespace,
        log_group_name=log_group_name,
        log_stream_name=log_stream_name,
        aws_region=aws_region,
        preferred_temporality=temporality_dict,
        **kwargs,
    )