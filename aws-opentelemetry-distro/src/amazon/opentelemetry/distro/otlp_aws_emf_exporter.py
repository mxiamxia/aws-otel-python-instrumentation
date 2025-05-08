"""
OpenTelemetry EMF (Embedded Metric Format) Exporter for CloudWatch.
This exporter converts OTel metrics into CloudWatch EMF format.
"""

import json
import time
import logging
from typing import Dict, List, Optional, Any, Union
from collections import defaultdict
from dataclasses import dataclass, field

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
from opentelemetry.util.types import Attributes

logger = logging.getLogger(__name__)


@dataclass
class EMFMetricData:
    """Represents an EMF metric data point."""
    unit: Optional[str] = None
    timestamp: Optional[int] = None
    values: List[Union[int, float]] = field(default_factory=list)
    counts: List[int] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "unit": self.unit,
            "values": self.values,
            "counts": self.counts,
        }


@dataclass
class EMFLog:
    """Represents a complete EMF log entry."""
    version: str = "0"
    dimensions: List[List[str]] = field(default_factory=list)
    metrics: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    _aws: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert EMF log to dictionary format."""
        result = {
            "_aws": self._aws,
            **self.metadata,
        }
        
        if self.dimensions:
            metrics_list = []
            for metric_dict in self.metrics:
                # Convert each metric dict with EMFMetricData to serializable dict
                serialized_metric = {}
                for metric_name, metric_data in metric_dict.items():
                    if isinstance(metric_data, EMFMetricData):
                        serialized_metric[metric_name] = metric_data.to_dict()
                    else:
                        serialized_metric[metric_name] = metric_data
                metrics_list.append(serialized_metric)
            
            result["_aws"]["CloudWatchMetrics"] = [{
                "Namespace": self._aws.get("Namespace", "default"),
                "Dimensions": self.dimensions,
                "Metrics": metrics_list,
            }]
        
        return result


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
        namespace: str = "OTelPython",
        log_group_name: str = "/aws/otel/python",
        log_stream_name: Optional[str] = None,
        aws_region: Optional[str] = None,
        dimension_rollup_option: str = "NoDimensionRollup",
        metric_declarations: Optional[List[Dict]] = None,
        parse_json_encoded_attr_values: bool = True,
        preferred_temporality: Optional[Dict[type, AggregationTemporality]] = None,
        **kwargs
    ):
        """
        Initialize the CloudWatch EMF exporter.
        
        Args:
            namespace: CloudWatch namespace for metrics
            log_group_name: CloudWatch log group name
            log_stream_name: CloudWatch log stream name (auto-generated if None)
            aws_region: AWS region (auto-detected if None)
            dimension_rollup_option: Dimension rollup behavior
            metric_declarations: Optional metric declarations for filtering
            parse_json_encoded_attr_values: Whether to parse JSON-encoded attribute values
            preferred_temporality: Optional dictionary mapping instrument types to aggregation temporality
            **kwargs: Additional arguments passed to boto3 client
        """
        super().__init__()
        
        self.namespace = namespace
        self.log_group_name = log_group_name
        self.log_stream_name = log_stream_name or self._generate_log_stream_name()
        self.dimension_rollup_option = dimension_rollup_option
        self.metric_declarations = metric_declarations or []
        self.parse_json_encoded_attr_values = parse_json_encoded_attr_values
        
        # Default to DELTA temporality for all instrument types if not specified
        if preferred_temporality is None:
            # Import all instrument types
            from opentelemetry.sdk.metrics import (
                Counter, Histogram, ObservableCounter, 
                ObservableGauge, ObservableUpDownCounter, UpDownCounter
            )
            self._preferred_temporality = {
                Counter: AggregationTemporality.DELTA,
                Histogram: AggregationTemporality.DELTA,
                ObservableCounter: AggregationTemporality.DELTA,
                ObservableGauge: AggregationTemporality.DELTA,
                ObservableUpDownCounter: AggregationTemporality.DELTA,
                UpDownCounter: AggregationTemporality.DELTA
            }
        else:
            self._preferred_temporality = preferred_temporality
        
        # Initialize CloudWatch Logs client
        session = boto3.Session(region_name=aws_region)
        self.logs_client = session.client("logs", **kwargs)
        
        # Ensure log group exists
        self._ensure_log_group_exists()
    
    def preferred_temporality(self, instrument_type):
        """
        Return the preferred aggregation temporality for the instrument type.
        
        Args:
            instrument_type: Type of the instrument
            
        Returns:
            The preferred AggregationTemporality for the instrument type
        """
        # Return the configured temporality for this instrument type
        # Default to DELTA if not specified
        return self._preferred_temporality.get(instrument_type, AggregationTemporality.DELTA)
    
    def _generate_log_stream_name(self) -> str:
        """Generate a unique log stream name."""
        import socket
        import uuid
        
        hostname = socket.gethostname()
        unique_id = str(uuid.uuid4())[:8]
        return f"otel-python-{hostname}-{unique_id}"
    
    def _ensure_log_group_exists(self):
        """Ensure the log group exists, create if it doesn't."""
        try:
            self.logs_client.describe_log_groups(
                logGroupNamePrefix=self.log_group_name,
                limit=1
            )
        except Exception:
            try:
                self.logs_client.create_log_group(
                    logGroupName=self.log_group_name
                )
                logger.info(f"Created log group: {self.log_group_name}")
            except Exception as e:
                logger.error(f"Failed to create log group {self.log_group_name}: {e}")
                raise
    
    def _get_metric_name(self, record) -> str:
        """Get the metric name from the metric record or data point."""
        # For metrics in MetricsData format
        if hasattr(record, 'name'):
            return record.name
        # For compatibility with older record format
        elif hasattr(record, 'instrument') and hasattr(record.instrument, 'name'):
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
            unit = getattr(instrument_or_metric, 'unit', None)
            
        if not unit:
            return None
        
        return self.UNIT_MAPPING.get(unit, unit)
    
    def _parse_attributes(self, attributes: Attributes) -> Dict[str, Any]:
        """Parse and process metric attributes."""
        parsed = {}
        
        for key, value in attributes.items():
            # Handle JSON-encoded values if enabled
            if self.parse_json_encoded_attr_values and isinstance(value, str):
                try:
                    # Try to parse as JSON
                    if value.startswith('{') or value.startswith('['):
                        parsed_value = json.loads(value)
                        if isinstance(parsed_value, dict):
                            # Flatten nested dictionaries
                            for nested_key, nested_value in parsed_value.items():
                                new_key = f"{key}.{nested_key}"
                                parsed[new_key] = str(nested_value)
                        else:
                            parsed[key] = str(value)
                    else:
                        parsed[key] = str(value)
                except json.JSONDecodeError:
                    parsed[key] = str(value)
            else:
                parsed[key] = str(value)
        
        return parsed
    
    def _get_dimension_names(self, attributes: Dict[str, Any]) -> List[str]:
        """Extract dimension names from attributes."""
        # Implement dimension selection logic
        # For now, use all attributes as dimensions
        return list(attributes.keys())
    
    def _match_metric_declaration(self, metric_name: str, attributes: Dict[str, Any]) -> Optional[Dict]:
        """Match metric against metric declarations."""
        for declaration in self.metric_declarations:
            # Check if metric name matches
            if "include" in declaration and metric_name in declaration["include"]:
                return declaration
            if "exclude" in declaration and metric_name in declaration["exclude"]:
                continue
        
        return None
    
    def _create_emf_log(self, metric_records, resource: Resource) -> Dict:
        """Create EMF log dictionary from metric records."""
        # Start with base structure
        emf_log = {
            "_aws": {
                "Namespace": self.namespace,
                "Timestamp": int(time.time() * 1000),
                "CloudWatchMetrics": []
            }
        }
        
        # Group metrics by dimensions
        dimension_groups = defaultdict(list)
        dimension_values = defaultdict(dict)
        
        for record in metric_records:
            metric_name = self._get_metric_name(record)
            unit = self._get_unit(record.instrument)
            
            # Process attributes
            attributes = self._parse_attributes(record.attributes or {})
            
            # Add resource attributes
            if resource and resource.attributes:
                for key, value in resource.attributes.items():
                    if key not in attributes:
                        attributes[f"resource.{key}"] = str(value)
            
            # Get dimensions for this metric
            dimension_names = self._get_dimension_names(attributes)
            dimension_key = tuple(sorted(dimension_names))
            
            # Store dimension values
            for name in dimension_names:
                dimension_values[dimension_key][name] = attributes.get(name, "")
            
            # Create metric data dict directly (no custom class)
            metric_data = {}
            if unit:
                metric_data["Unit"] = unit
            
            # Process different types of aggregations
            if hasattr(record, 'histogram_data'):
                # Histogram
                histogram = record.histogram_data
                if histogram.count > 0:
                    values = list(histogram.bucket_boundaries) if hasattr(histogram, 'bucket_boundaries') else []
                    counts = list(histogram.bucket_counts) if hasattr(histogram, 'bucket_counts') else []
                    # Store values directly in emf_log
                    emf_log[metric_name] = values[0] if values else 0
                    metric_data["Values"] = values
                    metric_data["Counts"] = counts
            elif hasattr(record, 'sum_data'):
                # Counter/UpDownCounter
                sum_data = record.sum_data
                # Store value directly in emf_log
                emf_log[metric_name] = sum_data.value
                metric_data["Value"] = sum_data.value
            else:
                # Other aggregations (e.g., LastValue)
                if hasattr(record, 'value'):
                    # Store value directly in emf_log
                    emf_log[metric_name] = record.value
                    metric_data["Value"] = record.value
            
            # Add to dimension group with the metric definition
            dimension_groups[dimension_key].append({
                "Name": metric_name,
                **metric_data
            })
        
        # Build CloudWatch Metrics structure
        for dimension_key, metrics in dimension_groups.items():
            # Add CloudWatch Metrics entry
            emf_log["_aws"]["CloudWatchMetrics"].append({
                "Namespace": self.namespace,
                "Dimensions": [list(dimension_key)],
                "Metrics": metrics
            })
            
            # Add dimension values to the root of the EMF log
            for name, value in dimension_values[dimension_key].items():
                emf_log[name] = value
        
        return emf_log
    
    def export(self, metrics_data: MetricsData, timeout_millis: Optional[int] = None, **kwargs) -> MetricExportResult:
        """
        Export metrics as EMF logs to CloudWatch.
        
        Args:
            metrics_data: MetricsData containing resource metrics and scope metrics
            timeout_millis: Optional timeout in milliseconds
            **kwargs: Additional keyword arguments
            
        Returns:
            MetricExportResult indicating success or failure
        """
        try:
            logger.debug("Starting to export metrics data")
            if not metrics_data.resource_metrics:
                return MetricExportResult.SUCCESS
            
            # Process all metrics from all resource metrics and scope metrics
            for resource_metrics in metrics_data.resource_metrics:
                # The resource is now part of each resource_metrics object
                resource = resource_metrics.resource
                
                for scope_metrics in resource_metrics.scope_metrics:
                    metric_records = []
                    
                    # Process all metrics in this scope
                    for metric in scope_metrics.metrics:
                        # Convert metrics to a format compatible with _create_emf_log
                        # Access data points through metric.data.data_points
                        if hasattr(metric, 'data') and hasattr(metric.data, 'data_points'):
                            for data_point in metric.data.data_points:
                                # Create a record-like object for compatibility
                                record = type('MetricRecord', (), {})()
                                record.instrument = type('Instrument', (), {})()
                                record.instrument.name = metric.name
                                record.instrument.unit = metric.unit
                                record.instrument.description = metric.description
                                
                                # Set attributes
                                record.attributes = data_point.attributes
                                
                                # Set appropriate data based on metric type
                                if hasattr(data_point, 'value'):
                                    # Sum data (Counter, UpDownCounter)
                                    record.sum_data = type('SumData', (), {})()
                                    record.sum_data.value = data_point.value
                                elif hasattr(data_point, 'bucket_counts'):
                                    # Histogram data
                                    record.histogram_data = type('HistogramData', (), {})()
                                    record.histogram_data.count = data_point.count
                                    record.histogram_data.sum = getattr(data_point, 'sum', 0)
                                    record.histogram_data.bucket_counts = data_point.bucket_counts
                                    record.histogram_data.bucket_boundaries = data_point.explicit_bounds
                                else:
                                    # Last value data (Gauge)
                                    record.value = getattr(data_point, 'value', 0)
                                
                                metric_records.append(record)
                    
                    if metric_records:
                        # Create EMF log for this batch of metrics - now returns a dict directly
                        emf_log_dict = self._create_emf_log(metric_records, resource)
                        
                        # Convert to JSON - no need for to_dict() conversion
                        log_event = {
                            "message": json.dumps(emf_log_dict),
                            "timestamp": int(time.time() * 1000)
                        }
                        logger.info(f"EMF Log Event: {log_event['message']}")
                        # Send to CloudWatch Logs
                        self._send_log_event(log_event)
            
            return MetricExportResult.SUCCESS
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return MetricExportResult.FAILURE
    
    def _send_log_event(self, log_event: Dict):
        """Send a log event to CloudWatch Logs."""
        try:
            # First check if log group exists and create it if needed
            try:
                self.logs_client.create_log_group(
                    logGroupName=self.log_group_name
                )
                logger.info(f"Created log group: {self.log_group_name}")
            except self.logs_client.exceptions.ResourceAlreadyExistsException:
                # Log group already exists, this is fine
                logger.debug(f"Log group {self.log_group_name} already exists")
            
            # Then check if log stream exists and create it if needed
            try:
                self.logs_client.create_log_stream(
                    logGroupName=self.log_group_name,
                    logStreamName=self.log_stream_name
                )
                logger.info(f"Created log stream: {self.log_stream_name}")
            except self.logs_client.exceptions.ResourceAlreadyExistsException:
                # Log stream already exists, this is fine
                logger.debug(f"Log stream {self.log_stream_name} already exists")
            
            # Put log event
            response = self.logs_client.put_log_events(
                logGroupName=self.log_group_name,
                logStreamName=self.log_stream_name,
                logEvents=[log_event]
            )
            logger.debug(f"CloudWatch PutLogEvents response: {response}")
            
        except Exception as e:
            logger.error(f"Failed to send log event: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def force_flush(self, timeout_millis: int = 10000) -> bool:
        """
        Force flush any pending metrics.
        
        Args:
            timeout_millis: Timeout in milliseconds
            
        Returns:
            True if successful, False otherwise
        """
        # No buffering in this implementation, so always return True
        return True
    
    def shutdown(self):
        """Shutdown the exporter."""
        pass


def create_emf_exporter(
    namespace: str = "OTelPython",
    log_group_name: str = "/aws/otel/python",
    log_stream_name: Optional[str] = None,
    aws_region: Optional[str] = None,
    debug: bool = False,
    **kwargs
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
    # Import all instrument types for temporality dictionary
    from opentelemetry.sdk.metrics import (
        Counter, Histogram, ObservableCounter, 
        ObservableGauge, ObservableUpDownCounter, UpDownCounter
    )
    
    # Set up temporality preference - always use DELTA for CloudWatch
    temporality_dict = {
        Counter: AggregationTemporality.DELTA,
        Histogram: AggregationTemporality.DELTA,
        ObservableCounter: AggregationTemporality.DELTA,
        ObservableGauge: AggregationTemporality.DELTA,
        ObservableUpDownCounter: AggregationTemporality.DELTA,
        UpDownCounter: AggregationTemporality.DELTA
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
        **kwargs
    )


# Example usage
if __name__ == "__main__":
    import random
    import time
    from opentelemetry import metrics
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    
    # Setup basic logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Create a resource
    resource = Resource.create({
        "service.name": "my-service",
        "service.version": "0.1.0",
        "deployment.environment": "production"
    })
    
    # Create EMF exporter with the helper function
    emf_exporter = create_emf_exporter(
        namespace="MyApplication",
        log_group_name="/aws/otel/my-app",
        aws_region="us-east-1",
        debug=True
    )
    
    # Create metric reader
    metric_reader = PeriodicExportingMetricReader(
        exporter=emf_exporter,
        export_interval_millis=5000  # Export every 5 seconds for testing
    )
    
    # Create meter provider with resource
    meter_provider = MeterProvider(
        metric_readers=[metric_reader],
        resource=resource
    )
    
    # Set global meter provider
    metrics.set_meter_provider(meter_provider)
    
    # Create a meter
    meter = meter_provider.get_meter("my-app-meter")
    
    # Create some metrics
    request_counter = meter.create_counter(
        name="http_requests_total",
        description="Total HTTP requests",
        unit="1"
    )
    
    request_duration = meter.create_histogram(
        name="http_request_duration_seconds",
        description="HTTP request duration",
        unit="s"
    )
    
    # Use the metrics in a loop to simulate traffic
    print("Generating metrics. Press Ctrl+C to stop...")
    try:
        while True:
            # Simulate HTTP GET request
            request_counter.add(1, {"method1": "GET", "status": "200"})
            request_duration.record(0.1 + (0.5 * random.random()), {"method": "GET", "status": "200"})
            
            # Simulate HTTP POST request
            request_counter.add(1, {"method": "POST", "status": "201"})
            request_duration.record(0.2 + (0.7 * random.random()), {"method": "POST", "status": "201"})
            
            # Simulate some errors
            if random.random() < 0.1:  # 10% error rate
                request_counter.add(1, {"method3": "GET", "status": "500"})
                request_duration.record(1.0 + (1.0 * random.random()), {"method": "GET", "status": "500"})
            
            # Sleep between 100ms and 300ms
            time.sleep(0.1 + (0.2 * random.random()))
    except KeyboardInterrupt:
        print("\nStopping metric generation.")
        # Force flush metrics
        print("Flushing metrics...")
        metric_reader.force_flush()