# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import time
import unittest
from unittest.mock import Mock, patch

from botocore.exceptions import ClientError

from amazon.opentelemetry.distro.exporter.otlp.aws.metrics.aws_cloudwatch_emf_exporter import AwsCloudWatchEMFExporter
from opentelemetry.sdk.metrics.export import Gauge, Sum, Histogram, MetricExportResult
from opentelemetry.sdk.resources import Resource


class MockDataPoint:
    """Mock datapoint for testing."""

    def __init__(self, value=10.0, attributes=None, time_unix_nano=None):
        self.value = value
        self.attributes = attributes or {}
        self.time_unix_nano = time_unix_nano or int(time.time() * 1_000_000_000)


class MockMetric:
    """Mock metric for testing."""

    def __init__(self, name="test_metric", unit="1", description="Test metric"):
        self.name = name
        self.unit = unit
        self.description = description


class MockHistogramDataPoint(MockDataPoint):
    """Mock histogram datapoint for testing."""
    def __init__(self, count=5, sum_val=25.0, min_val=1.0, max_val=10.0, **kwargs):
        super().__init__(**kwargs)
        self.count = count
        self.sum = sum_val
        self.min = min_val
        self.max = max_val


class MockExpHistogramDataPoint(MockDataPoint):
    """Mock exponential histogram datapoint for testing."""
    def __init__(self, count=10, sum_val=50.0, min_val=1.0, max_val=20.0, scale=2, **kwargs):
        super().__init__(**kwargs)
        self.count = count
        self.sum = sum_val
        self.min = min_val
        self.max = max_val
        self.scale = scale
        
        # Mock positive buckets
        self.positive = Mock()
        self.positive.offset = 0
        self.positive.bucket_counts = [1, 2, 3, 4]
        
        # Mock negative buckets
        self.negative = Mock()
        self.negative.offset = 0
        self.negative.bucket_counts = []
        
        # Mock zero count
        self.zero_count = 0


class MockGaugeData:
    """Mock gauge data that passes isinstance checks."""

    def __init__(self, data_points=None):
        self.data_points = data_points or []


class MockSumData:
    """Mock sum data that passes isinstance checks."""

    def __init__(self, data_points=None):
        self.data_points = data_points or []


class MockHistogramData:
    """Mock histogram data that passes isinstance checks."""

    def __init__(self, data_points=None):
        self.data_points = data_points or []


class MockExpHistogramData:
    """Mock exponential histogram data that passes isinstance checks."""

    def __init__(self, data_points=None):
        self.data_points = data_points or []


class MockMetricWithData:
    """Mock metric with data attribute."""

    def __init__(self, name="test_metric", unit="1", description="Test metric", data=None):
        self.name = name
        self.unit = unit
        self.description = description
        self.data = data or MockGaugeData()


class MockResourceMetrics:
    """Mock resource metrics for testing."""

    def __init__(self, resource=None, scope_metrics=None):
        self.resource = resource or Resource.create({"service.name": "test-service"})
        self.scope_metrics = scope_metrics or []


class MockScopeMetrics:
    """Mock scope metrics for testing."""

    def __init__(self, scope=None, metrics=None):
        self.scope = scope or Mock()
        self.metrics = metrics or []


# pylint: disable=too-many-public-methods
class TestAwsCloudWatchEMFExporter(unittest.TestCase):
    """Test AwsCloudWatchEMFExporter class."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock the botocore session to avoid AWS calls
        with patch("botocore.session.Session") as mock_session:
            mock_client = Mock()
            mock_session_instance = Mock()
            mock_session.return_value = mock_session_instance
            mock_session_instance.create_client.return_value = mock_client
            mock_client.create_log_group.return_value = {}
            mock_client.create_log_stream.return_value = {}

            self.exporter = AwsCloudWatchEMFExporter(namespace="TestNamespace", log_group_name="test-log-group")

    def test_initialization(self):
        """Test exporter initialization."""
        self.assertEqual(self.exporter.namespace, "TestNamespace")
        self.assertIsNotNone(self.exporter.log_stream_name)
        self.assertEqual(self.exporter.log_group_name, "test-log-group")

    @patch("botocore.session.Session")
    def test_initialization_with_custom_params(self, mock_session):
        """Test exporter initialization with custom parameters."""
        # Mock the botocore session to avoid AWS calls
        mock_client = Mock()
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.create_client.return_value = mock_client
        mock_client.create_log_group.return_value = {}
        mock_client.create_log_stream.return_value = {}

        exporter = AwsCloudWatchEMFExporter(
            namespace="CustomNamespace",
            log_group_name="custom-log-group",
            log_stream_name="custom-stream",
            aws_region="us-west-2",
        )
        self.assertEqual(exporter.namespace, "CustomNamespace")
        self.assertEqual(exporter.log_group_name, "custom-log-group")
        self.assertEqual(exporter.log_stream_name, "custom-stream")

    def test_get_unit_mapping(self):
        """Test unit mapping functionality."""
        # Test known units from UNIT_MAPPING
        self.assertEqual(self.exporter._get_unit(Mock(unit="ms")), "Milliseconds")
        self.assertEqual(self.exporter._get_unit(Mock(unit="s")), "Seconds")
        self.assertEqual(self.exporter._get_unit(Mock(unit="us")), "Microseconds")
        self.assertEqual(self.exporter._get_unit(Mock(unit="By")), "Bytes")
        self.assertEqual(self.exporter._get_unit(Mock(unit="bit")), "Bits")

        # Test units that map to empty string (should return empty string from mapping)
        self.assertEqual(self.exporter._get_unit(Mock(unit="1")), "")
        self.assertEqual(self.exporter._get_unit(Mock(unit="ns")), "")

        # Test EMF supported units directly (should return as-is)
        self.assertEqual(self.exporter._get_unit(Mock(unit="Count")), "Count")
        self.assertEqual(self.exporter._get_unit(Mock(unit="Percent")), "Percent")
        self.assertEqual(self.exporter._get_unit(Mock(unit="Kilobytes")), "Kilobytes")

        # Test unknown unit (not in mapping and not in supported units, returns None)
        self.assertIsNone(self.exporter._get_unit(Mock(unit="unknown")))

        # Test empty unit (should return None due to falsy check)
        self.assertIsNone(self.exporter._get_unit(Mock(unit="")))

        # Test None unit
        self.assertIsNone(self.exporter._get_unit(Mock(unit=None)))

    def test_get_metric_name(self):
        """Test metric name extraction."""
        # Test with record that has instrument.name
        record = Mock()
        record.instrument = Mock()
        record.instrument.name = "test_metric"

        result = self.exporter._get_metric_name(record)
        self.assertEqual(result, "test_metric")

        # Test with record that has empty instrument name (should return None)
        record_empty = Mock()
        record_empty.instrument = Mock()
        record_empty.instrument.name = ""

        result_empty = self.exporter._get_metric_name(record_empty)
        self.assertIsNone(result_empty)

    def test_get_dimension_names(self):
        """Test dimension names extraction."""
        attributes = {"service.name": "test-service", "env": "prod", "region": "us-east-1"}

        result = self.exporter._get_dimension_names(attributes)

        # Should return all attribute keys
        self.assertEqual(set(result), {"service.name", "env", "region"})

    def test_get_attributes_key(self):
        """Test attributes key generation."""
        attributes = {"service": "test", "env": "prod"}

        result = self.exporter._get_attributes_key(attributes)

        # Should be a string representation of sorted attributes
        self.assertIsInstance(result, str)
        self.assertIn("service", result)
        self.assertIn("test", result)
        self.assertIn("env", result)
        self.assertIn("prod", result)

    def test_get_attributes_key_consistent(self):
        """Test that attributes key generation is consistent."""
        # Same attributes in different order should produce same key
        attrs1 = {"b": "2", "a": "1"}
        attrs2 = {"a": "1", "b": "2"}

        key1 = self.exporter._get_attributes_key(attrs1)
        key2 = self.exporter._get_attributes_key(attrs2)

        self.assertEqual(key1, key2)

    def test_group_by_attributes_and_timestamp(self):
        """Test grouping by attributes and timestamp."""
        record = Mock()
        record.attributes = {"env": "test"}
        timestamp_ms = 1234567890

        result = self.exporter._group_by_attributes_and_timestamp(record, timestamp_ms)

        # Should return a tuple with attributes key and timestamp
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[1], timestamp_ms)

    def test_generate_log_stream_name(self):
        """Test log stream name generation."""
        name1 = self.exporter._generate_log_stream_name()
        name2 = self.exporter._generate_log_stream_name()

        # Should generate unique names
        self.assertNotEqual(name1, name2)
        self.assertTrue(name1.startswith("otel-python-"))
        self.assertTrue(name2.startswith("otel-python-"))

    def test_normalize_timestamp(self):
        """Test timestamp normalization."""
        timestamp_ns = 1609459200000000000  # 2021-01-01 00:00:00 in nanoseconds
        expected_ms = 1609459200000  # Same time in milliseconds

        result = self.exporter._normalize_timestamp(timestamp_ns)
        self.assertEqual(result, expected_ms)

    def test_create_metric_record(self):
        """Test metric record creation."""
        record = self.exporter._create_metric_record("test_metric", "Count", "Test description")

        self.assertIsNotNone(record)
        self.assertIsNotNone(record.instrument)
        self.assertEqual(record.instrument.name, "test_metric")
        self.assertEqual(record.instrument.unit, "Count")
        self.assertEqual(record.instrument.description, "Test description")

    def test_convert_gauge(self):
        """Test gauge conversion."""
        metric = MockMetric("gauge_metric", "Count", "Gauge description")
        dp = MockDataPoint(value=42.5, attributes={"key": "value"})

        record, timestamp = self.exporter._convert_gauge(metric, dp)

        self.assertIsNotNone(record)
        self.assertEqual(record.instrument.name, "gauge_metric")
        self.assertEqual(record.value, 42.5)
        self.assertEqual(record.attributes, {"key": "value"})
        self.assertIsInstance(timestamp, int)

    def test_create_emf_log(self):
        """Test EMF log creation."""
        # Create test records
        gauge_record = self.exporter._create_metric_record("gauge_metric", "Count", "Gauge")
        gauge_record.value = 50.0
        gauge_record.timestamp = int(time.time() * 1000)
        gauge_record.attributes = {"env": "test"}

        records = [gauge_record]
        resource = Resource.create({"service.name": "test-service"})

        result = self.exporter._create_emf_log(records, resource)

        self.assertIsInstance(result, dict)

        # Check that the result is JSON serializable
        json.dumps(result)  # Should not raise exception

    @patch("botocore.session.Session")
    def test_export_success(self, mock_session):
        """Test successful export."""
        # Mock CloudWatch Logs client
        mock_client = Mock()
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.create_client.return_value = mock_client
        mock_client.put_log_events.return_value = {"nextSequenceToken": "12345"}

        # Create empty metrics data to test basic export flow
        metrics_data = Mock()
        metrics_data.resource_metrics = []

        result = self.exporter.export(metrics_data)

        self.assertEqual(result, MetricExportResult.SUCCESS)

    def test_export_failure(self):
        """Test export failure handling."""
        # Create metrics data that will cause an exception during iteration
        metrics_data = Mock()
        # Make resource_metrics raise an exception when iterated over
        metrics_data.resource_metrics = Mock()
        metrics_data.resource_metrics.__iter__ = Mock(side_effect=Exception("Test exception"))

        result = self.exporter.export(metrics_data)

        self.assertEqual(result, MetricExportResult.FAILURE)

    def test_force_flush_no_pending_events(self):
        """Test force flush functionality with no pending events."""
        result = self.exporter.force_flush()

        self.assertTrue(result)

    @patch.object(AwsCloudWatchEMFExporter, "force_flush")
    def test_shutdown(self, mock_force_flush):
        """Test shutdown functionality."""
        mock_force_flush.return_value = True

        result = self.exporter.shutdown(timeout_millis=5000)

        self.assertTrue(result)
        mock_force_flush.assert_called_once_with(5000)

    def test_send_log_event_method_exists(self):
        """Test that _send_log_event method exists and can be called."""
        # Just test that the method exists and doesn't crash with basic input
        log_event = {"message": "test message", "timestamp": 1234567890}

        # Mock the AWS client methods to avoid actual AWS calls
        with patch.object(self.exporter.logs_client, "create_log_group"):
            with patch.object(self.exporter.logs_client, "create_log_stream"):
                with patch.object(self.exporter.logs_client, "put_log_events") as mock_put:
                    mock_put.return_value = {"nextSequenceToken": "12345"}

                    # Should not raise an exception
                    try:
                        response = self.exporter._send_log_event(log_event)
                        # Response may be None or a dict, both are acceptable
                        self.assertTrue(response is None or isinstance(response, dict))
                    except ClientError as error:
                        self.fail(f"_send_log_event raised an exception: {error}")

    def test_create_emf_log_with_resource(self):
        """Test EMF log creation with resource attributes."""
        # Create test records
        gauge_record = self.exporter._create_metric_record("gauge_metric", "Count", "Gauge")
        gauge_record.value = 50.0
        gauge_record.timestamp = int(time.time() * 1000)
        gauge_record.attributes = {"env": "test", "service": "api"}

        records = [gauge_record]
        resource = Resource.create({"service.name": "test-service", "service.version": "1.0.0"})

        result = self.exporter._create_emf_log(records, resource, 1234567890)

        # Verify EMF log structure
        self.assertIn("_aws", result)
        self.assertIn("CloudWatchMetrics", result["_aws"])
        self.assertEqual(result["_aws"]["Timestamp"], 1234567890)
        self.assertEqual(result["Version"], "1")

        # Check resource attributes are prefixed
        self.assertEqual(result["otel.resource.service.name"], "test-service")
        self.assertEqual(result["otel.resource.service.version"], "1.0.0")

        # Check metric attributes
        self.assertEqual(result["env"], "test")
        self.assertEqual(result["service"], "api")

        # Check metric value
        self.assertEqual(result["gauge_metric"], 50.0)

        # Check CloudWatch metrics structure
        cw_metrics = result["_aws"]["CloudWatchMetrics"][0]
        self.assertEqual(cw_metrics["Namespace"], "TestNamespace")
        self.assertEqual(set(cw_metrics["Dimensions"][0]), {"env", "service"})
        self.assertEqual(cw_metrics["Metrics"][0]["Name"], "gauge_metric")

    @patch("botocore.session.Session")
    def test_export_with_gauge_metrics(self, mock_session):
        """Test exporting actual gauge metrics."""
        # Mock CloudWatch Logs client
        mock_client = Mock()
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.create_client.return_value = mock_client
        mock_client.put_log_events.return_value = {"nextSequenceToken": "12345"}
        mock_client.create_log_group.side_effect = ClientError(
            {"Error": {"Code": "ResourceAlreadyExistsException"}}, "CreateLogGroup"
        )
        mock_client.create_log_stream.side_effect = ClientError(
            {"Error": {"Code": "ResourceAlreadyExistsException"}}, "CreateLogStream"
        )

        # Create mock metrics data
        resource = Resource.create({"service.name": "test-service"})

        # Create gauge data
        gauge_data = Gauge(data_points=[MockDataPoint(value=42.0, attributes={"key": "value"})])

        metric = MockMetricWithData(name="test_gauge", data=gauge_data)

        scope_metrics = MockScopeMetrics(metrics=[metric])
        resource_metrics = MockResourceMetrics(resource=resource, scope_metrics=[scope_metrics])

        metrics_data = Mock()
        metrics_data.resource_metrics = [resource_metrics]

        result = self.exporter.export(metrics_data)

        self.assertEqual(result, MetricExportResult.SUCCESS)
        # Test validates that export works with gauge metrics

    def test_get_metric_name_fallback(self):
        """Test metric name extraction fallback."""
        # Test with record that has no instrument attribute
        record = Mock(spec=[])

        result = self.exporter._get_metric_name(record)
        self.assertIsNone(result)

    def test_get_metric_name_empty_name(self):
        """Test metric name extraction with empty instrument name."""
        # Test with record that has empty instrument name
        record = Mock()
        record.instrument = Mock()
        record.instrument.name = ""

        result = self.exporter._get_metric_name(record)
        self.assertIsNone(result)

    def test_create_emf_log_skips_empty_metric_names(self):
        """Test that EMF log creation skips records with empty metric names."""
        # Create a record with no metric name but with proper instrument
        record_without_name = Mock()
        record_without_name.attributes = {"key": "value"}
        record_without_name.value = 10.0
        record_without_name.instrument = Mock()
        record_without_name.instrument.name = None  # No valid name

        # Create a record with valid metric name
        valid_record = self.exporter._create_metric_record("valid_metric", "Count", "Valid metric")
        valid_record.value = 20.0
        valid_record.attributes = {"key": "value"}

        records = [record_without_name, valid_record]
        resource = Resource.create({"service.name": "test-service"})

        result = self.exporter._create_emf_log(records, resource, 1234567890)

        # Only the valid record should be processed
        self.assertIn("valid_metric", result)
        self.assertEqual(result["valid_metric"], 20.0)

        # Check that only the valid metric is in the definitions (empty names are skipped)
        cw_metrics = result["_aws"]["CloudWatchMetrics"][0]
        self.assertEqual(len(cw_metrics["Metrics"]), 1)
        # Ensure our valid metric is present
        metric_names = [m["Name"] for m in cw_metrics["Metrics"]]
        self.assertIn("valid_metric", metric_names)

    @patch("os.environ.get")
    @patch("botocore.session.Session")
    def test_initialization_with_env_region(self, mock_session, mock_env_get):
        """Test initialization with AWS region from environment."""
        # Mock environment variable
        mock_env_get.side_effect = lambda key: "us-west-1" if key == "AWS_REGION" else None

        # Mock the botocore session to avoid AWS calls
        mock_client = Mock()
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.create_client.return_value = mock_client
        mock_client.create_log_group.return_value = {}
        mock_client.create_log_stream.return_value = {}

        exporter = AwsCloudWatchEMFExporter(namespace="TestNamespace", log_group_name="test-log-group")

        # Just verify the exporter was created successfully with region handling
        self.assertIsNotNone(exporter)
        self.assertEqual(exporter.namespace, "TestNamespace")

    @patch("botocore.session.Session")
    def test_ensure_log_group_exists_create_failure(self, mock_session):
        """Test log group creation failure."""
        # Mock the botocore session
        mock_client = Mock()
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.create_client.return_value = mock_client

        # Make create fail with access denied error
        mock_client.create_log_group.side_effect = ClientError({"Error": {"Code": "AccessDenied"}}, "CreateLogGroup")
        mock_client.create_log_stream.return_value = {}

        with self.assertRaises(ClientError):
            AwsCloudWatchEMFExporter(namespace="TestNamespace", log_group_name="test-log-group")

    @patch("botocore.session.Session")
    def test_ensure_log_group_exists_success(self, mock_session):
        """Test log group existence check when log group already exists."""
        # Mock the botocore session
        mock_client = Mock()
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.create_client.return_value = mock_client

        # Make create fail with ResourceAlreadyExistsException (log group exists)
        mock_client.create_log_group.side_effect = ClientError(
            {"Error": {"Code": "ResourceAlreadyExistsException"}}, "CreateLogGroup"
        )
        mock_client.create_log_stream.return_value = {}

        # This should not raise an exception
        exporter = AwsCloudWatchEMFExporter(namespace="TestNamespace", log_group_name="test-log-group")
        self.assertIsNotNone(exporter)
        # Verify create was called once
        mock_client.create_log_group.assert_called_once_with(logGroupName="test-log-group")

    def test_export_with_unsupported_metric_type(self):
        """Test export with unsupported metric types."""
        # Create mock metrics data with unsupported metric type
        resource = Resource.create({"service.name": "test-service"})

        # Create non-gauge data
        unsupported_data = Mock()
        unsupported_data.data_points = [MockDataPoint(value=42.0)]

        metric = MockMetricWithData(name="test_counter", data=unsupported_data)

        scope_metrics = MockScopeMetrics(metrics=[metric])
        resource_metrics = MockResourceMetrics(resource=resource, scope_metrics=[scope_metrics])

        metrics_data = Mock()
        metrics_data.resource_metrics = [resource_metrics]

        # Should still return success even with unsupported metrics
        result = self.exporter.export(metrics_data)
        self.assertEqual(result, MetricExportResult.SUCCESS)

    def test_export_with_metric_without_data(self):
        """Test export with metrics that don't have data attribute."""
        # Create mock metrics data
        resource = Resource.create({"service.name": "test-service"})

        # Create metric without data attribute
        metric = Mock(spec=[])

        scope_metrics = MockScopeMetrics(metrics=[metric])
        resource_metrics = MockResourceMetrics(resource=resource, scope_metrics=[scope_metrics])

        metrics_data = Mock()
        metrics_data.resource_metrics = [resource_metrics]

        # Should still return success
        result = self.exporter.export(metrics_data)
        self.assertEqual(result, MetricExportResult.SUCCESS)

    def test_convert_sum(self):
        """Test sum conversion."""
        metric = MockMetric("sum_metric", "Count", "Sum description")
        dp = MockDataPoint(value=100.0, attributes={"env": "test"})
        
        record, timestamp = self.exporter._convert_sum(metric, dp)
        
        self.assertIsNotNone(record)
        self.assertEqual(record.instrument.name, "sum_metric")
        self.assertTrue(hasattr(record, 'sum_data'))
        self.assertEqual(record.sum_data.value, 100.0)
        self.assertEqual(record.attributes, {"env": "test"})
        self.assertIsInstance(timestamp, int)

    def test_convert_histogram(self):
        """Test histogram conversion."""
        metric = MockMetric("histogram_metric", "ms", "Histogram description")
        dp = MockHistogramDataPoint(
            count=10,
            sum_val=150.0,
            min_val=5.0,
            max_val=25.0,
            attributes={"region": "us-east-1"}
        )
        
        record, timestamp = self.exporter._convert_histogram(metric, dp)
        
        self.assertIsNotNone(record)
        self.assertEqual(record.instrument.name, "histogram_metric")
        self.assertTrue(hasattr(record, 'histogram_data'))
        
        expected_value = {
            "Count": 10,
            "Sum": 150.0,
            "Min": 5.0,
            "Max": 25.0
        }
        self.assertEqual(record.histogram_data.value, expected_value)
        self.assertEqual(record.attributes, {"region": "us-east-1"})
        self.assertIsInstance(timestamp, int)

    def test_convert_exp_histogram(self):
        """Test exponential histogram conversion."""
        metric = MockMetric("exp_histogram_metric", "s", "Exponential histogram description")
        dp = MockExpHistogramDataPoint(
            count=8,
            sum_val=64.0,
            min_val=2.0,
            max_val=32.0,
            attributes={"service": "api"}
        )
        
        record, timestamp = self.exporter._convert_exp_histogram(metric, dp)
        
        self.assertIsNotNone(record)
        self.assertEqual(record.instrument.name, "exp_histogram_metric")
        self.assertTrue(hasattr(record, 'exp_histogram_data'))
        
        exp_data = record.exp_histogram_data.value
        self.assertIn("Values", exp_data)
        self.assertIn("Counts", exp_data)
        self.assertEqual(exp_data["Count"], 8)
        self.assertEqual(exp_data["Sum"], 64.0)
        self.assertEqual(exp_data["Min"], 2.0)
        self.assertEqual(exp_data["Max"], 32.0)
        self.assertEqual(record.attributes, {"service": "api"})
        self.assertIsInstance(timestamp, int)

    def test_export_with_sum_metrics(self):
        """Test export with Sum metrics."""
        # Create mock metrics data with Sum type
        resource = Resource.create({"service.name": "test-service"})
        
        sum_data = MockSumData([MockDataPoint(value=25.0, attributes={"env": "test"})])
        # Create a mock that will pass the type() check for Sum
        sum_data.__class__ = Sum
        metric = MockMetricWithData(name="test_sum", data=sum_data)
        
        scope_metrics = MockScopeMetrics(metrics=[metric])
        resource_metrics = MockResourceMetrics(resource=resource, scope_metrics=[scope_metrics])
        
        metrics_data = Mock()
        metrics_data.resource_metrics = [resource_metrics]
        
        result = self.exporter.export(metrics_data)
        self.assertEqual(result, MetricExportResult.SUCCESS)

    def test_export_with_histogram_metrics(self):
        """Test export with Histogram metrics."""
        # Create mock metrics data with Histogram type
        resource = Resource.create({"service.name": "test-service"})
        
        hist_dp = MockHistogramDataPoint(count=5, sum_val=25.0, min_val=1.0, max_val=10.0, attributes={"env": "test"})
        hist_data = MockHistogramData([hist_dp])
        # Create a mock that will pass the type() check for Histogram
        hist_data.__class__ = Histogram
        metric = MockMetricWithData(name="test_histogram", data=hist_data)
        
        scope_metrics = MockScopeMetrics(metrics=[metric])
        resource_metrics = MockResourceMetrics(resource=resource, scope_metrics=[scope_metrics])
        
        metrics_data = Mock()
        metrics_data.resource_metrics = [resource_metrics]
        
        result = self.exporter.export(metrics_data)
        self.assertEqual(result, MetricExportResult.SUCCESS)


class TestBatchProcessing(unittest.TestCase):
    """Test batch processing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock the botocore session to avoid AWS calls
        with patch("botocore.session.Session") as mock_session:
            mock_client = Mock()
            mock_session_instance = Mock()
            mock_session.return_value = mock_session_instance
            mock_session_instance.create_client.return_value = mock_client
            mock_client.create_log_group.return_value = {}
            mock_client.create_log_stream.return_value = {}
            
            self.exporter = AwsCloudWatchEMFExporter(
                namespace="TestNamespace",
                log_group_name="test-log-group"
            )
    
    def test_create_event_batch(self):
        """Test event batch creation."""
        batch = self.exporter._create_event_batch()
        
        self.assertEqual(batch["logEvents"], [])
        self.assertEqual(batch["byteTotal"], 0)
        self.assertEqual(batch["minTimestampMs"], 0)
        self.assertEqual(batch["maxTimestampMs"], 0)
        self.assertIsInstance(batch["createdTimestampMs"], int)
    
    def test_validate_log_event_valid(self):
        """Test log event validation with valid event."""
        log_event = {
            "message": "test message",
            "timestamp": int(time.time() * 1000)
        }
        
        result = self.exporter._validate_log_event(log_event)
        self.assertTrue(result)
    
    def test_validate_log_event_empty_message(self):
        """Test log event validation with empty message."""
        log_event = {
            "message": "",
            "timestamp": int(time.time() * 1000)
        }
        
        result = self.exporter._validate_log_event(log_event)
        self.assertFalse(result)
    
    def test_validate_log_event_oversized_message(self):
        """Test log event validation with oversized message."""
        # Create a message larger than the maximum allowed size
        large_message = "x" * (self.exporter.CW_MAX_EVENT_PAYLOAD_BYTES + 100)
        log_event = {
            "message": large_message,
            "timestamp": int(time.time() * 1000)
        }
        
        result = self.exporter._validate_log_event(log_event)
        self.assertTrue(result)  # Should still be valid after truncation
        # Check that message was truncated
        self.assertLess(len(log_event["message"]), len(large_message))
        self.assertTrue(log_event["message"].endswith(self.exporter.CW_TRUNCATED_SUFFIX))
    
    def test_validate_log_event_old_timestamp(self):
        """Test log event validation with very old timestamp."""
        # Timestamp from 15 days ago
        old_timestamp = int(time.time() * 1000) - (15 * 24 * 60 * 60 * 1000)
        log_event = {
            "message": "test message",
            "timestamp": old_timestamp
        }
        
        result = self.exporter._validate_log_event(log_event)
        self.assertFalse(result)
    
    def test_validate_log_event_future_timestamp(self):
        """Test log event validation with future timestamp."""
        # Timestamp 3 hours in the future
        future_timestamp = int(time.time() * 1000) + (3 * 60 * 60 * 1000)
        log_event = {
            "message": "test message",
            "timestamp": future_timestamp
        }
        
        result = self.exporter._validate_log_event(log_event)
        self.assertFalse(result)
    
    def test_event_batch_exceeds_limit_by_count(self):
        """Test batch limit checking by event count."""
        batch = self.exporter._create_event_batch()
        # Simulate batch with maximum events
        batch["logEvents"] = [{"message": "test"}] * self.exporter.CW_MAX_REQUEST_EVENT_COUNT
        
        result = self.exporter._event_batch_exceeds_limit(batch, 100)
        self.assertTrue(result)
    
    def test_event_batch_exceeds_limit_by_size(self):
        """Test batch limit checking by byte size."""
        batch = self.exporter._create_event_batch()
        batch["byteTotal"] = self.exporter.CW_MAX_REQUEST_PAYLOAD_BYTES - 50
        
        result = self.exporter._event_batch_exceeds_limit(batch, 100)
        self.assertTrue(result)
    
    def test_event_batch_within_limits(self):
        """Test batch limit checking within limits."""
        batch = self.exporter._create_event_batch()
        batch["logEvents"] = [{"message": "test"}] * 10
        batch["byteTotal"] = 1000
        
        result = self.exporter._event_batch_exceeds_limit(batch, 100)
        self.assertFalse(result)
    
    def test_is_batch_active_new_batch(self):
        """Test batch activity check for new batch."""
        batch = self.exporter._create_event_batch()
        current_time = int(time.time() * 1000)
        
        result = self.exporter._is_batch_active(batch, current_time)
        self.assertTrue(result)
    
    def test_is_batch_active_24_hour_span(self):
        """Test batch activity check for 24+ hour span."""
        batch = self.exporter._create_event_batch()
        current_time = int(time.time() * 1000)
        batch["minTimestampMs"] = current_time
        batch["maxTimestampMs"] = current_time
        
        # Test with timestamp 25 hours in the future
        future_timestamp = current_time + (25 * 60 * 60 * 1000)
        
        result = self.exporter._is_batch_active(batch, future_timestamp)
        self.assertFalse(result)
    
    def test_append_to_batch(self):
        """Test appending log event to batch."""
        batch = self.exporter._create_event_batch()
        log_event = {
            "message": "test message",
            "timestamp": int(time.time() * 1000)
        }
        event_size = 100
        
        self.exporter._append_to_batch(batch, log_event, event_size)
        
        self.assertEqual(len(batch["logEvents"]), 1)
        self.assertEqual(batch["byteTotal"], event_size)
        self.assertEqual(batch["minTimestampMs"], log_event["timestamp"])
        self.assertEqual(batch["maxTimestampMs"], log_event["timestamp"])
    
    def test_sort_log_events(self):
        """Test sorting log events by timestamp."""
        batch = self.exporter._create_event_batch()
        current_time = int(time.time() * 1000)
        
        # Add events with timestamps in reverse order
        events = [
            {"message": "third", "timestamp": current_time + 2000},
            {"message": "first", "timestamp": current_time},
            {"message": "second", "timestamp": current_time + 1000}
        ]
        
        batch["logEvents"] = events.copy()
        self.exporter._sort_log_events(batch)
        
        # Check that events are now sorted by timestamp
        self.assertEqual(batch["logEvents"][0]["message"], "first")
        self.assertEqual(batch["logEvents"][1]["message"], "second")
        self.assertEqual(batch["logEvents"][2]["message"], "third")
    
    @patch.object(AwsCloudWatchEMFExporter, '_send_log_batch')
    def test_force_flush_with_pending_events(self, mock_send_batch):
        """Test force flush functionality with pending events."""
        # Create a batch with events
        self.exporter._event_batch = self.exporter._create_event_batch()
        self.exporter._event_batch["logEvents"] = [{"message": "test", "timestamp": int(time.time() * 1000)}]
        
        result = self.exporter.force_flush()
        
        self.assertTrue(result)
        mock_send_batch.assert_called_once()
    
    def test_force_flush_no_pending_events(self):
        """Test force flush functionality with no pending events."""
        # No batch exists
        self.assertIsNone(self.exporter._event_batch)
        
        result = self.exporter.force_flush()
        
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
