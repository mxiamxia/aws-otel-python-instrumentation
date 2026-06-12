# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import re
from functools import lru_cache

from opentelemetry.semconv.resource import CloudPlatformValues
from opentelemetry.util.types import Attributes

cloud_platform_mapping = {
    CloudPlatformValues.AWS_LAMBDA.value: "AWS::Lambda::Function",
    CloudPlatformValues.AWS_ELASTIC_BEANSTALK.value: "AWS::ElasticBeanstalk::Environment",
    CloudPlatformValues.AWS_EC2.value: "AWS::EC2::Instance",
    CloudPlatformValues.AWS_ECS.value: "AWS::ECS::Container",
    CloudPlatformValues.AWS_EKS.value: "AWS::EKS::Container",
}


@lru_cache(maxsize=1024)
def _get_compiled_pattern(pattern: str):
    return re.compile(_Matcher.to_regex_pattern(pattern))


class _Matcher:
    @staticmethod
    def wild_card_match(text: str = None, pattern: str = None) -> bool:
        if pattern == "*":
            return True
        if text is None or pattern is None:
            return False
        if len(pattern) == 0:
            return len(text) == 0

        # Fast-paths for common patterns to bypass regex
        star_count = pattern.count("*")
        if star_count == 0 and "?" not in pattern:
            return text == pattern

        if "?" not in pattern:
            result = _Matcher._wild_card_fast_path(text, pattern, star_count)
            if result is not None:
                return result

        return _get_compiled_pattern(pattern).fullmatch(text) is not None

    @staticmethod
    def _wild_card_fast_path(text: str, pattern: str, star_count: int) -> bool:
        if star_count == 1:
            if pattern.startswith("*"):
                return text.endswith(pattern[1:])
            if pattern.endswith("*"):
                return text.startswith(pattern[:-1])
            parts = pattern.split("*")
            return text.startswith(parts[0]) and text.endswith(parts[1]) and len(text) >= len(pattern) - 1
        if star_count == 2 and pattern.startswith("*") and pattern.endswith("*"):
            return pattern[1:-1] in text
        return None

    @staticmethod
    def to_regex_pattern(rule_pattern: str) -> str:
        token_start = -1
        regex_pattern = ""
        for index, char in enumerate(rule_pattern):
            char = rule_pattern[index]
            if char in ("*", "?"):
                if token_start != -1:
                    regex_pattern += re.escape(rule_pattern[token_start:index])
                    token_start = -1
                if char == "*":
                    regex_pattern += ".*"
                else:
                    regex_pattern += "."
            else:
                if token_start == -1:
                    token_start = index
        if token_start != -1:
            regex_pattern += re.escape(rule_pattern[token_start:])
        return regex_pattern

    @staticmethod
    def attribute_match(attributes: Attributes = None, rule_attributes: dict = None) -> bool:
        if rule_attributes is None or len(rule_attributes) == 0:
            return True
        if attributes is None or len(attributes) == 0:
            return False

        # Optimization: iterate over rule_attributes (O(R)) instead of span attributes (O(A))
        # This is typically much faster as R << A.
        for key, pattern in rule_attributes.items():
            val = attributes.get(key)
            if val is None:
                return False

            # Ensure value is a string for wildcard matching
            text_to_match = val if isinstance(val, str) else str(val)
            if not _Matcher.wild_card_match(text_to_match, pattern):
                return False

        return True
