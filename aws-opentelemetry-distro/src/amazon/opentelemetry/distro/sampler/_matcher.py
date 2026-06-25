# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import functools
import re

from opentelemetry.semconv.resource import CloudPlatformValues
from opentelemetry.util.types import Attributes

cloud_platform_mapping = {
    CloudPlatformValues.AWS_LAMBDA.value: "AWS::Lambda::Function",
    CloudPlatformValues.AWS_ELASTIC_BEANSTALK.value: "AWS::ElasticBeanstalk::Environment",
    CloudPlatformValues.AWS_EC2.value: "AWS::EC2::Instance",
    CloudPlatformValues.AWS_ECS.value: "AWS::ECS::Container",
    CloudPlatformValues.AWS_EKS.value: "AWS::EKS::Container",
}


class _Matcher:
    @staticmethod
    def wild_card_match(text: str = None, pattern: str = None) -> bool:
        """Match text against a wildcard pattern.

        Supports '*' (any characters) and '?' (single character).
        Includes fast paths for common patterns like *, prefix*, *suffix, and *middle*.
        """
        if pattern == "*":
            return True
        if text is None or pattern is None:
            return False
        if len(pattern) == 0:
            return len(text) == 0

        fast_path_result = _Matcher._wild_card_match_fast_path(text, pattern)
        if fast_path_result is not None:
            return fast_path_result

        for char in pattern:
            if char in ("*", "?"):
                return re.fullmatch(_Matcher.to_regex_pattern(pattern), text) is not None
        return pattern == text

    @staticmethod
    def _wild_card_match_fast_path(text: str, pattern: str) -> bool:
        """Internal helper for fast-path wildcard matching."""
        if "?" not in pattern:
            wildcard_count = pattern.count("*")
            if wildcard_count == 1:
                if pattern.endswith("*"):
                    return text.startswith(pattern[:-1])
                if pattern.startswith("*"):
                    return text.endswith(pattern[1:])
            if wildcard_count == 2 and pattern.startswith("*") and pattern.endswith("*"):
                return pattern[1:-1] in text
        return None

    @staticmethod
    @functools.lru_cache(maxsize=1024)
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
        """Match rule attributes against span attributes.

        This method iterates over the rule attributes (usually a smaller set) and
        checks if they are present and matching in the span attributes.
        """
        if not rule_attributes:
            return True
        if not attributes:
            return False

        for key, pattern in rule_attributes.items():
            val = attributes.get(key)
            if val is None:
                return False
            # Ensure the value is a string for wildcard matching
            if not _Matcher.wild_card_match(str(val), pattern):
                return False
        return True
