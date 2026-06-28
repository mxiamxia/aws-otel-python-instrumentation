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


class _Matcher:
    @staticmethod
    @lru_cache(maxsize=1024)
    def _get_compiled_regex(pattern: str):
        return re.compile(_Matcher.to_regex_pattern(pattern))

    @staticmethod
    def wild_card_match(text: str = None, pattern: str = None) -> bool:
        """
        Matches a text against a pattern containing '*' (zero or more characters)
        and '?' (exactly one character).
        """
        if pattern == "*":
            return True
        if text is None or pattern is None:
            return False
        if len(pattern) == 0:
            return len(text) == 0

        # Fast paths for common patterns to avoid regex overhead.
        # Refactored to private helper to comply with R0911 (too many return statements).
        fast_path_result = _Matcher._wild_card_match_fast_path(text, pattern)
        if fast_path_result is not None:
            return fast_path_result

        if "*" in pattern or "?" in pattern:
            return _Matcher._get_compiled_regex(pattern).fullmatch(text) is not None
        return pattern == text

    @staticmethod
    def _wild_card_match_fast_path(text: str, pattern: str):
        """
        Evaluates fast-path optimizations for common single-wildcard patterns.
        Returns the match result (bool) if a fast-path is applicable, else None.
        """
        if "?" in pattern:
            return None

        star_count = pattern.count("*")
        if star_count == 1:
            if pattern.startswith("*"):
                return text.endswith(pattern[1:])
            if pattern.endswith("*"):
                return text.startswith(pattern[:-1])
            parts = pattern.split("*")
            # For "prefix*suffix", ensure both match and that the text is long enough to cover both parts
            return text.startswith(parts[0]) and text.endswith(parts[1]) and len(text) >= len(pattern) - 1

        if star_count == 2 and pattern.startswith("*") and pattern.endswith("*"):
            return pattern[1:-1] in text

        return None

    @staticmethod
    def to_regex_pattern(rule_pattern: str) -> str:
        token_start = -1
        regex_pattern = ""
        for index, char in enumerate(rule_pattern):
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
        if attributes is None or len(attributes) == 0 or len(rule_attributes) > len(attributes):
            return False

        # Iterate over rule_attributes instead of span attributes for better performance
        # when the span has many attributes, and to allow for early exit on the first mismatch.
        # Performance rationale: O(RuleAttributes) instead of O(SpanAttributes).
        for key, pattern in rule_attributes.items():
            val = attributes.get(key)
            if val is None:
                return False
            # Ensure value is converted to string for matching
            if not _Matcher.wild_card_match(str(val), pattern):
                return False
        return True
