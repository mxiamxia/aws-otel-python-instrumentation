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
        """
        Matches text against a pattern that can contain '*' (any sequence of characters)
        and '?' (any single character).
        """
        if pattern == "*":
            return True
        if text is None or pattern is None:
            return False
        if len(pattern) == 0:
            return len(text) == 0

        # Optimization: Fast-path for exact match if no wildcards are present.
        if "*" not in pattern and "?" not in pattern:
            return text == pattern

        # Optimization: Fast-paths for common simple wildcard patterns.
        # Case: "prefix*"
        if pattern.endswith("*") and "*" not in pattern[:-1] and "?" not in pattern:
            return text.startswith(pattern[:-1])
        # Case: "*suffix"
        if pattern.startswith("*") and "*" not in pattern[1:] and "?" not in pattern:
            return text.endswith(pattern[1:])
        # Case: "*middle*"
        if pattern.startswith("*") and pattern.endswith("*") and "*" not in pattern[1:-1] and "?" not in pattern:
            return pattern[1:-1] in text

        # Fallback to regex for complex patterns, using a cache to avoid repeated compilation.
        return _Matcher._get_compiled_regex(pattern).fullmatch(text) is not None

    @staticmethod
    @functools.lru_cache(maxsize=1024)
    def _get_compiled_regex(pattern: str) -> re.Pattern:
        """Translates a wildcard pattern to a compiled regex and caches the result."""
        return re.compile(_Matcher.to_regex_pattern(pattern))

    @staticmethod
    def to_regex_pattern(rule_pattern: str) -> str:
        """Converts a wildcard pattern (with * and ?) into a regex string."""
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
        """
        Checks if all rule attributes match the given span attributes.
        Optimized to iterate over the rule attributes and early-return on the first mismatch.
        """
        if rule_attributes is None or len(rule_attributes) == 0:
            return True
        if attributes is None or len(attributes) == 0 or len(rule_attributes) > len(attributes):
            return False

        # Optimization: Iterate over rule_attributes (O(R)) instead of span attributes (O(A)).
        # R is typically much smaller than A, and this allows for an early return.
        for key, pattern in rule_attributes.items():
            text_to_match = attributes.get(key)
            if text_to_match is None:
                return False
            if not _Matcher.wild_card_match(text_to_match, pattern):
                return False

        return True
