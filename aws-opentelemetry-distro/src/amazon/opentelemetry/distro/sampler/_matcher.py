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
    def wild_card_match(text: str = None, pattern: str = None) -> bool:
        """
        Match text against a wildcard pattern. Supports '*' (any number of characters)
        and '?' (exactly one character).
        """
        # Fast path for '*'
        if pattern == "*":
            return True
        if text is None or pattern is None:
            return False
        # Fast path for empty pattern
        if not pattern:
            return not text

        # Optimize common wildcard patterns
        if "?" not in pattern:
            star_count = pattern.count("*")
            if star_count == 0:
                # Exact match
                return text == pattern
            if star_count == 1:
                if pattern.endswith("*"):
                    # Prefix match: "prefix*"
                    return text.startswith(pattern[:-1])
                if pattern.startswith("*"):
                    # Suffix match: "*suffix"
                    return text.endswith(pattern[1:])
            elif star_count == 2 and pattern.startswith("*") and pattern.endswith("*"):
                # Substring match: "*middle*"
                return pattern[1:-1] in text

        # Fallback to regex for complex patterns
        regex = _Matcher._get_compiled_regex(pattern)
        return regex.fullmatch(text) is not None

    @staticmethod
    @lru_cache(maxsize=1024)
    def _get_compiled_regex(pattern: str) -> re.Pattern:
        """Converts a wildcard pattern to a compiled regex and caches it."""
        return re.compile(_Matcher.to_regex_pattern(pattern))

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
        """
        Checks if all rule attributes match the given span attributes.
        Iterates over rule_attributes (O(R)) instead of span attributes (O(A))
        for better performance when rules are fewer than span attributes.
        """
        if not rule_attributes:
            return True
        if not attributes or len(rule_attributes) > len(attributes):
            return False

        # Iterate over rule attributes and check against span attributes.
        # This is more efficient as it allows for early return on first mismatch.
        for key, pattern in rule_attributes.items():
            val = attributes.get(key)
            if val is None or not _Matcher.wild_card_match(val, pattern):
                return False

        return True
