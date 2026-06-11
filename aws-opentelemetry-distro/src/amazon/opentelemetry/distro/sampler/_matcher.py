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
        Performs a wildcard match on the given text using the provided pattern.
        Supports '*' (matches any number of characters) and '?' (matches any single character).
        """
        if pattern == "*":
            return True
        if text is None or pattern is None:
            return False
        if not pattern:
            return not text

        # Fast-paths for common wildcard patterns to avoid regex overhead
        if "?" not in pattern:
            match = _Matcher._fast_wild_card_match(text, pattern)
            if match is not None:
                return match

        # Fallback to regex for complex patterns
        return re.fullmatch(_Matcher._get_compiled_regex(pattern), text) is not None

    @staticmethod
    def _fast_wild_card_match(text: str, pattern: str) -> bool:
        star_count = pattern.count("*")
        if star_count == 0:
            return text == pattern
        if star_count == 1:
            if pattern.endswith("*"):  # prefix*
                return text.startswith(pattern[:-1])
            if pattern.startswith("*"):  # *suffix
                return text.endswith(pattern[1:])
        elif star_count == 2 and pattern.startswith("*") and pattern.endswith("*"):  # *middle*
            return pattern[1:-1] in text
        return None

    @staticmethod
    @lru_cache(maxsize=1024)
    def _get_compiled_regex(pattern: str):
        return re.compile(_Matcher.to_regex_pattern(pattern))

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
        """
        Checks if the given attributes match the rule attributes.
        Iterates over rule attributes (O(R)) and performs early return for efficiency.
        """
        if not rule_attributes:
            return True
        if not attributes or len(rule_attributes) > len(attributes):
            return False

        for key, pattern in rule_attributes.items():
            val = attributes.get(key)
            if val is None:
                return False
            # Ensure value is a string for wildcard matching
            if not _Matcher.wild_card_match(str(val), pattern):
                return False
        return True
