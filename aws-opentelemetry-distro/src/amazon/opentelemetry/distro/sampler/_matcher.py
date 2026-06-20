# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import re
from functools import lru_cache
from typing import Optional

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
        Matches a text against a pattern that can contain wildcards '*' and '?'.
        Optimized with fast-paths for common patterns and regex caching.
        """
        # Fast-path for '*' pattern which matches everything
        if pattern == "*":
            return True
        if text is None or pattern is None:
            return False
        if len(pattern) == 0:
            return len(text) == 0

        # Check for common patterns to bypass regex for better performance
        fast_path_result = _Matcher._wild_card_match_fast_path(text, pattern)
        if fast_path_result is not None:
            return fast_path_result

        # Fallback to regex for complex patterns
        return re.fullmatch(_Matcher.to_regex_pattern(pattern), text) is not None

    @staticmethod
    def _wild_card_match_fast_path(text: str, pattern: str) -> Optional[bool]:
        """
        Implements fast-path matching for common wildcard patterns to avoid regex overhead.
        Supported patterns: prefix*, *suffix, *middle* (where * is the only wildcard).
        """
        wildcard_count = pattern.count("*")
        question_count = pattern.count("?")

        # If no wildcards, perform an exact match
        if wildcard_count == 0 and question_count == 0:
            return text == pattern

        # Handle patterns with exactly one '*' and no '?'
        if wildcard_count == 1 and question_count == 0:
            if pattern.endswith("*"):  # prefix*
                return text.startswith(pattern[:-1])
            if pattern.startswith("*"):  # *suffix
                return text.endswith(pattern[1:])

        # Handle patterns like *middle* (exactly two '*' at ends and no '?' in between)
        if wildcard_count == 2 and question_count == 0:
            if pattern.startswith("*") and pattern.endswith("*"):
                return pattern[1:-1] in text

        return None

    @staticmethod
    @lru_cache(maxsize=1024)
    def to_regex_pattern(rule_pattern: str) -> str:
        """
        Converts a wildcard pattern to a regular expression pattern.
        Results are cached to avoid redundant processing.
        """
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
        Matches span attributes against rule attributes.
        Optimized to iterate over rule attributes for early return and better performance
        when the number of span attributes is much larger than rule attributes.
        """
        if rule_attributes is None or len(rule_attributes) == 0:
            return True
        if attributes is None or len(attributes) == 0:
            return False

        # Performance optimization: iterate over the set of rule attributes (O(R))
        # instead of span attributes (O(A)), as R is usually smaller than A.
        for key, pattern in rule_attributes.items():
            val = attributes.get(key)
            if val is None:
                return False

            # Ensure value is converted to string for wildcard matching
            if not _Matcher.wild_card_match(str(val), pattern):
                return False

        return True
