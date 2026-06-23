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
        """Matches text against a pattern that may contain wildcards '*' and '?'.

        Optimized with fast-paths for common patterns and regex caching.
        """
        if pattern == "*":
            return True
        if text is None or pattern is None:
            return False
        if len(pattern) == 0:
            return len(text) == 0

        # Fast-path evaluation for common patterns
        fast_path_result = _Matcher._wild_card_match_fast_path(text, pattern)
        if fast_path_result is not None:
            return fast_path_result

        # Fallback to regex matching for complex patterns
        return re.fullmatch(_Matcher.to_regex_pattern(pattern), text) is not None

    @staticmethod
    def _wild_card_match_fast_path(text: str, pattern: str) -> Optional[bool]:
        """Evaluates fast-paths for common wildcard patterns."""
        wildcard_count = pattern.count("*") + pattern.count("?")
        if wildcard_count == 0:
            return pattern == text

        # Only handle simple single-wildcard patterns in fast-path
        if wildcard_count == 1 and pattern.count("*") == 1:
            if pattern.startswith("*"):
                return text.endswith(pattern[1:])
            if pattern.endswith("*"):
                return text.startswith(pattern[:-1])
            parts = pattern.split("*")
            return text.startswith(parts[0]) and text.endswith(parts[1]) and len(text) >= len(pattern) - 1

        return None

    @staticmethod
    @lru_cache(maxsize=1024)
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
        """Matches span attributes against a set of rule attributes.

        Optimized by iterating over rule attributes O(R) and early-returning on first mismatch.
        """
        if rule_attributes is None or len(rule_attributes) == 0:
            return True
        if attributes is None or len(attributes) == 0:
            return False

        for key, pattern in rule_attributes.items():
            val = attributes.get(key)
            if val is None:
                return False
            # Ensure value is converted to string for matching
            if not _Matcher.wild_card_match(str(val), pattern):
                return False
        return True
