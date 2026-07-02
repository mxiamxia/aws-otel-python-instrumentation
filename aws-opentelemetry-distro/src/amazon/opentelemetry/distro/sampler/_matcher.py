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
        Matches a string against a pattern with wildcards (* and ?).
        Optimized with fast-paths for common patterns and regex caching.
        """
        if pattern == "*":
            return True
        if text is None or pattern is None:
            return False
        if len(pattern) == 0:
            return len(text) == 0

        # Bolt Optimization: Fast-path for common wildcard patterns to avoid regex overhead.
        # This covers cases like 'prefix*', '*suffix', '*middle*', and 'exact_match'.
        if "?" not in pattern:
            fast_path_result = _Matcher._wild_card_match_fast_path(text, pattern)
            if fast_path_result is not None:
                return fast_path_result

        # Fallback to regex for more complex patterns.
        # Bolt Optimization: to_regex_pattern is cached via lru_cache.
        return re.fullmatch(_Matcher.to_regex_pattern(pattern), text) is not None

    @staticmethod
    def _wild_card_match_fast_path(text: str, pattern: str) -> bool:
        """Helper for fast-path wildcard evaluation to reduce return statement count in main method."""
        star_count = pattern.count("*")
        if star_count == 0:
            return pattern == text
        if star_count == 1:
            if pattern.startswith("*"):
                return text.endswith(pattern[1:])
            if pattern.endswith("*"):
                return text.startswith(pattern[:-1])
        elif star_count == 2 and pattern.startswith("*") and pattern.endswith("*"):
            return pattern[1:-1] in text
        return None

    @staticmethod
    @lru_cache(maxsize=1024)
    def to_regex_pattern(rule_pattern: str) -> re.Pattern:
        """
        Converts a wildcard pattern to a compiled regex pattern.
        Bolt Optimization: Cached to avoid redundant string processing and compilation.
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
        return re.compile(regex_pattern)

    @staticmethod
    def attribute_match(attributes: Attributes = None, rule_attributes: dict = None) -> bool:
        """
        Checks if all rule attributes match the provided span attributes.
        Bolt Optimization: Iterates over rule_attributes for early return and better performance
        when the span has many attributes.
        """
        if rule_attributes is None or len(rule_attributes) == 0:
            return True
        if attributes is None or len(attributes) == 0:
            return False

        # Bolt Optimization: Iterate over the rules instead of all span attributes.
        # This is more efficient as typically len(rule_attributes) << len(attributes).
        # Also allows early return on first mismatch.
        for key, pattern in rule_attributes.items():
            val = attributes.get(key)
            if val is None:
                return False
            # Ensure value is treated as string for matching
            if not _Matcher.wild_card_match(str(val), pattern):
                return False

        return True
