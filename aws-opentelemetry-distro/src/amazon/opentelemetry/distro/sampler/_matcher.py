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
        Match text against a pattern with wildcards (* and ?).
        Includes fast-paths for common patterns and regex caching for others.
        """
        if pattern == "*":
            return True
        if text is None or pattern is None:
            return False
        if len(pattern) == 0:
            return len(text) == 0

        # Fast-paths for common patterns without using the regex engine
        wildcard_count = pattern.count("*") + pattern.count("?")
        if wildcard_count == 0:
            return pattern == text

        fast_path_result = _Matcher._wild_card_match_fast_path(text, pattern, wildcard_count)
        if fast_path_result is not None:
            return fast_path_result

        # Fallback to regex for more complex patterns
        return _Matcher._get_compiled_regex(pattern).fullmatch(text) is not None

    @staticmethod
    def _wild_card_match_fast_path(text: str, pattern: str, wildcard_count: int) -> bool:
        """Helper to handle fast-path wildcard matches and keep return count low."""
        if wildcard_count == 1:
            if pattern.endswith("*"):
                return text.startswith(pattern[:-1])
            if pattern.startswith("*"):
                return text.endswith(pattern[1:])
        elif (
            wildcard_count == 2
            and pattern.startswith("*")
            and pattern.endswith("*")
            and len(pattern) >= 2
        ):
            return pattern[1:-1] in text
        return None

    @staticmethod
    @functools.lru_cache(maxsize=1024)
    def _get_compiled_regex(pattern: str) -> re.Pattern:
        """Cache the compilation of wildcard patterns into regular expressions."""
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
        Verify if the given span attributes match all the required rule attributes.
        Optimized to iterate over rule attributes (O(R)) and early-return on mismatch.
        """
        if not rule_attributes:
            return True
        if not attributes:
            return False

        # Iterate over rule attributes (typically much fewer than span attributes)
        for key, pattern in rule_attributes.items():
            val = attributes.get(key)
            if val is None:
                return False

            # Ensure the attribute value is a string for wildcard matching
            text_to_match = str(val)
            if not _Matcher.wild_card_match(text_to_match, pattern):
                # Early return on first mismatch
                return False

        return True
