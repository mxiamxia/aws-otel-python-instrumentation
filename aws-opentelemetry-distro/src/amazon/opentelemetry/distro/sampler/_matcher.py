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
def _get_compiled_regex(pattern: str) -> re.Pattern:
    return re.compile(_Matcher.to_regex_pattern(pattern))


class _Matcher:
    @staticmethod
    def wild_card_match(text: str = None, pattern: str = None) -> bool:
        """
        Matches text against a pattern that can contain wildcards '*' (zero or more characters)
        and '?' (exactly one character).
        """
        if pattern == "*":
            return True
        if text is None or pattern is None:
            return False

        pattern_len = len(pattern)
        if pattern_len == 0:
            return len(text) == 0

        # Fast paths for common cases without '?' and with simple '*'
        # This bypasses the regex engine for common patterns like 'exact', 'prefix*', '*suffix', '*middle*'
        if "?" not in pattern:
            if "*" not in pattern:
                return pattern == text

            first_star = pattern.find("*")
            last_star = pattern.rfind("*")

            if first_star == last_star:  # Only one star
                if first_star == 0:  # *suffix
                    return text.endswith(pattern[1:])
                if first_star == pattern_len - 1:  # prefix*
                    return text.startswith(pattern[:-1])
            elif first_star == 0 and last_star == pattern_len - 1:  # *middle*
                middle = pattern[1:-1]
                if "*" not in middle:
                    return middle in text

        # Fallback to regex with cache to avoid repeated pattern conversion and compilation
        return _get_compiled_regex(pattern).fullmatch(text) is not None

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
        Matches attributes against rule attributes.
        Optimized to iterate over rule attributes (usually smaller set) and use early return.
        """
        if rule_attributes is None or len(rule_attributes) == 0:
            return True
        if attributes is None or len(attributes) == 0:
            return False

        for key, pattern in rule_attributes.items():
            val = attributes.get(key)
            if val is None:
                return False
            if not _Matcher.wild_card_match(val, pattern):
                return False
        return True
