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
        Matches text against a pattern that can contain '*' (matches 0 or more characters)
        and '?' (matches exactly one character).
        """
        if pattern == "*":
            return True
        if text is None or pattern is None:
            return False
        if len(pattern) == 0:
            return len(text) == 0

        # Fast path for patterns without wildcards
        if "*" not in pattern and "?" not in pattern:
            return pattern == text

        # Fast path for common single-wildcard patterns
        if pattern.count("*") == 1 and "?" not in pattern:
            if pattern.startswith("*"):
                return text.endswith(pattern[1:])
            if pattern.endswith("*"):
                return text.startswith(pattern[:-1])
            parts = pattern.split("*")
            # Ensure text is long enough and matches both prefix and suffix
            return len(text) >= len(pattern) - 1 and text.startswith(parts[0]) and text.endswith(parts[1])

        # Fast path for *middle* pattern
        if pattern.count("*") == 2 and pattern.startswith("*") and pattern.endswith("*") and "?" not in pattern:
            return pattern[1:-1] in text

        return re.fullmatch(_Matcher.to_regex_pattern(pattern), text) is not None

    @staticmethod
    @lru_cache(maxsize=1024)
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
        Matches a set of span attributes against a set of rule attributes.
        Returns True if all rule attributes are present in span attributes and match the patterns.
        """
        if not rule_attributes:
            return True
        if not attributes or len(rule_attributes) > len(attributes):
            return False

        # Iterate over rule attributes instead of span attributes to allow early return
        # and better performance when len(rule_attributes) < len(attributes).
        for key, pattern in rule_attributes.items():
            val = attributes.get(key)
            if val is None:
                return False
            if not _Matcher.wild_card_match(str(val), pattern):
                return False
        return True
