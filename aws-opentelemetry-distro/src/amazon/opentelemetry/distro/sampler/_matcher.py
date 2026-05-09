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
        Match text against a wildcard pattern. Supports '*' for multiple characters
        and '?' for a single character.
        """
        if pattern == "*":
            return True
        if text is None or pattern is None:
            return False
        if len(pattern) == 0:
            return len(text) == 0

        # Optimization: check if it's an exact match before regex overhead
        if "*" not in pattern and "?" not in pattern:
            return pattern == text

        return re.fullmatch(_Matcher.to_regex_pattern(pattern), text) is not None

    @staticmethod
    @lru_cache(maxsize=1024)
    def to_regex_pattern(rule_pattern: str) -> str:
        """
        Convert a wildcard pattern to a regex pattern. Cached to avoid redundant processing.
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
        Match span attributes against rule attributes.
        Optimized to iterate over rule_attributes (usually much smaller O(R))
        instead of span attributes (can be large O(A)).
        """
        if not rule_attributes:
            return True
        if not attributes or len(rule_attributes) > len(attributes):
            return False

        # Optimization: Iterate over rule_attributes instead of attributes.
        # This is more efficient because rule_attributes is typically a small subset
        # of the total span attributes.
        for key, pattern in rule_attributes.items():
            val = attributes.get(key)
            if val is None:
                return False
            if not _Matcher.wild_card_match(val, pattern):
                return False

        return True
