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
        Match text against a wildcard pattern.
        Optimized with fast-paths for common patterns and regex caching.
        """
        if pattern == "*":
            return True
        if text is None or pattern is None:
            return False
        if len(pattern) == 0:
            return len(text) == 0

        # Fast-paths for common wildcard patterns to avoid regex overhead
        if "*" not in pattern and "?" not in pattern:
            return text == pattern

        return _Matcher._wild_card_match_complex(text, pattern)

    @staticmethod
    def _wild_card_match_complex(text: str, pattern: str) -> bool:
        count_star = pattern.count("*")
        count_question = pattern.count("?")

        if count_question == 0:
            if count_star == 1:
                if pattern[0] == "*":  # *suffix
                    return text.endswith(pattern[1:])
                if pattern[-1] == "*":  # prefix*
                    return text.startswith(pattern[:-1])
            if count_star == 2 and pattern[0] == "*" and pattern[-1] == "*":  # *middle*
                return pattern[1:-1] in text

        return re.fullmatch(_Matcher.to_regex_pattern(pattern), text) is not None

    @staticmethod
    @lru_cache(maxsize=1024)
    def to_regex_pattern(rule_pattern: str) -> str:
        """
        Convert wildcard pattern to regex pattern.
        Cached to avoid redundant regex generation and compilation.
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
        Optimized to iterate over rule attributes (usually fewer) and early return.
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
