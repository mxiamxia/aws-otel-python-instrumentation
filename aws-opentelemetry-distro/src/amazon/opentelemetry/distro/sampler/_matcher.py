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
        if pattern == "*":
            return True
        if text is None or pattern is None:
            return False
        if not pattern:
            return not text

        # Fast path for common patterns without '?'
        if "?" not in pattern:
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

        return _Matcher._cached_regex_match(pattern, text)

    @staticmethod
    @lru_cache(maxsize=1024)
    def _cached_regex_match(pattern: str, text: str) -> bool:
        regex = re.compile(_Matcher.to_regex_pattern(pattern))
        return regex.fullmatch(text) is not None

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
        if not rule_attributes:
            return True
        if not attributes:
            return False

        for key, pattern in rule_attributes.items():
            value = attributes.get(key)
            if value is None or not _Matcher.wild_card_match(value, pattern):
                return False
        return True
