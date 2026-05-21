# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import functools
import re
from typing import Tuple

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
    @functools.lru_cache(maxsize=1024)
    def _get_regex(pattern: str):
        return re.compile(_Matcher.to_regex_pattern(pattern))

    @staticmethod
    def wild_card_match(text: str = None, pattern: str = None) -> bool:
        """
        Matches a string against a wildcard pattern.
        '*' matches any number of characters.
        '?' matches any single character.
        """
        if pattern == "*":
            return True
        if text is None or pattern is None:
            return False
        if len(pattern) == 0:
            return len(text) == 0

        # Fast path for exact match
        if "*" not in pattern and "?" not in pattern:
            return text == pattern

        # Fast paths for simple wildcard patterns without '?'
        if "?" not in pattern:
            is_match, result = _Matcher._fast_wild_card_match(text, pattern)
            if is_match:
                return result

        return _Matcher._get_regex(pattern).fullmatch(text) is not None

    @staticmethod
    def _fast_wild_card_match(text: str, pattern: str) -> Tuple[bool, bool]:
        star_count = pattern.count("*")
        if star_count == 1:
            if pattern.endswith("*"):
                return True, text.startswith(pattern[:-1])
            if pattern.startswith("*"):
                return True, text.endswith(pattern[1:])
            parts = pattern.split("*")
            return (
                True,
                len(text) >= len(pattern) - 1 and text.startswith(parts[0]) and text.endswith(parts[1]),
            )
        if star_count == 2 and pattern.startswith("*") and pattern.endswith("*"):
            # *middle*
            return True, pattern[1:-1] in text
        return False, False

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
        if rule_attributes is None or len(rule_attributes) == 0:
            return True
        if attributes is None or len(attributes) == 0 or len(rule_attributes) > len(attributes):
            return False

        matched_count = 0
        for key, val in attributes.items():
            text_to_match = val
            pattern = rule_attributes.get(key, None)
            if pattern is None:
                continue
            if _Matcher.wild_card_match(text_to_match, pattern):
                matched_count += 1
        return matched_count == len(rule_attributes)
