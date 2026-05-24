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
    @functools.lru_cache(maxsize=1024)
    def _get_regex(pattern: str) -> re.Pattern:
        return re.compile(_Matcher.to_regex_pattern(pattern))

    @staticmethod
    def wild_card_match(text: str = None, pattern: str = None) -> bool:
        """
        Matches a string against a wildcard pattern.
        Supported wildcards are '*' (matches 0 or more characters) and '?' (matches exactly 1 character).
        """
        if pattern == "*":
            return True
        if text is None or pattern is None:
            return False

        pattern_len = len(pattern)
        if pattern_len == 0:
            return len(text) == 0

        # Fast paths for common patterns to avoid regex overhead
        matched, result = _Matcher._fast_path_match(text, pattern)
        if matched:
            return result

        return _Matcher._get_regex(pattern).fullmatch(text) is not None

    @staticmethod
    def _fast_path_match(text: str, pattern: str) -> tuple:
        if "*" not in pattern and "?" not in pattern:
            return True, text == pattern

        # Simple prefix match: "prefix*"
        if pattern.endswith("*") and "*" not in pattern[:-1] and "?" not in pattern:
            return True, text.startswith(pattern[:-1])

        # Simple suffix match: "*suffix"
        if pattern.startswith("*") and "*" not in pattern[1:] and "?" not in pattern:
            return True, text.endswith(pattern[1:])

        # Simple middle match: "*middle*"
        if pattern.startswith("*") and pattern.endswith("*") and "*" not in pattern[1:-1] and "?" not in pattern:
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
