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
        if pattern == "*":
            return True
        if text is None or pattern is None:
            return False
        if len(pattern) == 0:
            return len(text) == 0

        # Fast-paths for common patterns
        fast_path_result = _Matcher._try_fast_path_match(text, pattern)
        if fast_path_result is not None:
            return fast_path_result

        for char in pattern:
            if char in ("*", "?"):
                return _Matcher.compiled_regex_match(pattern, text)
        return pattern == text

    @staticmethod
    def _try_fast_path_match(text: str, pattern: str) -> Optional[bool]:
        if "*" in pattern and "?" not in pattern:
            # Check for prefix*
            if pattern.endswith("*") and pattern.count("*") == 1:
                return text.startswith(pattern[:-1])
            # Check for *suffix
            if pattern.startswith("*") and pattern.count("*") == 1:
                return text.endswith(pattern[1:])
            # Check for *middle*
            if pattern.startswith("*") and pattern.endswith("*") and pattern.count("*") == 2:
                return pattern[1:-1] in text
        return None

    @staticmethod
    @lru_cache(maxsize=1024)
    def compiled_regex_match(pattern: str, text: str) -> bool:
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
        if rule_attributes is None or len(rule_attributes) == 0:
            return True
        if attributes is None or len(attributes) == 0:
            return False

        # Optimized: Iterate over rule attributes O(R) and early-return on mismatch.
        for key, pattern in rule_attributes.items():
            text_to_match = attributes.get(key)
            if text_to_match is None or not _Matcher.wild_card_match(text_to_match, pattern):
                return False
        return True
