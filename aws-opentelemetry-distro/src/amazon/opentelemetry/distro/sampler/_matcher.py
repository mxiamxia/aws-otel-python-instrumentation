# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import functools
import re
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
    def wild_card_match(text: Optional[str] = None, pattern: Optional[str] = None) -> bool:
        if pattern == "*":
            return True
        if text is None or pattern is None:
            return False
        pattern_len = len(pattern)
        if pattern_len == 0:
            return len(text) == 0

        # Optimization: use fast path for common patterns or pre-compiled regex
        first_star = pattern.find("*")
        first_ques = pattern.find("?")

        if first_star == -1 and first_ques == -1:
            return pattern == text

        # Fast paths for single '*' at start or end, or double '*' at both ends to avoid regex
        if first_ques == -1:
            if first_star == pattern_len - 1:  # "prefix*"
                return text.startswith(pattern[:-1])
            if first_star == 0:
                last_star = pattern.find("*", 1)
                if last_star == -1:  # "*suffix"
                    return text.endswith(pattern[1:])
                if last_star == pattern_len - 1 and pattern.find("*", 1, last_star) == -1:  # "*contains*"
                    return pattern[1:last_star] in text

        return _Matcher._get_compiled_regex(pattern).fullmatch(text) is not None

    @staticmethod
    @functools.lru_cache(maxsize=1024)
    def _get_compiled_regex(pattern: str) -> re.Pattern:
        """Returns a compiled regex for the given wildcard pattern, cached for performance."""
        return re.compile(_Matcher.to_regex_pattern(pattern))

    @staticmethod
    @functools.lru_cache(maxsize=1024)
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
