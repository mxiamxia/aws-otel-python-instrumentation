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
        Matches text against a wildcard pattern. Supports * (zero or more chars) and ? (exactly one char).
        """
        if pattern == "*":
            return True
        if text is None or pattern is None:
            return False
        if len(pattern) == 0:
            return len(text) == 0

        # Optimization: fast-path for common wildcard patterns without '?'
        if "?" not in pattern:
            fast_path_match = _Matcher._wild_card_match_fast_path(text, pattern)
            if fast_path_match is not None:
                return fast_path_match

        if "*" in pattern or "?" in pattern:
            return _Matcher.to_regex_pattern(pattern).fullmatch(text) is not None

        return pattern == text

    @staticmethod
    def _wild_card_match_fast_path(text: str, pattern: str) -> bool:
        wildcard_count = pattern.count("*")
        if wildcard_count == 1:
            if pattern.startswith("*"):
                return text.endswith(pattern[1:])
            if pattern.endswith("*"):
                return text.startswith(pattern[:-1])
        elif wildcard_count == 2 and pattern.startswith("*") and pattern.endswith("*"):
            return pattern[1:-1] in text
        return None

    @staticmethod
    @lru_cache(maxsize=1024)
    def to_regex_pattern(rule_pattern: str) -> re.Pattern:
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
        return re.compile(regex_pattern)

    @staticmethod
    def attribute_match(attributes: Attributes = None, rule_attributes: dict = None) -> bool:
        """
        Matches span attributes against rule attributes.
        """
        if not rule_attributes:
            return True
        if not attributes or len(rule_attributes) > len(attributes):
            return False

        # Optimization: iterate over rule_attributes (smaller set) and return False early on mismatch.
        for key, pattern in rule_attributes.items():
            val = attributes.get(key)
            if val is None or not _Matcher.wild_card_match(str(val), pattern):
                return False

        return True
