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
    def wild_card_match(text: str = None, pattern: str = None) -> bool:
        """
        Matches text against a wildcard pattern.
        Uses fast paths for common cases and cached regex for complex ones.
        """
        if pattern == "*":
            return True
        if text is None or pattern is None:
            return False
        if not pattern:
            return not text

        # Fast path check for exact, prefix, suffix, and substring matches
        fast_path_result = _Matcher._wild_card_match_fast_path(text, pattern)
        if fast_path_result is not None:
            return fast_path_result

        return _Matcher._to_regex_compiled(pattern).fullmatch(text) is not None

    @staticmethod
    def _wild_card_match_fast_path(text: str, pattern: str) -> Optional[bool]:
        """
        Handles exact, prefix, suffix, and substring matches without regex.
        Returns None if the pattern is too complex for these fast paths.
        """
        wildcard_count = pattern.count("*")
        question_count = pattern.count("?")

        # Exact match (no wildcards)
        if wildcard_count == 0 and question_count == 0:
            return text == pattern

        # Simple wildcard cases (only '*' and no '?')
        if question_count == 0:
            if wildcard_count == 1:
                if pattern.endswith("*"):  # Prefix match: "text*"
                    return text.startswith(pattern[:-1])
                if pattern.startswith("*"):  # Suffix match: "*text"
                    return text.endswith(pattern[1:])
            elif wildcard_count == 2 and pattern.startswith("*") and pattern.endswith("*"):
                # Substring match: "*text*"
                return pattern[1:-1] in text

        return None

    @staticmethod
    @functools.lru_cache(maxsize=1024)
    def _to_regex_compiled(pattern: str) -> re.Pattern:
        """Caches compiled regex patterns for wildcard matching."""
        return re.compile(_Matcher.to_regex_pattern(pattern))

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
        Matches span attributes against rule attributes.
        Optimized to iterate over rule attributes since they are typically fewer.
        """
        if not rule_attributes:
            return True
        if not attributes or len(rule_attributes) > len(attributes):
            return False

        for key, pattern in rule_attributes.items():
            val = attributes.get(key)
            if val is None or not _Matcher.wild_card_match(str(val), pattern):
                return False
        return True
