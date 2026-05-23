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
    def wild_card_match(text: str = None, pattern: str = None) -> bool:
        if pattern == "*":
            return True
        if text is None or pattern is None:
            return False

        if len(pattern) == 0:
            return len(text) == 0

        fast_match = _Matcher._wild_card_match_fast_path(text, pattern)
        if fast_match is not None:
            return fast_match

        # Fallback to regex with caching
        return _get_compiled_regex(pattern).fullmatch(text) is not None

    @staticmethod
    def _wild_card_match_fast_path(text: str, pattern: str) -> bool:
        """Internal fast-path for common wildcard patterns to avoid regex overhead."""
        pattern_len = len(pattern)

        if "*" not in pattern and "?" not in pattern:
            return pattern == text

        if pattern_len >= 2 and pattern[0] == "*" and pattern[-1] == "*":
            middle = pattern[1:-1]
            if "*" not in middle and "?" not in middle:
                return middle in text

        if pattern[-1] == "*":
            prefix = pattern[:-1]
            if "*" not in prefix and "?" not in prefix:
                return text.startswith(prefix)

        if pattern[0] == "*":
            suffix = pattern[1:]
            if "*" not in suffix and "?" not in suffix:
                return text.endswith(suffix)

        return None

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


@functools.lru_cache(maxsize=1024)
def _get_compiled_regex(pattern: str) -> re.Pattern:
    """Helper to cache compiled regex objects for wildcard matching."""
    return re.compile(_Matcher.to_regex_pattern(pattern))
