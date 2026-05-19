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


@lru_cache(maxsize=1024)
def _get_compiled_regex(pattern: str):
    """
    Memoize the compilation of regex patterns derived from wildcards.
    """
    return re.compile(_Matcher.to_regex_pattern(pattern))


class _Matcher:
    @staticmethod
    def wild_card_match(text: str = None, pattern: str = None) -> bool:
        """
        Match text against a wildcard pattern. Supports '*' (zero or more chars)
        and '?' (exactly one char).
        """
        if pattern == "*":
            return True
        if text is None or pattern is None:
            return False
        if len(pattern) == 0:
            return len(text) == 0

        # Fast paths for common wildcard scenarios to avoid regex overhead
        if "?" not in pattern:
            match = _Matcher._fast_wild_card_match(text, pattern)
            if match is not None:
                return match

        # Fallback to regex for complex patterns
        return _get_compiled_regex(pattern).fullmatch(text) is not None

    @staticmethod
    def _fast_wild_card_match(text: str, pattern: str) -> bool | None:
        star_count = pattern.count("*")
        if star_count == 0:
            return pattern == text
        if star_count == 1:
            if pattern[0] == "*":  # *suffix
                return text.endswith(pattern[1:])
            if pattern[-1] == "*":  # prefix*
                return text.startswith(pattern[:-1])
        elif star_count == 2 and pattern[0] == "*" and pattern[-1] == "*":  # *middle*
            return pattern[1:-1] in text
        return None

    @staticmethod
    def to_regex_pattern(rule_pattern: str) -> str:
        """
        Convert a wildcard pattern to a regex string.
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
        Match a set of attributes against a set of rule attributes.
        Optimized to iterate over rule attributes and return early on mismatch.
        """
        if not rule_attributes:
            return True
        if attributes is None or len(rule_attributes) > len(attributes):
            return False

        for key, pattern in rule_attributes.items():
            val = attributes.get(key)
            if val is None:
                return False
            # Wildcard match expects string input
            if not _Matcher.wild_card_match(str(val), pattern):
                return False
        return True
