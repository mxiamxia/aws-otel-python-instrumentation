# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import re
from functools import lru_cache
from typing import Optional, Pattern

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
def _to_regex_pattern(rule_pattern: str) -> Pattern:
    """Converts a wildcard pattern to a compiled regex pattern with caching."""
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


class _Matcher:
    @staticmethod
    def wild_card_match(text: Optional[str] = None, pattern: Optional[str] = None) -> bool:
        """
        Matches text against a wildcard pattern. Supports '*' for multiple chars
        and '?' for a single char.
        """
        if pattern == "*":
            return True
        if text is None or pattern is None:
            return False
        if len(pattern) == 0:
            return len(text) == 0

        # Fast-path for common wildcard patterns to avoid regex overhead
        if "*" in pattern or "?" in pattern:
            return _Matcher._wild_card_match_fast_path(text, pattern)

        return pattern == text

    @staticmethod
    def _wild_card_match_fast_path(text: str, pattern: str) -> bool:
        """Helper for wildcard matching logic to keep wild_card_match clean."""
        # Check for simple prefix/suffix/contains patterns without '?'
        if "?" not in pattern:
            star_count = pattern.count("*")
            if star_count == 1:
                if pattern.startswith("*"):
                    return text.endswith(pattern[1:])
                if pattern.endswith("*"):
                    return text.startswith(pattern[:-1])
            elif star_count == 2 and pattern.startswith("*") and pattern.endswith("*"):
                return pattern[1:-1] in text

        return _to_regex_pattern(pattern).fullmatch(text) is not None

    @staticmethod
    def to_regex_pattern(rule_pattern: str) -> str:
        """Returns the regex string for a wildcard pattern. Maintained for compatibility."""
        return _to_regex_pattern(rule_pattern).pattern

    @staticmethod
    def attribute_match(attributes: Attributes = None, rule_attributes: dict = None) -> bool:
        """
        Checks if span attributes match the specified rule attributes.
        Optimized to iterate over rule attributes and early-return on mismatch.
        """
        if not rule_attributes:
            return True
        if not attributes:
            return False

        # Optimization: Iterate over rule_attributes instead of span attributes.
        # Rule attributes are typically fewer, and we can early-return on mismatch.
        for key, pattern in rule_attributes.items():
            val = attributes.get(key)
            if val is None:
                return False
            if not _Matcher.wild_card_match(str(val), pattern):
                return False
        return True
