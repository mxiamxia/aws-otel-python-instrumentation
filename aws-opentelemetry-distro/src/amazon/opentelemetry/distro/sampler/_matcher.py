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

        pat_len = len(pattern)
        if pat_len == 0:
            return len(text) == 0

        if "*" not in pattern and "?" not in pattern:
            return pattern == text

        # Fast-paths for common wildcard patterns to avoid regex overhead
        res = _Matcher._try_fast_wild_card_match(text, pattern, pat_len)
        if res is not None:
            return res

        # Fallback to regex for complex patterns
        return _get_compiled_regex(pattern).fullmatch(text) is not None

    @staticmethod
    def _try_fast_wild_card_match(text: str, pattern: str, pat_len: int) -> Optional[bool]:
        if "?" not in pattern:
            star_count = pattern.count("*")
            if star_count == 1:
                if pattern[0] == "*":
                    return text.endswith(pattern[1:])
                if pattern[pat_len - 1] == "*":
                    return text.startswith(pattern[:-1])
            if star_count == 2 and pattern[0] == "*" and pattern[pat_len - 1] == "*":
                return pattern[1:-1] in text
        return None

    @staticmethod
    def to_regex_pattern(rule_pattern: str) -> str:
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
        if rule_attributes is None or len(rule_attributes) == 0:
            return True
        if attributes is None or len(attributes) == 0 or len(rule_attributes) > len(attributes):
            return False

        # Optimized to iterate over rule_attributes (O(R)) instead of span attributes (O(A))
        # and early return on the first mismatch.
        for key, pattern in rule_attributes.items():
            val = attributes.get(key)
            if val is None or not _Matcher.wild_card_match(val, pattern):
                return False
        return True


@lru_cache(maxsize=1024)
def _get_compiled_regex(pattern: str) -> re.Pattern:
    return re.compile(_Matcher.to_regex_pattern(pattern))
