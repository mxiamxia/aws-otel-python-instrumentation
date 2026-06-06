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
def _get_compiled_regex(pattern: str) -> re.Pattern:
    return re.compile(_Matcher.to_regex_pattern(pattern))


class _Matcher:
    @staticmethod
    def wild_card_match(text: str = None, pattern: str = None) -> bool:
        if pattern == "*":
            return True
        if text is None or pattern is None:
            return False

        pattern_len = len(pattern)
        if pattern_len == 0:
            return len(text) == 0

        # Fast path for common cases
        if "*" not in pattern and "?" not in pattern:
            return pattern == text

        return _Matcher._wild_card_match_complex(text, pattern)

    @staticmethod
    def _wild_card_match_complex(text: str, pattern: str) -> bool:
        if "?" not in pattern:
            if pattern.count("*") == 1:
                if pattern.endswith("*"):
                    # prefix*
                    return text.startswith(pattern[:-1])
                if pattern.startswith("*"):
                    # *suffix
                    return text.endswith(pattern[1:])
            elif pattern.count("*") == 2 and pattern.startswith("*") and pattern.endswith("*"):
                # *middle*
                return pattern[1:-1] in text

        return re.fullmatch(_get_compiled_regex(pattern), text) is not None

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
        if not rule_attributes:
            return True
        if not attributes or len(rule_attributes) > len(attributes):
            return False

        for key, pattern in rule_attributes.items():
            text_to_match = attributes.get(key)
            if text_to_match is None or not _Matcher.wild_card_match(text_to_match, pattern):
                return False
        return True
