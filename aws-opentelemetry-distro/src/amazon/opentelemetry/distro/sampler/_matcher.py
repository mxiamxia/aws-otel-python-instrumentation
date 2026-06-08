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
        if pattern == "*":
            return True
        if text is None or pattern is None:
            return False
        if len(pattern) == 0:
            return len(text) == 0

        # Fast paths for common patterns
        star_count = pattern.count("*")
        question_count = pattern.count("?")

        if star_count == 0 and question_count == 0:
            return text == pattern

        if star_count == 1 and question_count == 0:
            if pattern.startswith("*"):
                return text.endswith(pattern[1:])
            if pattern.endswith("*"):
                return text.startswith(pattern[:-1])
            parts = pattern.split("*")
            return text.startswith(parts[0]) and text.endswith(parts[1]) and len(text) >= len(pattern) - 1

        if star_count == 2 and question_count == 0 and pattern.startswith("*") and pattern.endswith("*"):
            return pattern[1:-1] in text

        return re.fullmatch(_Matcher.to_regex_pattern(pattern), text) is not None

    @staticmethod
    @lru_cache(maxsize=1024)
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

        for key, pattern in rule_attributes.items():
            val = attributes.get(key)
            if val is None:
                return False
            # Ensure text_to_match is a string as wild_card_match expects it
            if not _Matcher.wild_card_match(str(val) if val is not None else None, pattern):
                return False
        return True
