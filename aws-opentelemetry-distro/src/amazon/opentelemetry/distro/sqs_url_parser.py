# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from functools import lru_cache
from typing import List, Optional, Tuple

from amazon.opentelemetry.distro._utils import is_account_id

_HTTP_SCHEMA: str = "http://"
_HTTPS_SCHEMA: str = "https://"


class SqsUrlParser:
    @staticmethod
    def get_queue_name(url: str) -> Optional[str]:
        """
        Best-effort logic to extract queue name from an HTTP url. This method should only be used with a string that is,
        with reasonably high confidence, an SQS queue URL. Handles new/legacy/some custom URLs. Essentially, we require
        that the URL should have exactly three parts, delimited by /'s (excluding schema), the second part should be an
        account id consisting of digits, and the third part should be a valid queue name, per SQS naming conventions.
        """
        return SqsUrlParser.parse_url(url)[0]

    @staticmethod
    def get_account_id(url: str) -> Optional[str]:
        """
        Extracts the account ID from an SQS URL.
        """
        return SqsUrlParser.parse_url(url)[1]

    @staticmethod
    def get_region(url: str) -> Optional[str]:
        """
        Extracts the region from an SQS URL.
        """
        return SqsUrlParser.parse_url(url)[2]

    @staticmethod
    @lru_cache(maxsize=1024)
    def parse_url(url: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Parses an SQS URL and extracts its components.
        URL Format: https://sqs.<region>.amazonaws.com/<accountId>/<queueName>
        """
        if url is None:
            return None, None, None

        # Optimization: use slicing instead of replace for fixed-length protocols
        if url.startswith(_HTTPS_SCHEMA):
            url_without_protocol = url[len(_HTTPS_SCHEMA) :]
        elif url.startswith(_HTTP_SCHEMA):
            url_without_protocol = url[len(_HTTP_SCHEMA) :]
        else:
            url_without_protocol = url

        split_url: List[str] = url_without_protocol.split("/")
        if len(split_url) != 3 or not is_account_id(split_url[1]) or not _is_valid_queue_name(split_url[2]):
            return None, None, None

        domain: str = split_url[0]
        domain_parts: List[str] = domain.split(".")

        # In case of custom URLs (e.g. 127.0.0.1:1212), domain_parts might not have a region
        region = domain_parts[1] if len(domain_parts) == 4 else None
        return split_url[2], split_url[1], region


def _is_valid_queue_name(input_str: str) -> bool:
    if input_str is None or len(input_str) == 0 or len(input_str) > 80:
        return False

    for char in input_str:
        if char != "_" and char != "-" and not char.isalpha() and not char.isdigit():
            return False

    return True
