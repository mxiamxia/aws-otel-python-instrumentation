import time
import re
from amazon.opentelemetry.distro.sampler._matcher import _Matcher

def benchmark_wildcard():
    patterns = [
        "Hello*",
        "*World",
        "Hell?W*d",
        "exact_match",
        "*",
        "a*b*c*d*e*f*g*h*i*j*k*l*m*n*o*p*q*r*s*t*u*v*w*x*y*z*",
    ]
    texts = ["HelloWorld", "HelloWorld", "HelloWorld", "exact_match", "anything", "abcdefghijklmnopqrstuvwxyz"]

    iterations = 10000
    start = time.time()
    for _ in range(iterations):
        for p, t in zip(patterns, texts):
            _Matcher.wild_card_match(t, p)
    end = time.time()
    print(f"Wildcard match took {end - start:.4f} seconds for {iterations * len(patterns)} matches")


def benchmark_attributes():
    attributes = {f"key{i}": f"value{i}" for i in range(100)}
    rule_attributes = {"key50": "value50", "key99": "value*"}

    iterations = 10000
    start = time.time()
    for _ in range(iterations):
        _Matcher.attribute_match(attributes, rule_attributes)
    end = time.time()
    print(f"Attribute match took {end - start:.4f} seconds for {iterations} matches")


if __name__ == "__main__":
    benchmark_wildcard()
    benchmark_attributes()
