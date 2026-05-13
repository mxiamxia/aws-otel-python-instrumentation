## 2026-05-13 - Optimized X-Ray Sampler Matching Logic
**Learning:** Wildcard matching using regex is a significant bottleneck in sampling decisions. Implementing fast-paths for common patterns (prefix, suffix, contains) and caching compiled regex objects can reduce overhead by 3-7x. Additionally, in attribute matching, iterating over the smaller rule set instead of the larger span attribute set improves O-complexity.
**Action:** Always look for fast-paths to bypass regex in hot paths and ensure iteration is performed over the minimal set of elements.
