## 2026-05-29 - Optimize attribute matching by iterating over rules
**Learning:** In OpenTelemetry sampling rules, the number of attributes to match against (R) is typically much smaller than the total number of attributes on a span (A). The original implementation iterated over A, resulting in O(A) complexity. Switching to iterate over R achieves O(R) complexity.
**Action:** In performance-critical paths involving attribute filtering or matching, always iterate over the smaller set of keys (usually the rules/filters) and use early returns to optimize execution time.
