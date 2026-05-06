## 2026-05-06 - Optimized attribute matching in sampler
**Learning:** In the AWS X-Ray sampler, matching span attributes against sampling rules was implemented by iterating over all span attributes ($O(A)$). Since spans can have dozens of attributes while rules typically specify only a few, iterating over rule attributes ($O(R)$) is significantly more efficient.
**Action:** Always prefer iterating over the smaller set of keys (usually the rule/filter criteria) when performing attribute matching or filtering in performance-critical paths like the sampling logic.
