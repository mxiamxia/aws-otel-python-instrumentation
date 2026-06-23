## 2025-05-15 - Optimizing _Matcher Performance
**Learning:** In performance-critical paths like the X-Ray sampler, iterating over the set of rules (R)$ instead of the set of span attributes (A)$ is significantly more efficient because  \ll A$ in most production scenarios. Additionally, Python's Pylint configuration enforces a strict limit of 6 return statements (R0911), requiring complex matching logic to be refactored into private helper methods.

**Action:** Always prioritize iterating over the smaller known set of keys in matching logic and use private helper methods to keep function complexity within linting limits while implementing fast-path optimizations.
