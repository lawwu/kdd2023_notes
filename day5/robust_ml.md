# Robust ML

# Generative Perturbation Analysis for Probabilistic Black-Box Anomaly Attribution

- LIME is the derivate of IG?
- Paper: https://dl.acm.org/doi/10.1145/3580305.3599365
- Code: https://github.com/Idesan/gpa
- Perturbation as explanation: goal is to find a perturbation of gamma
- Method is called GPA: Generative Perturbation Aanlysis
- "Why does this house look so unusual?" House hunting use case [slide]
    - Boston housing data
    - Computed attribution scores for the top outlier
    - GPA returns anomaly score
- "Why does this patient look so unusual?" Healthcare use-case
    - Diabetes data
    - Computed attribution score for the top outlier
- Summary
   - GPA is the first black-box attribution framework allowing probabilistic attribution.
   - We have showed a strong impossibility result: LIME, S, and I are deviation-agnostic, and hence, not suitable for anomaly attribution.
   - We have also uncovered a relationship between LIME, Sv, and IG for the first
time.