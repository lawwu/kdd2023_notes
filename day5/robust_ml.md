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


# Doubly Robust AUC Optimization against Noisy and Adversarial Samples

Paper: https://dl.acm.org/doi/10.1145/3580305.3599316
Abstract: Area under the ROC curve (AUC) is an important and widely used metric in machine learning especially for imbalanced datasets. In current practical learning problems, not only adversarial samples but also noisy samples seriously threaten the performance of learning models. Nowadays, there have been a lot of research works proposed to defend the adversarial samples and noisy samples separately. Unfortunately, to the best of our knowledge, none of them with AUC optimization can secure against the two kinds of harmful samples simultaneously. To fill this gap and also address the challenge, in this paper, we propose a novel doubly robust dAUC optimization (DRAUC) algorithm. Specifically, we first exploit the deep integration of self-paced learning and adversarial training under the framework of AUC optimization, and provide a statistical upper bound to the AUC adversarial risk. Inspired by the statistical upper bound, we propose our optimization objective followed by an efficient alternatively stochastic descent algorithm, which can effectively improve the performance of learning models by guarding against adversarial samples and noisy samples. Experimental results on several standard datasets demonstrate that our DRAUC algorithm has better noise robustness and adversarial robustness than the state-of-the-art algorithms.

- Contribution: first AUC optimization algo robust against adversarial examples