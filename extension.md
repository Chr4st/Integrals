# Papers on Solving Symbolic Integration

A comprehensive survey of research papers that attempt to solve symbolic integration using computational methods, organized by approach.

---

## I. Classical Algorithmic Approaches

| # | Title | Authors | Venue | Summary |
|---|-------|---------|-------|---------|
| 1 | A Heuristic Program that Solves Symbolic Integration Problems in Freshman Calculus (SAINT) | James R. Slagle | MIT PhD (1961); JACM (1963) | First automated integration system using LISP and heuristic pattern-matching on a decision tree. |
| 2 | Symbolic Integration (SIN) | Joel Moses | MIT Tech Report MAC-TR-47 (1967) | Redesigned SAINT with Hermite reduction and stronger heuristics; integrated into MACSYMA. |
| 3 | Symbolic Integration: The Stormy Decade | Joel Moses | Communications of the ACM, 14(8), 1971 | Retrospective on the first decade of symbolic integration research. |
| 4 | The Problem of Integration in Finite Terms | Robert H. Risch | Trans. Amer. Math. Soc., 1969 | Defines the Risch algorithm, the first complete decision procedure for elementary antiderivatives. |
| 5 | Integration of Elementary Functions | James H. Davenport; Manuel Bronstein | EUROSAM '79; Axiom impl. (~1990) | Extends Risch's algorithm to purely algebraic functions with near-complete general implementation. |
| 6 | Symbolic Integration I: Transcendental Functions | Manuel Bronstein | Book, Springer (1997, 2nd ed. 2005) | Definitive textbook on the Risch algorithm for transcendental functions. |
| 7 | The Lazy Hermite Reduction | Manuel Bronstein | INRIA Tech Report 3562 (1998) | Hermite reduction for algebraic integrands without computing an integral basis. |
| 8 | Rule-based integration (RUBI) | Albert Rich, Patrick Scheibe, Nasser Abbasi | JOSS, 3(32), 2018 | Over 6,700 transformation rules organized as a decision tree; outperforms Mathematica and Maple on large test suites. |
| 9 | Generalization of Risch's Algorithm to Special Functions | Clemens G. Raab | CASC 2013; arXiv:1305.1481 | Extends the Risch framework to handle hypergeometric and Bessel functions. |

---

## II. Hybrid Symbolic-Numeric Approaches

| # | Title | Authors | Venue | Summary |
|---|-------|---------|-------|---------|
| 10 | Hybrid Symbolic-Numeric Integration in MAPLE | K.O. Geddes, G.J. Fee | ISSAC '92, ACM Press | Early hybrid system combining symbolic transformation with numerical methods. |
| 11 | Symbolic-Numeric Integration of Rational Functions | R.H.C. Moir, R.M. Corless, M. Moreno Maza, N. Xie | Numerical Algorithms, 2019; arXiv:1712.01752 | Hermite reduction for the rational part, multiprecision rootfinding for the transcendental part. |
| 12 | Symbolic-Numeric Integration based on Sparse Regression | S. Iravanian, S. Gowda, C. Rackauckas | ACM Comm. in Computer Algebra, 2022; arXiv:2201.12468 | SINDy-style sparse regression on a symbolic ansatz; implemented in Julia's SymbolicNumericIntegration.jl. |
| 13 | Hybrid Symbolic-Numeric and Numerically-Assisted Symbolic Integration | S. Iravanian, S. Gowda, C. Rackauckas | ISSAC '24, Raleigh, NC | Extends sparse-regression hybrid with complex-field numerical filtering. |
| 14 | Stability Problems in Symbolic Integration | Shaoshi Chen et al. | ISSAC '22; arXiv:2202.06305 | Studies stability in differential fields to improve numerical robustness of CAS output. |

---

## III. Neural Seq2Seq and Transformer Direct Integration

| # | Title | Authors | Venue | Summary |
|---|-------|---------|-------|---------|
| 15 | Deep Learning for Symbolic Mathematics | Guillaume Lample, Francois Charton | ICLR 2020; arXiv:1912.01412 | Seminal paper: seq2seq transformer on synthetic data outperforms Mathematica on their test set. |
| 16 | The Use of Deep Learning for Symbolic Integration: A Review | Ronan Dor, Uri Leron | arXiv:1912.05752 | Critical review arguing Lample & Charton's test set has built-in biases from the training distribution. |
| 17 | Pretrained Language Models are Symbolic Mathematics Solvers too! | K. Noorbakhsh, M. Sulaiman, M. Sharifi, K. Roy, P. Jamshidi | arXiv:2110.03501 | Pretraining on language translation then fine-tuning on integration matches Lample & Charton with ~1.5 orders of magnitude less training data. |
| 18 | Symbolic Brittleness in Sequence Models | Sean Welleck, Peter West, Jize Cao, Yejin Choi | AAAI 2022; arXiv:2109.13986 | Shows seq2seq integration models fail on robustness, compositional generalization, and OOD problems. |
| 19 | Mastering Symbolic Operations: Augmenting LMs with Compiled Neural Networks | Yixuan Weng et al. | ICLR 2024; arXiv:2304.01665 | Compiled neural networks with artificial attention weights encoding symbolic rules for exact integration. |

---

## IV. Step-by-Step, Rule Prediction, and Hybrid Search

| # | Title | Authors | Venue | Summary |
|---|-------|---------|-------|---------|
| 20 | SIRD: Symbolic Integration Rules Dataset | V. Sharma, A. Nagpal, M.F. Balin | NeurIPS MATH-AI Workshop, 2023 | 2M (function, rule) pairs for 24 rules; transformer predicts next rule, 6x search reduction, 2.28x speedup. |
| 21 | AlphaIntegrator: Transformer Action Search for Symbolic Integration Proofs | M. Unsal, T. Gehr, M. Vechev | arXiv:2410.02666, 2024 | First correct-by-construction learning system: GPT-style transformer guides search over axiomatically correct CAS actions. |

---

## V. Algorithm Selection and Routing within CAS

| # | Title | Authors | Venue | Summary |
|---|-------|---------|-------|---------|
| 22 | Symbolic Integration Algorithm Selection with ML: LSTMs vs Tree LSTMs | R. Barket, M. England, J. Gerhard | ICMS 2024; arXiv:2404.14973 | TreeLSTM selects among Maple's 12 integration sub-algorithms; 84.6% optimal vs 60.5% for Maple's meta-algorithm. |
| 23 | Transformers to Predict the Applicability of Symbolic Integration Routines | R. Barket, U. Shafiq, M. England, J. Gerhard | NeurIPS MATH-AI Workshop, 2024; arXiv:2410.23948 | Transformers predict which integration sub-routine will succeed before running it. |
| 24 | Tree-Based Deep Learning for Ranking Symbolic Integration Algorithms | R. Barket, M. England, J. Gerhard | arXiv:2508.06383, 2025 | Two-stage pipeline: classify applicable methods then rank by predicted output complexity; ~90% accuracy. |

---

## VI. Data Generation for ML Training

| # | Title | Authors | Venue | Summary |
|---|-------|---------|-------|---------|
| 25 | Generating Elementary Integrable Expressions | R. Barket, M. England, J. Gerhard | CASC 2023; arXiv:2306.15572 | Uses Risch algorithm to construct a large-scale dataset avoiding known biases in prior ML training sets. |
| 26 | The Liouville Generator for Producing Integrable Expressions | R. Barket, M. England, J. Gerhard | CASC 2024; arXiv:2406.11631 | Data generator grounded in Liouville's theorem producing more realistic integrands for benchmarking and ML. |

---

## VII. Neural Antiderivative Approximation

| # | Title | Authors | Venue | Summary |
|---|-------|---------|-------|---------|
| 27 | Computing Anti-Derivatives using Deep Neural Networks | D. Chakraborty, S. Gopalakrishnan | arXiv:2209.09084, 2022 | DNN architecture producing closed-form antiderivatives of non-elementary and oscillatory functions. |
| 28 | Anti-derivatives approximator for enhancing PINNs (ADA-F) | -- | Computer Methods in Applied Mechanics and Engineering, 2024 | Fourier-series-based antiderivative approximator as adaptive activation function inside PINNs. |
| 29 | Learning Neural Antiderivatives | F. Rubab, N.E. Nsampi, M. Balint et al. | arXiv:2509.17755, 2025 | Neural representations of repeated antiderivatives across dimensionalities and integration orders. |

---

## VIII. Reinforcement Learning and Autonomous Search

| # | Title | Authors | Venue | Summary |
|---|-------|---------|-------|---------|
| 30 | Symbolic Equation Solving via Reinforcement Learning | Lennart Dabelow, Masahito Ueda | Neurocomputing 613, 2024; arXiv:2401.13447 | RL agent operates a symbolic stack calculator; correct-by-construction solutions with applicability to integration. |
| 31 | Deep Symbolic Optimization: RL for Symbolic Mathematics | C.F. Hayes, F.L. Da Silva, M. Landajuela, B.K. Petersen et al. | arXiv:2505.10762, 2025 | Frames symbolic discovery as sequential decision-making; neural network learns distribution over expression trees via RL. |

---

## IX. LLM Capability Studies and Reviews

| # | Title | Authors | Venue | Summary |
|---|-------|---------|-------|---------|
| 32 | A Neural Network Solves, Explains, and Generates University Math Problems | I. Drori, S. Zhang et al. | PNAS 2022; arXiv:2112.15594 | Codex (GPT-3) via few-shot + program synthesis solves MIT calculus including integration at 81% accuracy. |
| 33 | Investigating Symbolic Capabilities of Large Language Models | -- | arXiv:2405.13209, 2024 | Benchmarks eight LLMs on symbolic tasks including integration; evaluates based on Chomsky's Hierarchy. |
| 34 | Advancing Symbolic Integration in LLMs: Beyond Conventional Neurosymbolic AI | M. Rani, B.K. Mishra, D. Thakker | arXiv:2510.21425, 2025 | Survey and taxonomy of approaches for integrating symbolic methods into LLMs for mathematical tasks. |

---

## X. CAS+ML Libraries

| # | Title | Authors | Venue | Summary |
|---|-------|---------|-------|---------|
| 35 | CALT: A Library for Computer Algebra with Transformer | Kento Kato et al. | ISSAC 2025; arXiv:2506.08600 | Python/SageMath library enabling non-experts to train transformers for symbolic computation including integration. |

---

## XI. Feynman Integral Reduction (High-Energy Physics)

These papers address computing Feynman loop integrals, a specialized form of multi-dimensional symbolic/numeric integration central to particle physics.

| # | Title | Authors | Venue | Summary |
|---|-------|---------|-------|---------|
| 36 | Targeting Multi-Loop Integrals with Neural Networks | R. Winterhalder et al. | SciPost Physics 12, 2022; arXiv:2112.09145 | Normalizing flows optimize integration contour in the complex plane for multi-loop Feynman integrals. |
| 37 | Machine Learning Post-Minkowskian Integrals | Ryusuke Jinno et al. | JHEP 07 (2023); arXiv:2209.01091 | Neural importance sampling (i-flow) accelerates Monte Carlo evaluation of multi-loop integrals. |
| 38 | Learning Feynman Integrals from DEs with Neural Networks | F. Calisto, R. Moodie, S. Zoia | JHEP 07 (2024); arXiv:2312.02067 | Physics-informed deep learning to approximate Feynman integral solutions to their DEs; ~1% accuracy at two loops. |
| 39 | Refining IBP Reduction of Feynman Integrals with ML | M. von Hippel, M. Wilhelm | JHEP 05 (2025); arXiv:2502.05121 | FunSearch (LLM + genetic programming) discovers and improves IBP heuristics for Feynman integral reduction. |
| 40 | Explainable AI-assisted Optimization for Feynman Integral Reduction | -- | arXiv:2502.09544, 2025 | Priority function for IBP reduction using FunSearch; up to 3058x reduction in seeding integrals. |
| 41 | RL and Metaheuristics for Feynman Integral Reduction | -- | Phys. Rev. D, 2025; arXiv:2504.16045 | RL agent learns optimal IBP actions; small neural network compresses parameters for simulated annealing. |
| 42 | Learning to Unscramble Feynman Loop Integrals with SAILIR | -- | arXiv:2604.05034, 2025 | Self-supervised transformer classifier guides IBP reduction using 40% less memory than Kira. |
| 43 | Uncovering Singularities in Feynman Integrals via ML | Y. Liu, Y. Xu, Y. Zhang | arXiv:2510.10099, 2025 | Symbolic regression extracts singularity structure of multi-loop Feynman integrals without prior knowledge. |

---

**Total: 43 papers** spanning 1961 to 2025, covering classical algorithms, hybrid symbolic-numeric methods, neural seq2seq, step-by-step search, algorithm selection, reinforcement learning, LLM studies, and Feynman integral computation.
