# Novel Research Directions: Symbolic Integration via Neural Methods

## Executive Summary

We surveyed the full landscape of neural symbolic integration research (43 existing papers, see `extension.md`) and identified **10 open research directions** where no prior work exists or the gap is substantial. Of these, we propose **7 concrete paper ideas** ranked by novelty and feasibility. The three strongest directions are: (1) symbolic contour integration via residue prediction, (2) closed-form definite integral prediction, and (3) neural antiderivatives with formal Lean proof certificates. All three have clean task definitions, synthetic data generation pipelines, and perfect or near-perfect verification oracles.

---

## Gap Analysis

| Area | Existing Work | Gap | Novelty |
|------|--------------|-----|---------|
| Complex contour integrals (symbolic) | Numerical only (Feynman physics) | Total gap in symbolic setting | Very high |
| Definite integrals (closed-form) | One pre-transformer 2019 paper | Direct Lample-Charton extension untouched | High |
| Multivariate antiderivatives | Numerical only (JHEP 2023) | Symbolic task completely absent | High |
| Differential forms / exterior calculus | Numerical FEEC, type-system SR | Symbolic manipulation of forms absent | Very high |
| Special function integration | Gap explicitly noted in Oct 2025 survey | Non-elementary output prediction absent | High |
| Oscillatory integrals (symbolic asymptotics) | Numerical methods active 2024-2026 | Symbolic asymptotic prediction absent | Medium-high |
| Tree PE for symbolic math | Active for OCR/code, not math reasoning | Ablation on integration tasks missing | Medium |
| Neural integration + Lean/Coq proofs | LLM theorem proving booming, calculus absent | Zero work connecting these fields | Very high |
| Integration on manifolds | Nothing in ML | Completely blank | Extreme |

---

## Proposed Papers

### Paper 1: Symbolic Contour Integration via Transformer Residue Prediction

**Title idea:** "Complex Analysis as Sequence Translation: Transformer-Based Symbolic Contour Integration via Residue Prediction"

**Task definition.** Given a meromorphic function f(z) and a contour specification (e.g., "circle |z|=R", "upper half-plane semicircle", "keyhole around branch cut"), output the symbolic value of the contour integral using the residue theorem.

**Why it's novel.** Every existing ML integration paper works on real-valued indefinite integrals. The Feynman integral literature uses NNs to *numerically deform contours*, not to symbolically compute residue sums. No dataset, no model, no benchmark exists for symbolic complex integration.

**Data generation.**
- Sample rational and meromorphic functions with known pole structure: f(z) = p(z)/q(z) where q has roots at known locations.
- Compute residues at each pole symbolically via CAS (Laurent expansion or limit formula).
- For each contour type, the answer is 2*pi*i * sum of enclosed residues.
- Extend to branch cuts: log, fractional powers. The answer involves discontinuity integrals.
- Generate 500K-1M training pairs across difficulty tiers (simple poles, higher-order poles, essential singularities, branch cuts).

**Verification oracle.** (a) CAS residue computation cross-check, (b) high-precision numerical quadrature along the contour, (c) for rational functions, partial fraction decomposition gives exact answers. Oracle quality: near-perfect (numerical verification to 10^-12 precision).

**Architecture.** Extend the Lample-Charton prefix encoding to include contour tokens: `[CONTOUR circle R 3]` or `[CONTOUR halfplane upper]`. The model must learn to: identify poles, determine which are enclosed, compute residues, sum. Feature extraction includes pole-location features.

**Expected contribution level:** Top venue (ICML/NeurIPS/ICLR). First paper defining and solving this task.

**Estimated difficulty:** Medium. Data generation is tractable. The main challenge is handling branch cuts and essential singularities.

**Key references to position against:** Lample & Charton (ICLR 2020), AlphaIntegrator (2024), Winterhalder et al. (SciPost 2022, numerical contour deformation)

---

### Paper 2: Closed-Form Definite Integral Prediction

**Title idea:** "From Indefinite to Definite: Transformer-Based Prediction of Closed-Form Definite Integrals"

**Task definition.** Given an integrand f(x) and bounds [a, b] (including improper integrals with a or b = infinity), predict the closed-form symbolic value. Examples: integral_0^inf e^{-x^2} dx = sqrt(pi)/2, integral_0^pi sin(x)/x dx = Si(pi).

**Why it's novel.** Lample & Charton (2020) and all follow-on work exclusively handle indefinite integration. No paper in the neural symbolic math literature trains a model to output closed-form definite integral values. The only prior work is a 2019 pre-transformer paper using functional link ANNs (numerical, not symbolic).

**Data generation.**
- **Route 1 (FTC-based):** Generate (f, F) pairs via backward generation, pick symbolic bounds (0, 1, pi, inf, -inf), compute F(b) - F(a) symbolically. This covers cases where an elementary antiderivative exists.
- **Route 2 (Table-based):** Scrape Gradshteyn-Ryzhik (GR) tables, DLMF identities, and Wolfram's definite integral database. These contain thousands of non-trivial definite integral identities.
- **Route 3 (CAS-generated):** Use SymPy/Mathematica to evaluate definite integrals symbolically on randomly generated integrands; filter for clean closed-form results.
- Target: 200K-500K training pairs.

**Verification oracle.** (a) FTC: if F exists, check F(b) - F(a), (b) high-precision numerical quadrature (mpmath, 50+ digits) compared against the symbolic answer evaluated numerically, (c) CAS cross-validation. Oracle quality: near-perfect via numerical agreement.

**New challenges vs. indefinite case.**
- Convergence reasoning: the model must learn when integral_0^inf f(x) dx converges.
- Boundary evaluation: F(inf) requires limit computation, which may involve L'Hopital or asymptotic analysis.
- Special constants: outputs include pi, e, log(2), Catalan's constant, gamma (Euler-Mascheroni), Gamma function values. The output vocabulary needs these constants.

**Expected contribution level:** Top venue. Direct and natural extension of the most-cited paper in the field.

**Estimated difficulty:** Medium. Route 1 is straightforward. Route 2 (GR tables) requires some data engineering. The hardest part is improper integrals where the antiderivative is not elementary.

**Key references:** Lample & Charton (ICLR 2020), DLMF (NIST), Gradshteyn-Ryzhik tables

---

### Paper 3: Neural Antiderivatives with Lean 4 Proof Certificates

**Title idea:** "Formally Verified Symbolic Integration: Neural Antiderivative Prediction with Lean Proof Certificates"

**Task definition.** Given an integrand f(x), the system: (1) predicts an antiderivative F(x) using a transformer, (2) generates a Lean 4 tactic proof that `deriv F x = f x`, verified by the Lean type-checker. Every returned result comes with a machine-checkable certificate of correctness.

**Why it's novel.** Neural theorem proving (AlphaProof, Leanstral, miniF2F) and neural integration (Lample & Charton) are both active fields, but they have never intersected. No paper connects symbolic antiderivative computation to formal proofs. This produces correctness guarantees strictly stronger than differentiation-and-simplify (which can have false negatives when simplification fails).

**Architecture.**
- Stage 1: Standard Lample-Charton style transformer predicts F(x) as a symbolic expression.
- Stage 2: A proof-generation module (could be a separate LLM fine-tuned on Mathlib proofs, or a template-based tactic generator) produces a Lean proof of `HasDerivAt F f x` or `deriv F x = f x`.
- Verification: Lean type-checker. This is a perfect oracle: decidable, no false positives, no false negatives.

**Data generation.**
- Mine Mathlib's `Analysis.SpecialFunctions` and `MeasureTheory.Integral` for existing proved integration lemmas as seed data.
- Generate (f, F, proof) triples: for each backward-generated (f, F) pair, construct the Lean proof that `deriv F = f`. For elementary functions, these proofs follow mechanical patterns using `simp`, `ring`, `deriv_rules`.
- Bootstrap: start with simple proofs (polynomial derivatives), progressively add trig, exp, log, compositions via chain rule.

**Key insight.** The differentiation proof for elementary functions is highly structured and can be generated programmatically from the expression tree. Each node in the AST corresponds to a differentiation rule (chain rule, product rule, etc.) that maps to a Mathlib lemma. The proof generation is essentially a recursive walk of the expression tree, emitting the appropriate tactic at each step.

**Expected contribution level:** Top venue. Bridges two literatures in a novel way with a clean, well-defined contribution.

**Estimated difficulty:** High. Lean/Mathlib infrastructure for differentiation is mature but requires expertise. The proof generation module is the main engineering challenge.

**Key references:** Lample & Charton (ICLR 2020), AlphaProof (DeepMind 2024), Mathlib, Polu & Sutskever (2020)

---

### Paper 4: Non-Elementary Antiderivatives via Special Function Prediction

**Title idea:** "Beyond Elementary Functions: Transformer-Based Integration with Special Function Output Spaces"

**Task definition.** Given an integrand f(x) whose antiderivative is not expressible in elementary functions, predict the antiderivative in terms of special functions: erf, Ei, Si, Ci, Li, Bessel functions, elliptic integrals, hypergeometric functions, polylogarithms, etc.

**Why it's novel.** Lample & Charton and all follow-on work train exclusively on elementary function pairs. The Oct 2025 survey (arXiv:2510.21425) explicitly identifies this as an open gap. No ML paper handles the transition from "no elementary antiderivative exists" to "the answer involves erf(x)."

**Data generation.**
- Use SymPy's Meijer G-function integration engine, which can express antiderivatives in terms of hypergeometric and special functions.
- Generate integrands from compositions involving exp(-x^2), 1/log(x), sin(x)/x, etc.
- CAS-compute antiderivatives; filter for those returning special functions.
- Additionally: differentiate known special function identities backward.
- Target: 300K pairs, roughly 50K per major special function family.

**New modeling challenges.**
- Output vocabulary expansion: need tokens for erf, Ei, Si, Ci, J_n, Y_n, K, E (elliptic), _2F_1, Li_s, etc.
- The model must learn to *recognize* when an elementary antiderivative does not exist and switch to special function output mode.
- Feature extraction: add features for oscillatory structure, exponential decay, algebraic singularities that signal which special function family is relevant.

**Verification oracle.** Differentiation of the special function antiderivative. SymPy can differentiate all standard special functions, so d/dx erf(x) = 2/sqrt(pi) * e^{-x^2} is checkable. Oracle quality: perfect.

**Expected contribution level:** Strong venue (ICML/NeurIPS). Addresses a known and explicitly stated gap.

**Estimated difficulty:** Medium. The main challenge is data generation diversity and ensuring the model learns the boundary between elementary and non-elementary cases.

**Key references:** Lample & Charton (ICLR 2020), Raab (CASC 2013, Risch generalization), arXiv:2510.21425 (survey identifying this gap)

---

### Paper 5: Multivariate Symbolic Antidifferentiation

**Title idea:** "Multivariate Symbolic Integration via Transformer with Variable-Aware Attention"

**Task definition.** Given a multivariate expression f(x, y) and a target variable (say x), predict the partial antiderivative F(x, y) such that dF/dx = f(x, y). Extend to iterated integrals: given f(x,y), compute integral integral f dx dy.

**Why it's novel.** All neural integration work is univariate. The only multivariate ML integration paper (Maitre & Santos-Mateos, JHEP 2023) is purely numerical. No symbolic multivariate antiderivative prediction exists.

**Data generation.**
- Generate bivariate expression trees F(x, y) via extended backward generation (two free symbols).
- Compute dF/dx to get the integrand.
- The "constants of integration" become arbitrary functions of y: the model must learn that integral of x*y dx = x^2*y/2 + g(y).
- For iterated integrals: compute d^2F/dxdy, train model to recover F.

**New modeling challenges.**
- Variable-aware attention: the model must distinguish which variable to integrate with respect to. Inject this via a `[VAR x]` token.
- Function-valued constants: the model must output g(y) terms. These can be represented as sub-expressions in the remaining variables.
- Commutativity of mixed partials: d^2F/dxdy = d^2F/dydx provides additional verification.

**Verification oracle.** Partial differentiation: d/dx of the predicted F must equal f. Oracle quality: perfect (same as univariate).

**Expected contribution level:** Strong venue. Natural and important generalization of the foundational work.

**Estimated difficulty:** Medium-high. Expression trees become larger. The function-valued constant of integration adds complexity.

---

### Paper 6: Symbolic Asymptotic Expansion for Oscillatory Integrals

**Title idea:** "Neural Stationary Phase: Transformer-Predicted Asymptotic Expansions for Oscillatory Integrals"

**Task definition.** Given a highly oscillatory integral of the form integral f(x) * e^{i*omega*g(x)} dx, predict the leading-order asymptotic expansion as omega -> infinity. The answer involves values at stationary points of g(x) and their local curvature.

**Why it's novel.** The numerical oscillatory quadrature literature is active (2024-2026) but purely numeric. No ML paper predicts symbolic asymptotic expansions. The method of stationary phase / steepest descent is a cornerstone of applied mathematics taught in every graduate program but has never been automated via neural methods.

**Data generation.**
- Generate pairs (f, g) where g has known stationary points.
- Compute the stationary phase expansion symbolically: find x_0 where g'(x_0) = 0, compute f(x_0), g''(x_0), assemble the asymptotic formula.
- Higher-order terms involve derivatives of f and g at stationary points.
- Generate 200K pairs across: single stationary point, multiple stationary points, coalescing stationary points, saddle points in the complex plane.

**Verification oracle.** Numerical verification: evaluate the integral numerically for large omega and compare against the predicted asymptotic value. Agreement to O(1/omega) confirms the leading term. Not a perfect oracle (approximate), but very strong.

**Expected contribution level:** Strong venue, especially if framed for the applied math / scientific computing community (SISC, JCP, or NeurIPS).

**Estimated difficulty:** High. Asymptotic analysis is harder to formalize than antidifferentiation. The approximate oracle is a weakness compared to the exact differentiation oracle in other proposals.

---

### Paper 7: Algebraically-Aware Tree Positional Encodings for Symbolic Math

**Title idea:** "Commutativity-Aware Tree Positional Encodings for Neural Symbolic Mathematics"

**Task definition.** Design and evaluate tree positional encodings that encode algebraic properties of expression trees (commutativity of +/*, associativity, distributivity) rather than treating all binary operators as generic nodes. Evaluate on integration, simplification, and equation solving benchmarks.

**Why it's novel.** Existing tree PE work (Shiv & Quirk NeurIPS 2019, EMNLP 2022) targets code ASTs and handwriting recognition, not symbolic math reasoning. No ablation study compares PE strategies on integration tasks. No PE encodes that the children of Add/Mul nodes are order-invariant.

**Proposed encodings.**
- **Algebraic PE:** For commutative operators, canonicalize child ordering (e.g., alphabetical) and encode both children at the same depth level. For non-commutative operators (Pow, Div), encode left/right asymmetrically.
- **Depth-breadth PE:** Encode (depth, breadth-position) as a 2D sinusoidal embedding, where breadth-position is computed after canonical ordering.
- **Path PE:** Encode the root-to-node path as a sequence of (operator, child-index) pairs, with canonical ordering for commutative nodes.

**Evaluation.** Ablation on Lample-Charton's integration benchmark: compare flat sequence PE, vanilla tree PE, and algebraically-aware tree PE. Measure token accuracy, exact match, and solve rate. Test generalization to out-of-distribution expression depths.

**Expected contribution level:** Workshop or mid-tier venue (EMNLP, ACL). Incremental but practically useful.

**Estimated difficulty:** Low-medium. Primarily an engineering and ablation study.

---

## Implications

**For the Integrals project specifically:** Papers 2 (definite integrals) and 4 (special functions) are the most natural extensions of the current codebase. The existing tokenizer, feature extractor, and training pipeline can be adapted with moderate effort. Paper 7 (tree PE) could be integrated into the existing model as an architectural improvement.

**For the field broadly:** Papers 1 (contour integrals) and 3 (Lean proofs) open genuinely new research directions rather than incremental improvements. They would attract attention from the complex analysis and formal methods communities respectively, broadening the audience beyond the neural symbolic math niche.

**Publication strategy:** Papers 1-3 are top-venue material (ICML/NeurIPS/ICLR). Paper 4 is strong but incremental. Papers 5-6 are solid contributions. Paper 7 is a workshop/short paper.

---

## Risks and Caveats

- **Data generation feasibility:** Papers 1 and 6 require generating training data for tasks (contour integrals, asymptotic expansions) where CAS support is less mature than for indefinite integration. SymPy's residue computation is good but not comprehensive for essential singularities.
- **Oracle quality:** Papers 2, 6, and partially 1 rely on numerical verification rather than exact symbolic verification. This weakens the "perfect oracle" narrative that makes indefinite integration special.
- **Lean infrastructure (Paper 3):** Mathlib's differentiation library is mature but interfacing with it programmatically for proof generation requires significant Lean expertise.
- **Scope creep:** Papers 4 (special functions) and 5 (multivariate) each could be scoped too broadly. Recommend restricting to 2-3 special function families or bivariate-only for a first paper.
- **Reviewer skepticism:** The integration community may question whether these extensions face the same "biased test set" criticism that Dor & Leron (2019) raised against Lample & Charton. Mitigate by using held-out test sets from independent sources (GR tables, DLMF, Mathlib lemmas).

---

## Recommendation

**If pursuing one paper:** Start with **Paper 2 (Definite Integrals)**. It is the most direct, lowest-risk extension of Lample & Charton with clear data generation routes, a near-perfect verification oracle, and immediate practical value. The FTC-based data generation pipeline can reuse the existing backward generation infrastructure.

**If pursuing two papers in parallel:** Pair Paper 2 with **Paper 4 (Special Functions)**. Both extend the output space of the existing model and share infrastructure (tokenizer, training loop, verification). Together they cover "harder bounds" and "harder integrands," the two natural axes of difficulty.

**If pursuing a high-risk/high-reward direction:** Go with **Paper 1 (Contour Integrals)** or **Paper 3 (Lean Proofs)**. These open entirely new research threads rather than extending existing ones.

---

## Sources

- Lample & Charton, "Deep Learning for Symbolic Mathematics," ICLR 2020 (arXiv:1912.01412)
- Unsal et al., "AlphaIntegrator," arXiv:2410.02666 (2024)
- Barket et al., "Transformers to Predict Applicability of Integration Routines," NeurIPS MATH-AI 2024 (arXiv:2410.23948)
- Rani et al., "Advancing Symbolic Integration in LLMs," arXiv:2510.21425 (2025) -- explicitly identifies special function gap
- Winterhalder et al., "Targeting Multi-Loop Integrals with NNs," SciPost Physics 12 (2022) (arXiv:2112.09145)
- Maitre & Santos-Mateos, "Multi-variable Integration with a Neural Network," JHEP 2023 (arXiv:2211.02834)
- Shiv & Quirk, "Novel Positional Encodings to Enable Tree-Based Transformers," NeurIPS 2019
- Dor & Leron, "Review of Lample and Charton," arXiv:1912.05752 (2019)
- Raab, "Generalization of Risch's Algorithm to Special Functions," CASC 2013 (arXiv:1305.1481)
- Iravanian et al., "Symbolic-Numeric Integration based on Sparse Regression," arXiv:2201.12468 (2022)
- Song, "Neuro-Symbolic Theorem Proving with Lean," tutorial (2024)
- Calisto et al., "Learning Feynman Integrals from DEs with NNs," JHEP 2024 (arXiv:2312.02067)
- Welleck et al., "Symbolic Brittleness in Sequence Models," AAAI 2022 (arXiv:2109.13986)
- Gradshteyn & Ryzhik, "Table of Integrals, Series, and Products," Academic Press
- NIST Digital Library of Mathematical Functions (DLMF), https://dlmf.nist.gov
