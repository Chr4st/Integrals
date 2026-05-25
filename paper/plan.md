 NeurIPS 2025 Review: "Tree-Native Neural Integration: Graph Message Passing for Symbolic Antidifferentiation"

**Verdict: REJECT — no experimental results exist. Paper is unreviewable in current state.**

---

## Paper Summary

Proposes tree-native architecture for symbolic integration: GNN encoder (8-round bidirectional message passing + variable-aware attention) paired with top-down autoregressive tree decoder. Claims 12.1M parameters (~8x smaller than Lample & Charton 2020's 95M seq2seq). Trained on 1.5M backward-generated integrand-antiderivative pairs across 5 task types. Uses grammar-constrained decoding + sample-and-verify (N=25) exploiting differentiation as verification oracle. Claims first neural model for multivariate symbolic integration.

---

## CRITICAL Issues (2)

### C1. Zero Experimental Results

Every results table (Tables 2, 3) and Figure 1 contain placeholder dashes. The abstract claims "strong solve rates" and "roughly an order of magnitude fewer parameters than previously reported" — backed by nothing. Ablation text uses qualitative hedges ("improves steadily," "reduces substantially") with no data. The NeurIPS checklist is entirely unfilled (`\answerTODO{}` on all 16 items) — an independent desk-reject trigger.

**This is fatal.** No methodology assessment compensates for absent evidence.

### C2. Core Comparative Claim Is Unfalsifiable

The only comparison target is Lample & Charton (2020), but the paper confounds five variables simultaneously:

| Factor | This paper | Lample & Charton |
|--------|-----------|-----------------|
| Training pairs | 1.5M | 40M |
| Parameters | 12.1M (disputed) | 95M |
| Split strategy | Skeleton (harder) | Random (easier) |
| Task types | 5 | 1 |
| Evaluation set | Private | Private |

The paper acknowledges "numbers are not directly comparable" then places them in the same table. A 95M seq2seq baseline was designed (phase-08 docs) but never trained on the same data. Without that head-to-head, the tree-native efficiency thesis has zero evidence.

---

## HIGH Issues (5)

### H1. Parameter Count Contradictions

| Source | Total | Encoder | Decoder |
|--------|-------|---------|---------|
| Paper (abstract, §3) | 12.1M | 5.6M | 6.5M |
| Phase docs + README | ~9M | ~4.5M | ~4.2M |

The message-passing module alone: paper claims 5.26M, phase-10b computes 8 × 525K = 4.2M. A ~3M discrepancy (~25% of claimed model size) undermines the parameter-efficiency narrative. Whether the ratio to 95M is 7.8x or 10.6x changes the validity of the "order of magnitude" framing.

### H2. Sample Independence Assumption Invalid

The $1-(1-p)^N$ formula (§3.5) requires i.i.d. samples. Temperature sampling at $T=0.7$ with $\text{top-}p=0.95$ from the same model produces correlated candidates — they share identical encoder representations and learned failure modes. $T=0.7$ *sharpens* the distribution (below $T=1.0$), increasing correlation. The paper's own checklist acknowledges this requires "sufficiently high temperature" — then uses a temperature that isn't.

The claimed 99.97% pipeline success rate (for hypothetical $p=0.3$) is a misleading theoretical upper bound. No empirical measurement of sample diversity, unique candidate rate, or effective $N$ is provided.

### H3. "Perfect Verification Oracle" Is Neither Perfect Nor an Oracle

The paper uses "perfect verification oracle" and "mathematical certainty" (§3.5, line 362) to describe a system with a probabilistic numerical fallback:
- Accepts if 18/20 points match within $10^{-6}$ relative tolerance
- Points are hardcoded in the Rust verifier (not random as described)
- Tolerance differs across implementations: $10^{-6}$ (paper), $10^{-8}$ (phase-13a), $10^{-10}$ (phase-05)
- No false-positive/false-negative rate analysis
- Multivariate verification has no Rust implementation — must use SymPy with 4-second timeout (the exact CAS-timeout problem the paper criticizes in §1)

### H4. Multivariate Novelty Claim Overbroad

"First neural model for multivariate symbolic integration" (§1, line 105) is unsupported:
- Results table (Table 3) is empty
- Rubab et al. (2025) claim "neural antiderivatives for multivariate functions" — significant overlap
- The multivariate task ($\int f(x,y)\,dx$) is structurally identical to univariate integration with inert parameters — not a fundamentally new capability
- Max depth for multivariate data is 8 vs 10 for univariate (phase-04b), making multivariate systematically easier — cross-task comparisons would be misleading

### H5. Training Loss Contradicts Task Definition (Multivariate)

For $\int f(x,y)\,dx = F(x,y) + g(y)$, infinitely many valid antiderivatives exist. Verification correctly checks $\partial \hat{F}/\partial x = f$. But training uses cross-entropy loss against ONE specific $F$ — predicting $F + \sin(y)$ (mathematically correct) is penalized. The model learns to reproduce the backward generator's output form, not to perform partial antidifferentiation. No discussion of this loss-vs-equivalence-class tension exists.

---

## MEDIUM Issues (8)

### m1. No Complexity Analysis

Zero formal complexity comparison between tree GNN and seq2seq transformer. Back-of-envelope: GNN encoder O(8 × N × d²) ≈ 268M FLOPs for N=512; transformer encoder O(10 × N² × d) ≈ 1.68B FLOPs. The ~6x advantage is the paper's best argument and it's never computed. Conversely, scatter operations are memory-bound and poorly GPU-optimized vs. dense matmuls — parameter count alone is a bad efficiency proxy.

### m2. No Graph Batching → Training Time Implausible

The code processes single trees (no PyG `Batch` object, no batch index tensor). At batch_size=1: 1.2M examples × 40 epochs = 48M forward passes. At ~5ms each on A100 ≈ 67 hours — far exceeding the claimed 11-18 hours. Either undocumented batching exists, or the training time claim is wrong.

### m3. Config/Paper Hyperparameter Mismatches

`default.toml`: epochs=90, patience=15, warmup=5. Paper: epochs=60, patience=10, warmup=3. Curriculum phase boundaries differ between code and paper. These suggest the implementation and paper description diverged without reconciliation.

### m4. Decoder Depth Cap (8) vs Input Depth (10)

Integration frequently increases expression complexity. Max output depth 8 means some correct antiderivatives are structurally unrepresentable. No analysis of what fraction of test cases are affected.

### m5. Over-Smoothing Unaddressed

8 rounds of message passing on trees of depth ≤ 8 means every node incorporates information from every other node. Over-smoothing (Li et al., AAAI 2018; Rusch et al., ICML 2023) causes node embeddings to converge. The ablation showing diminishing returns beyond 8 rounds is consistent with this, but the paper never diagnoses the cause.

### m6. Mean Pooling Destroys Operand Order

Child→parent aggregation uses `scatter_mean` (phase-10b). Mean pooling is order-invariant — $f(x, y)$ and $f(y, x)$ produce identical parent messages. For non-commutative operators (−, ÷, ^), child order is critical. If positional encoding resolves this, it isn't documented.

### m7. No Error Bars or Multi-Seed Runs

NeurIPS requires confidence intervals. The paper promises "error bars to be included with final results" — a commitment, not compliance. Single-run results are insufficient.

### m8. Ablations on 10% Subset

Ablations use 150K pairs for 20 epochs (10% data, 33% epochs). Curriculum-sensitive conclusions may not transfer to full-scale training. The ablation code contains `raise NotImplementedError` stubs — ablations haven't actually been run.

---

## Missing References (Priority-Ordered)

### Must cite before submission

| Paper | Why |
|-------|-----|
| **Welleck et al. (AAAI 2022)** — "Symbolic Brittleness in Sequence Models" | Direct predecessor of skeleton-split evaluation. Shows seq2seq models fail on structural generalization via automated failure discovery. Not citing this is a serious omission. |
| **Cobbe et al. (2021)** — "Training Verifiers to Solve Math Word Problems" | Sample-and-verify is best-of-N rejection sampling — a well-studied paradigm. Paper presents it as novel without citing the lineage. |
| **Poesia et al. (NeurIPS 2022)** — Synchromesh; **Park et al. (NeurIPS 2024)** — Grammar-Aligned Decoding | Grammar-constrained decoding via arity stack is standard pushdown automaton technique, not a contribution. |
| **Barket et al. (NeurIPS MATH-AI 2024)** — Neural integration routine selection | Most directly adjacent concurrent work on neural-guided symbolic integration. |

### Strongly recommended



> [!IMPORTANT]
| Paper | Why |
|-------|-----|
| **SNIP (ICLR 2024 Spotlight)** | Contrastive pre-training between symbolic/numeric math representations — directly relevant to GNN encoder design |
| **Shojaee et al. (ICML 2024)** — TPSR | MCTS + transformer for symbolic math; planning-augmented decoding as alternative to sample-and-verify |
| **GraphDSR (Neural Networks 2025)** | DAG-based (not tree) GNN for symbolic expression generation — competing architecture |
| **Noorbakhsh et al. (2022)** | Integration via pre-trained LMs with 1.5 OOM less data — prior art on data efficiency |
| **Kamienny et al. (NeurIPS 2022)** | Same Charton group; tree-output symbolic regression with transformers |
| GNN depth literature: **Alon & Yahav (ICLR 2021)** on over-squashing; **Topping et al. (ICML 2022)** | Needed to explain why 8 rounds is optimal |

### Glaring baseline omissions

- **No CAS baseline** (SymPy/Mathematica). Given backward construction, CAS likely solves ~100% of the test set. Without this baseline, the paper's utility claim is hollow.
- **No LLM baseline** (GPT-4, Claude). For a 2025 submission, omitting frontier LLMs on symbolic integration is indefensible.

---

## Strengths Worth Preserving

1. **Skeleton-based data splitting** — genuinely more rigorous than random splitting. Eliminates structural leakage. Methodological contribution if properly credited to Welleck et al.'s intellectual lineage.
2. **Backward construction + verification oracle** — architecturally sound pipeline. The insight that differentiation is cheap/reliable while integration is hard maps naturally to sample-and-verify.
3. **System design scope** — 5 task types, grammar constraints, curriculum learning, Rust data pipeline. Ambitious and well-engineered infrastructure.
4. **Tree-native intuition** — encoding expressions as trees rather than flattened sequences is mathematically motivated. The architecture design is thoughtful even if unvalidated.

---

## Required Fixes Before Resubmission

| Priority | Action |
|----------|--------|
| **P0** | Run training. Populate all tables and figures with real numbers. |
| **P0** | Train 95M seq2seq baseline on same skeleton-split data for fair comparison. Add parameter-matched (~12M) seq2seq variant. |
| **P0** | Fill NeurIPS checklist (all 16 items currently `\answerTODO{}`). |
| **P1** | Reconcile parameter counts — run `model.count_parameters()` and report actual number. |
| **P1** | Report per-sample accuracy $p$, unique candidate rate, and empirical vs. theoretical success rate to validate independence assumption. |
| **P1** | Add CAS baseline (SymPy on test set) and LLM baseline (GPT-4/Claude). |
| **P1** | Run ablations (currently `NotImplementedError` stubs). Report 3+ seeds with error bars. |
| **P2** | Add complexity analysis table: GNN O(R·N·d²) vs Transformer O(L·N²·d). |
| **P2** | Address multivariate loss-vs-equivalence-class conflict explicitly. |
| **P2** | Downgrade "perfect oracle" to "high-confidence verification." Characterize false-positive rate. |
| **P2** | Add missing references (Welleck 2022, Cobbe 2021, Poesia 2022, Barket 2024 minimum). |
| **P3** | Complete broader impacts section (currently truncated mid-sentence). |
| **P3** | Reconcile hyperparameters between paper and config files. |

---

## Bottom Line

Strong methods proposal with zero validation. Architecture is well-motivated, data pipeline is rigorous, and skeleton splitting is a genuine methodological contribution. But the paper makes quantitative claims without quantitative evidence, overstates verification reliability, presents standard techniques (grammar decoding, best-of-N sampling) as contributions, and omits critical baselines (CAS, LLMs, fair head-to-head). In current form: a detailed blueprint, not a submittable paper.
