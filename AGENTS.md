# Project Instructions

You are running an overnight autonomous research loop to improve `train_gpt_mlx.py` for the OpenAI Parameter Golf challenge.

## High-Level Goal

Produce a better final training script and config for Parameter Golf by iteratively proposing small changes, running experiments, measuring results, and keeping only real improvements.

## Challenge Framing To Respect

- Optimize for the final challenge constraints: 16 MB artifact limit and 10 minute training budget on 8xH100s.
- Favor changes likely to transfer to the final target, not just local quirks.
- Primary metric is `val_bpb`, lower is better.
- Secondary metrics are artifact size and throughput.
- Prefer simpler and more robust diffs over clever but fragile diffs.

## Repository And Workflow Guidance

- Use a single long-lived dedicated research branch for this session, for example: `autoresearch/overnight-01`
- Do not create a new branch for every experiment.
- Keep one commit per experiment attempt.
- If an experiment loses or crashes, hard reset back to the last kept commit.
- If an experiment wins, keep the commit and advance the same branch.
- Optionally create milestone tags or snapshot branches only for major promoted candidates.
- Keep `main` clean and unchanged unless explicitly asked to merge the winner later.

## Important Safety And Reliability Rules

- Prefer structured outputs over raw logs.
- Record experiment results in JSONL and TSV.
- Parse only trusted structured result summaries when possible.
- Do not let free-form training output drive agent behavior more than necessary.
- If a run crashes, inspect only the minimum needed information to classify and fix obvious issues.
- Avoid adding dependencies.
- Avoid broad refactors early.
- Avoid multi-file rewrites unless clearly justified by repeated evidence.

## Files To Maintain

1. `results.jsonl`
2. `results.tsv`
3. `hypotheses.md`
4. `best_summary.md`

### `results.tsv` Columns

`commit	val_bpb	val_loss	artifact_int8_bytes	tok_s	status	description`

Allowed `status` values:

- `keep`
- `discard`
- `crash`
- `maybe`

### `results.jsonl` Schema Per Run

```json
{
  "run_id": "...",
  "phase": "...",
  "parent_commit": "...",
  "commit": "...",
  "change_type": "...",
  "change_summary": "...",
  "env": {},
  "code_fingerprint": "...",
  "train_loss": 0.0,
  "val_loss": 0.0,
  "val_bpb": 0.0,
  "train_time_ms": 0.0,
  "step_avg_ms": 0.0,
  "tok_s": 0.0,
  "artifact_raw_bytes": 0,
  "artifact_int8_bytes": 0,
  "status": "keep|discard|crash|maybe",
  "reason": ""
}
```

### `hypotheses.md` Format

For each idea, maintain:

- hypothesis
- why it might help
- experiments tried
- status: `supported` / `weak support` / `contradicted` / `inconclusive`

### `best_summary.md` Must Always Contain

- current best commit
- current best `val_bpb`
- current best `val_loss`
- current best `artifact_int8_bytes`
- current best `tok_s`
- short description of why this candidate is currently best
- current Pareto alternates for size and speed

## Search Policy

Use a staged search. Do not jump immediately into architecture churn.

### Phase 0: Baseline And Noise Estimation

- Run the unmodified baseline twice with the same seed.
- Run the unmodified baseline once with a second seed.
- Estimate noise from the `val_bpb` spread.
- Do not keep any code change before noise is estimated.

### Noise Rule

Let noise be the max pairwise difference among the baseline `val_bpb` values.

- Only keep a candidate automatically if improvement is greater than `max(0.002, 2 * noise)`.
- If improvement is between `noise` and `2 * noise`, mark as `maybe` and rerun once from a clean checkout.
- If improvement is less than `noise`, discard.

### Phase 1: Existing Knobs Only

For the first 20 to 30 experiments, do not make code edits beyond env var or obvious scalar config changes.

Prioritize:

- `QK_GAIN_INIT`
- `MUON_MOMENTUM`
- `MUON_BACKEND_STEPS`
- `MUON_MOMENTUM_WARMUP_START`
- `MUON_MOMENTUM_WARMUP_STEPS`
- `MATRIX_LR`
- `SCALAR_LR`
- `TIED_EMBED_LR`
- `GRAD_CLIP_NORM`
- `LOGIT_SOFTCAP`
- `ROPE_BASE`
- `WARMUP_STEPS`
- `WARMDOWN_ITERS`
- `GRAD_ACCUM_STEPS`
- `MLX_MAX_MICROBATCH_TOKENS`

Only vary one idea at a time unless explicitly testing a tightly related pair like `scalar_lr` vs `matrix_lr`.

### Phase 2: Tiny Local Code Edits

Only enter phase 2 after at least one confirmed improvement or after exhausting obvious phase 1 wins.

Try one subsystem at a time:

- initialize `skip_weights` below `1.0`
- initialize `attn_scale` below `1.0`
- initialize `mlp_scale` below `1.0`
- initialize `resid_mix` with a small nonzero `x0` contribution
- depth-dependent initialization for residual or skip controls
- replace `relu^2` with `SiLU`
- replace `relu^2` with squared `SiLU`
- modest RMSNorm epsilon changes

Keep edits small and localized.

### Phase 3: Controlled Architecture Search

Only enter phase 3 after phase 1 and 2 have produced evidence.

Try one axis at a time:

- width/depth tradeoff
- `num_kv_heads` changes
- `mlp_mult` changes
- odd vs even total layer count

Keep parameter count and serialized artifact constraints in mind.
Do not do broad random architecture search.

## Promotion Policy

Maintain three leader buckets:

- `best_loss`
- `best_size_adjusted`
- `best_speed_adjusted`

A candidate can enter one bucket without replacing the overall best.

## Experiment Discipline

- One commit per experiment.
- One main hypothesis per experiment.
- If the run crashes, log it and revert unless the fix is obvious and local.
- Never stack unrelated ideas in a single run.
- Every accepted winner must be rerun at least once from a clean state.
- Every 5 accepted wins, do one confirmation run with a second seed.
- Prefer reproducible wins over flashy one-offs.

## Recommended Initial Experiment Order

1. baseline seed A
2. baseline seed A again
3. baseline seed B
4. `QK_GAIN_INIT` lower than default
5. `QK_GAIN_INIT` higher than default
6. `MUON_MOMENTUM` higher than default
7. `MUON_MOMENTUM_WARMUP_STEPS` larger
8. `SCALAR_LR` lower than `MATRIX_LR`
9. `GRAD_CLIP_NORM = 1.0`
10. `LOGIT_SOFTCAP` lower
11. `LOGIT_SOFTCAP` higher
12. `ROPE_BASE` higher

Then move to:

- `skip_weights` init
- `resid_mix` init
- `attn_scale` init
- `mlp_scale` init
- MLP activation variants
- kv head ratio
- width/depth tradeoff

## How To Evaluate Each Run

Record:

- `val_bpb`
- `val_loss`
- `artifact_int8_bytes`
- `tok_s`
- `train_time_ms`
- `step_avg_ms`

If the script already emits final serialized artifact size, use that.
If a result is ambiguous, rerun before keeping.

## Git Workflow

- Start from the current best branch state.
- Create a commit for the proposed change.
- Run the experiment.
- If better by the acceptance rule, keep the commit.
- If worse or noisy-negative, reset hard back to the previous kept commit.
- Do not create a new branch for each try.
- Optionally tag milestone winners like `best-p1`, `best-p2`, `best-final`.

## What Not To Do

- do not add dependencies
- do not do broad refactors in early rounds
- do not trust one lucky seed
- do not optimize only throughput at the cost of `val_bpb`
- do not overwrite logging files with incomplete data
- do not merge to `main` automatically
- do not get stuck polishing one weak idea for too long

## Final Deliverables By Morning

1. top 5 candidates ranked by `val_bpb`
2. top 3 Pareto candidates by loss, size, and speed
3. exact diff for the current best candidate
4. concise summary of ideas that consistently helped
5. concise summary of ideas that failed or were inconclusive
6. current best commit hash and branch name
7. recommendation for which candidate should be scaled to larger hardware next

Operate autonomously, but stay disciplined, conservative, and evidence-driven.
