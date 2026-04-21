# Round 6 Plan

## Goal

Use the next full-night batch for higher-information experiments than round 5.

Reason:
- Round 5 finished early because the candidate list was too short.
- Round 5 did not produce a promotable win.
- The current local base appears stable enough for ranking ideas, but tiny scalar nudges are showing diminishing returns.

## Current Provisional Local Base

- `SCALAR_LR=0.02`
- `GRAD_CLIP_NORM=1.0`
- `LOGIT_SOFTCAP=20.0`
- `WARMDOWN_ITERS=3200`

## Round 5 Takeaway

- `TIED_EMBED_LR=0.04` and `TIED_EMBED_LR=0.035` were mildly positive but below promotion threshold.
- `NUM_KV_HEADS=2` was mildly positive but not decisive and cost artifact size.
- `NUM_KV_HEADS=8` was contaminated.
- No clear new base emerged.

## Round 6 Direction

Do not spend another whole night on only tiny scalar tweaks.

Use a larger, fuller queue with:

1. Structural lane
- activation swap lane
- `relu^2 -> SiLU`
- `relu^2 -> squared SiLU`
- optionally one clean recurrence retry only if artifact serialization remains usable

2. Architecture lane
- `NUM_KV_HEADS=2` confirmation
- one neighboring architecture follow-up only if justified

3. Cleanup lane
- one final `TIED_EMBED_LR` confirmation block if time remains

## Execution Shape

- Make round 6 a true full-night queue.
- Prefer `10-12` candidate blocks, or a two-stage queue.
- If phase A finishes early, use leftover time for confirmation reruns.

Suggested structure:

- Phase A:
  - `4-5` structural / architecture experiments
- Phase B:
  - `3-5` follow-up scalar or confirmation experiments
- Phase C:
  - rerun the best phase-A candidate once if time remains

## Round 6 Candidate Priorities

Highest value:
- activation swap (`SiLU`, squared `SiLU`)
- `NUM_KV_HEADS=2` confirmation

Secondary:
- `TIED_EMBED_LR=0.04` or `0.035` confirmation

Deprioritized for now:
- more warmdown-only search
- more warmup-only search
- more tiny Muon schedule nudges without a stronger signal

## Strategic Note

Local MLX should continue to rank ideas, not decide the final submission.

The next strong local signal should be something that is more likely to transfer to the actual submission environment than another tiny laptop-specific schedule effect.
