# Research Memory

## Environment

- Repo: `/Users/sanilbaweja/Projects/parameter-golf`
- Active research branch: `autoresearch/overnight-01`
- Goal: improve `train_gpt_mlx.py` locally, then transfer the best ideas into `train_gpt.py` and a `/records` submission
- MLX/Metal only works when commands run outside the Codex sandbox; sandboxed `mlx.core` import crashes during device enumeration
- This M3 laptop has meaningful run-to-run drift even with unchanged settings
- Runtime state matters: compile/warmup, thermal drift, memory pressure, and MLX/Metal state all move both `tok_s` and final `val_bpb`
- Wide unattended local grids are unreliable on this machine unless tightly bracketed by nearby baselines
- The most reliable local procedure is paired `baseline -> candidate -> baseline`
- CUDA/H100 work must go through `train_gpt.py`, not only `train_gpt_mlx.py`
- RunPod H100 pods must be created in `US-MO-1` with network volume `j2e4t9p20a` attached and mounted at `/workspace/persist`; `scripts/runpod_bootstrap.sh` uses the populated cache at `/workspace/persist/parameter-golf-cache` so future runs do not redownload the `sp1024` dataset.

## Reliable Findings

- Early stable scalar gains on the MLX proxy base:
  - `SCALAR_LR=0.02`
  - `GRAD_CLIP_NORM=1.0`
  - `LOGIT_SOFTCAP=20.0`
- Warmdown above the stock `1200` clearly helps in the local proxy loop; the useful region is roughly `2400-3200`
- `WARMDOWN_ITERS=3200` became the stable local base after rounds 2-4
- Activation swap is real:
  - `MLP_ACT=silu` beat `relu2` in matched 10-minute local runs
  - `MLP_ACT=silu2` was worse and should be treated as a dead end
- Round 6 established that the strongest interaction is:
  - `MLP_ACT=silu + TIED_EMBED_LR=0.04`
  - `TIED_EMBED_LR` reductions alone were not winners
  - `NUM_KV_HEADS=2` lost in that regime
- Round 7 produced the strongest architecture signal so far:
  - `MODEL_DIM=384 NUM_HEADS=8 NUM_KV_HEADS=4` on top of the `silu + tied_embed_lr=0.04` base
  - This improved in stage 1, on seed `2024`, and in a 10-minute confirmation run on seed `7`
  - It also materially reduced artifact size
- Strong secondary lead from round 7:
  - `MLP_MULT=1`
  - good loss and much smaller artifact, but not as strong as the `MODEL_DIM=384` result

## Unreliable / Conditional Findings

- The earliest local baselines drifted badly; old full-baseline numbers are not directly comparable to later warmed proxy runs
- Top-5 leaderboard-inspired local analogues were useful for intuition, not direct adoption:
  - `parallel residuals` looked interesting once, then lost in the cleaner grid
  - higher `QK_GAIN_INIT` did not hold up in cleaner comparisons
- Recurrence and SDClip-style local analogues showed some promising pre-quant BPB, but those results are conditional:
  - several runs failed during artifact serialization with `mx.savez(...): std::bad_cast`
  - they are not yet safe mainline candidates
- `NUM_LAYERS=8` looked strong in short screens, but failed the 10-minute confirmation in round 7
- `NUM_KV_HEADS` changes have not paid off in the current regime
- `silu + WARMDOWN_ITERS=2800` had one good night but did not confirm as cleanly as `silu + tied_embed_lr=0.04`
- Any long unattended local batch should be treated as conditional unless the block has low baseline spread

## Current Best Local Base

- Current best local recipe to start from:
  - `SCALAR_LR=0.02`
  - `GRAD_CLIP_NORM=1.0`
  - `LOGIT_SOFTCAP=20.0`
  - `WARMDOWN_ITERS=3200`
  - `MLP_ACT=silu`
  - `TIED_EMBED_LR=0.04`
- Current leading local architecture candidate:
  - `MODEL_DIM=384`
  - `NUM_HEADS=8`
  - `NUM_KV_HEADS=4`
- Strong secondary compact candidate:
  - `MLP_MULT=1`

## Best Evidence Snapshot

- Matched 10-minute local activation runs:
  - `relu2`: `val_bpb=1.76625256`, artifact `11892511`
  - `silu`: `val_bpb=1.75528746`, artifact `11955849`
  - `silu2`: `val_bpb=1.83008542`, artifact `11618778`
- Round 6 best stage-1 candidate:
  - `MLP_ACT=silu + TIED_EMBED_LR=0.04`
  - `val_bpb=1.71864792`
  - delta vs baseline mean `-0.07037458`
- Round 7 best repeated candidate:
  - stage 1: `MODEL_DIM=384 NUM_HEADS=8 NUM_KV_HEADS=4`, `val_bpb=1.81968446`, artifact `6539102`
  - seed `2024`: `val_bpb=1.81509108`, artifact `6517312`
  - seed `7`, 10-minute confirm: `val_bpb=1.67830404`, artifact `7575034`

## Run Protocol

- Local MLX:
  - use paired `baseline -> candidate -> baseline`
  - keep all non-target knobs fixed
  - prefer 6-10 minute capped runs or short proxy runs with clean comparisons
  - re-baseline often; hour-old baselines are not trustworthy on this laptop
  - discard or quarantine runs with obvious timing/throughput pathologies
- Promotion:
  - do not promote a local win from one lucky run
  - prefer repeated wins across at least one additional seed or a longer confirmation
- Search posture now:
  - scalar-only tuning is near diminishing returns
  - next valuable lanes are compact architecture, transfer to CUDA/H100, and later recurrence/quantization
- RunPod/H100:
  - always attach network volume `j2e4t9p20a` in `US-MO-1`
  - always mount the volume at `/workspace/persist`
  - always run the updated `scripts/runpod_bootstrap.sh` before training so `data/datasets`, `data/tokenizers`, and `HF_HOME` point to `/workspace/persist/parameter-golf-cache`
  - use cheap CPU pods for volume/cache maintenance when possible; keep GPU pods off except for actual training runs
  - use the auto-stop lifecycle runner so pods stop immediately after results are fetched

## Current Code State

- `train_gpt_mlx.py`
  - contains env-controlled local analogues for:
    - `MLP_ACT` with `relu2`, `silu`, `silu2`
    - parallel residuals
    - recurrence
    - SDClip-style quant clipping controls
- `train_gpt.py`
  - now also supports `MLP_ACT` with `relu2`, `silu`, `silu2`
  - this was added specifically so the best local activation finding can be validated on H100/RunPod

## H100 / PR Status

- The first remote validation path has been prepared, but not yet confirmed in this memory as completed
- H100 helper scripts now exist:
  - [RUNPOD_FIRST_PR.md](/Users/sanilbaweja/Projects/parameter-golf/RUNPOD_FIRST_PR.md)
  - [scripts/runpod_bootstrap.sh](/Users/sanilbaweja/Projects/parameter-golf/scripts/runpod_bootstrap.sh)
  - [scripts/runpod_sync_repo.sh](/Users/sanilbaweja/Projects/parameter-golf/scripts/runpod_sync_repo.sh)
  - [scripts/runpod_first_h100_queue.sh](/Users/sanilbaweja/Projects/parameter-golf/scripts/runpod_first_h100_queue.sh)
  - [scripts/runpod_fetch_results.sh](/Users/sanilbaweja/Projects/parameter-golf/scripts/runpod_fetch_results.sh)
  - [scripts/parse_train_log.py](/Users/sanilbaweja/Projects/parameter-golf/scripts/parse_train_log.py)
  - [scripts/prepare_non_record_submission.py](/Users/sanilbaweja/Projects/parameter-golf/scripts/prepare_non_record_submission.py)
- The first planned RunPod queue is:
  - baseline `train_gpt.py`
  - `MLP_ACT=silu`
  - `MLP_ACT=silu + TIED_EMBED_LR=0.04`
  - `MLP_ACT=silu + TIED_EMBED_LR=0.04 + MODEL_DIM=384`
- Submission strategy:
  - first goal is a real non-record PR under `records/track_non_record_16mb/`
  - this is intended to support a compute-grant request for larger H100 budget

## Important Files

- Main research ledger:
  - [results.tsv](/Users/sanilbaweja/Projects/parameter-golf/results.tsv)
  - [results.jsonl](/Users/sanilbaweja/Projects/parameter-golf/results.jsonl)
  - [hypotheses.md](/Users/sanilbaweja/Projects/parameter-golf/hypotheses.md)
  - [best_summary.md](/Users/sanilbaweja/Projects/parameter-golf/best_summary.md)
- Local high-signal batch summaries:
  - [logs/overnight_local_ab_round6/summary.md](/Users/sanilbaweja/Projects/parameter-golf/logs/overnight_local_ab_round6/summary.md)
  - [logs/overnight_local_ab_round7/summary.md](/Users/sanilbaweja/Projects/parameter-golf/logs/overnight_local_ab_round7/summary.md)
- Final-candidate scratch space:
  - [records/final_candidates](/Users/sanilbaweja/Projects/parameter-golf/records/final_candidates)
  - [records/final_m3_max](/Users/sanilbaweja/Projects/parameter-golf/records/final_m3_max)
- Learning suite and exploratory artifacts:
  - [logs/top5_learning_suite](/Users/sanilbaweja/Projects/parameter-golf/logs/top5_learning_suite)
  - [top5_learning_suite.py](/Users/sanilbaweja/Projects/parameter-golf/top5_learning_suite.py)
- Overnight harnesses:
  - [overnight_local_ab.py](/Users/sanilbaweja/Projects/parameter-golf/overnight_local_ab.py)
  - [overnight_local_ab_round2.py](/Users/sanilbaweja/Projects/parameter-golf/overnight_local_ab_round2.py)
  - [overnight_local_ab_round3.py](/Users/sanilbaweja/Projects/parameter-golf/overnight_local_ab_round3.py)
  - [overnight_local_ab_round4.py](/Users/sanilbaweja/Projects/parameter-golf/overnight_local_ab_round4.py)
  - [overnight_local_ab_round5.py](/Users/sanilbaweja/Projects/parameter-golf/overnight_local_ab_round5.py)
  - [overnight_local_ab_round6.py](/Users/sanilbaweja/Projects/parameter-golf/overnight_local_ab_round6.py)
  - [overnight_local_ab_round7.py](/Users/sanilbaweja/Projects/parameter-golf/overnight_local_ab_round7.py)
