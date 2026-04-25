# H100 Campaign Decision Log

## Batch 00: Existing evidence ingest

- Ingested the four completed `logs/runpod_first_h100` runs into the new campaign ledger.
- Baseline remains the raw-quality anchor on `1xH100`.
- `MLP_ACT=silu` alone is clearly worse and should not be explored further by itself.
- `MLP_ACT=silu + TIED_EMBED_LR=0.04` is a marginal maybe-win and needs confirmation before promotion.
- `MODEL_DIM=384` remains a compact branch for size-sensitive tradeoffs, not the best raw-`val_bpb` branch.
- Budget note: starting balance observed through `runpodctl user` was approximately `$37.48`, so screening needs to stay bounded and preserve room for a later `8xH100` run plus recovery buffer.

## Batch 01: Parallel throughput sweep

- Ran three parallel throughput experiments from the `GRAD_CLIP_NORM=1.0` baseline.
- `TRAIN_BATCH_TOKENS=786432` improved raw loss modestly but slowed step time versus the best prior configuration.
- `TRAIN_BATCH_TOKENS=1048576` increased utilization and VRAM use but regressed both throughput and `val_bpb`.
- `GRAD_ACCUM_STEPS=4` was the clear winner: `val_bpb=1.30813214`, `step_avg_ms=356.17`, `gpu_util_mean_pct=81.23`.
- Current promoted throughput winner is `h100_tp_gradacc4_gc10`.
- Interpretation: reducing gradient accumulation improved useful work and model quality more than simply increasing token batch size.

## Batch 02: Independent Phase-1 knob sweep

- Ran four baseline-parent, on-plan screens in parallel lanes: `MUON_MOMENTUM=0.975`, `MUON_MOMENTUM_WARMUP_STEPS=1000`, `LOGIT_SOFTCAP=20`, and `LOGIT_SOFTCAP=40`.
- `MUON_MOMENTUM=0.975` produced the best result in this batch at `val_bpb=1.33240794`, but the improvement over baseline (`-0.00976229`) is still inside the campaign's `2 * noise` keep gate, so it is a `maybe` and needs one clean confirmation rerun.
- `MUON_MOMENTUM_WARMUP_STEPS=1000` improved throughput materially (`tok_s=1167237.35`) but the quality gain was smaller than the estimated noise band, so it is a discard for the raw-loss line.
- `LOGIT_SOFTCAP=20` and `LOGIT_SOFTCAP=40` both regressed versus baseline and should be dropped from the current search path.
- Current guidance: keep the promoted throughput recipe separate, and use the next small budget slice on a confirmation rerun of `MUON_MOMENTUM=0.975` plus the next independent baseline-parent screen (`ROPE_BASE` higher) rather than combining knobs prematurely.
- Budget note: balance observed before this batch was approximately `$30.36`; both H100 pods were shut down cleanly after the runs completed.

## Batch 03: Clean confirmation reruns

- Ran two clean confirmation pods in parallel using the auto-stop lifecycle runner: `h100_tp_gradacc4_gc10_seed1337_r2` and `h100_muon_momentum_0975_seed1337_r2`.
- Both pods started cleanly with no pre-existing GPU processes, fetched results locally, sent `ntfy`, and stopped immediately after completion.
- `h100_tp_gradacc4_gc10_seed1337_r2` confirmed and improved the promoted throughput recipe: `val_bpb=1.30014199`, `val_loss=2.19523530`, `tok_s=1626354.81`, `gpu_util_mean_pct=85.01`, `vram_used_mean_pct=26.02`, `bytes_total_int8_zlib=14785037`.
- `h100_muon_momentum_0975_seed1337_r2` also confirmed as a real Phase-1 optimizer improvement: `val_bpb=1.31752605`, `val_loss=2.22458755`, `tok_s=1333116.35`, `gpu_util_mean_pct=77.41`, `vram_used_mean_pct=14.15`, `bytes_total_int8_zlib=14857373`.
- Current promoted 1xH100 candidate is `GRAD_ACCUM_STEPS=4 + GRAD_CLIP_NORM=1.0`. `MUON_MOMENTUM=0.975` is a confirmed alternate, but it should not replace the throughput recipe.
- Next decision: either run one second-seed confirmation of the promoted throughput recipe, or spend a small batch on planned Phase-1 continuations that combine cleanly with the confirmed throughput base.

## Batch 04: Promoted recipe follow-up

- Ran two parallel H100 follow-ups with the cached network volume attached at `/workspace/persist`; bootstrap used `/workspace/persist/parameter-golf-cache` and did not redownload the dataset.
- `h100_tp_gradacc4_gc10_seed2024` confirmed the promoted throughput recipe on a second seed: `val_bpb=1.30794507`, `val_loss=2.20841046`, `tok_s=1472595.01`, `bytes_total_int8_zlib=14455311`.
- `h100_tp_gradacc4_gc10_muon0975_seed1337` produced the raw lowest bpb so far: `val_bpb=1.29971188`, `val_loss=2.19450906`, `tok_s=1610864.29`, `bytes_total_int8_zlib=15513390`.
- Interpretation: the combined `MUON_MOMENTUM=0.975` run is directionally positive, but its improvement over `h100_tp_gradacc4_gc10_seed1337_r2` is only `-0.00043011` bpb and it adds roughly `728353` bytes. Mark as `maybe/hold`, not a promoted replacement.
- Current promoted 1xH100 recipe remains `GRAD_ACCUM_STEPS=4 + GRAD_CLIP_NORM=1.0`; the combined run is a raw-loss alternate that needs either rerun support or an additional reason before scaling.
