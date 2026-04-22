# RunPod First PR

## Goal

Get one real `1xH100` validation pass today, then turn the best run into a same-day non-record PR scaffold.

## Budget

- Recommended spend now: `$25-30`
- Keep `$10-15` in reserve
- Use the reserve only for one of:
  - rerun the single best candidate with a second seed
  - rerun the winner if the first result looks suspicious or contaminated
  - one narrow follow-up only if the first queue strongly favors compact architecture
- Do not spend the reserve on `8xH100` before the first PR unless the `1xH100` result is surprisingly strong and clearly worth escalating

## What This Uses

- CUDA path: [train_gpt.py](/Users/sanilbaweja/Projects/parameter-golf/train_gpt.py)
- Remote bootstrap: [scripts/runpod_bootstrap.sh](/Users/sanilbaweja/Projects/parameter-golf/scripts/runpod_bootstrap.sh)
- Repo sync: [scripts/runpod_sync_repo.sh](/Users/sanilbaweja/Projects/parameter-golf/scripts/runpod_sync_repo.sh)
- First H100 queue: [scripts/runpod_first_h100_queue.sh](/Users/sanilbaweja/Projects/parameter-golf/scripts/runpod_first_h100_queue.sh)
- Result fetch: [scripts/runpod_fetch_results.sh](/Users/sanilbaweja/Projects/parameter-golf/scripts/runpod_fetch_results.sh)
- Log parser: [scripts/parse_train_log.py](/Users/sanilbaweja/Projects/parameter-golf/scripts/parse_train_log.py)
- PR scaffold helper: [scripts/prepare_non_record_submission.py](/Users/sanilbaweja/Projects/parameter-golf/scripts/prepare_non_record_submission.py)

## First Remote Runs

The first queue runs four screens:

1. baseline `train_gpt.py`
2. `MLP_ACT=silu`
3. `MLP_ACT=silu TIED_EMBED_LR=0.04`
4. `MLP_ACT=silu TIED_EMBED_LR=0.04 MODEL_DIM=384`

That is enough to answer whether the strongest local signals transfer to CUDA before spending more credit.

## Operator Steps

1. Create a RunPod `1xH100` pod from the official Parameter Golf template in the README.
2. Enable SSH and note the SSH target, for example `root@<pod-ip>`.
3. Sync this repo to the pod:

```bash
cd /Users/sanilbaweja/Projects/parameter-golf
bash scripts/runpod_sync_repo.sh root@<pod-ip>
```

4. SSH into the pod and bootstrap data:

```bash
ssh root@<pod-ip>
cd /workspace/parameter-golf
bash scripts/runpod_bootstrap.sh
```

5. Run the first H100 queue:

```bash
cd /workspace/parameter-golf
bash scripts/runpod_first_h100_queue.sh
```

6. Back on the laptop, fetch the logs:

```bash
cd /Users/sanilbaweja/Projects/parameter-golf
bash scripts/runpod_fetch_results.sh root@<pod-ip>
```

7. Review the summary:

```bash
cat /Users/sanilbaweja/Projects/parameter-golf/logs/runpod_first_h100/summary.tsv
```

8. Select the best clean run automatically:

```bash
cd /Users/sanilbaweja/Projects/parameter-golf
python3 scripts/pick_best_runpod_run.py logs/runpod_first_h100/summary.tsv
```

9. Turn the best log into a non-record submission scaffold:

```bash
cd /Users/sanilbaweja/Projects/parameter-golf
python3 scripts/prepare_non_record_submission.py \
  --log logs/runpod_first_h100/<best-log>.log \
  --folder records/track_non_record_16mb/2026-04-20_RunPod_H100_FirstPR \
  --name "RunPod H100 First PR" \
  --author "<your name>" \
  --github-id "<your github id>" \
  --gpu "1xH100" \
  --blurb "First 1xH100 non-record submission validating the local SiLU and compact-model leads before an 8xH100 push."
```

10. Edit the generated `README.md` in that folder:
   - replace the TODO bullets with the actual hypothesis and what transferred
   - note which run won and why

11. If needed, use the reserve-budget follow-up queue:

```bash
cd /workspace/parameter-golf
bash scripts/runpod_followup_h100_queue.sh rerun_best_seed2024 h100_silu_embed004_dim384_sp1024
```

or

```bash
cd /workspace/parameter-golf
bash scripts/runpod_followup_h100_queue.sh dim384_vs_mlp1
```

12. Open the PR:
   - include only the new folder under `records/track_non_record_16mb/`
   - do not mix in local research harness files

## Notes

- The first PR does not need to be leaderboard-competitive. It needs to be real, reproducible, and interesting enough to support the compute-grant request.
- The current best local transfer candidates are `MLP_ACT=silu`, `TIED_EMBED_LR=0.04`, and `MODEL_DIM=384`.
- Candidate choice rule:
  - prefer the lowest clean `val_bpb` under the 16MB total artifact cap
  - if `silu + tied_embed_lr=0.04 + dim384` wins cleanly, use it
  - if plain `silu + tied_embed_lr=0.04` beats `dim384`, use the simpler candidate
  - if only `silu` transfers, use `silu`
  - if none of the local improvements transfer, still submit the best clean `1xH100` run as a non-record transfer study
