# Hypotheses

## Baseline Noise Estimation

- hypothesis: The unmodified MLX baseline has measurable run-to-run variation, and tuning decisions should be gated by an empirical noise floor.
- why it might help: Without a noise estimate, small apparent improvements can be lucky variance rather than real signal.
- experiments tried: `1f0fac01be2ebd56409cabc4ad2cdc3d53ac150e` crashed inside the sandbox before training because MLX could not enumerate a Metal device; `c2f461379d6f9d939072568a7ee27f758e2e1e4c`, `8aff8d93f52163c8d57f12cb83248f5156293845`, and `8382f619674b020a31ff316338d4f2556498c4bb` are successful same-seed (`1337`) baselines with final `val_bpb` values `9.93275667`, `7.66305888`, and `6.72339385`; `714fe1052582df739c86fbf40524840e7b9c5fad` is a failed second-seed attempt blocked by sandboxed Metal access.
- status: inconclusive

## QK Gain Lower

- hypothesis: Lowering `QK_GAIN_INIT` below the default may stabilize early optimization and improve proxy `val_bpb`.
- why it might help: Smaller initial QK gain could reduce attention sharpness early in training and make the short training budget more sample-efficient.
- experiments tried: `12641775103ed2b14f0c4010532542c7e129bd0c` lowered the default `QK_GAIN_INIT` from `1.5` to `1.25` and regressed on the smoke proxy from `2.30477171` to `2.37863099`, while also increasing compressed size from `9897212` to `11127690`.
- status: contradicted

## QK Gain Higher

- hypothesis: Raising `QK_GAIN_INIT` modestly above the default may improve early attention formation and reduce short-budget proxy `val_bpb`.
- why it might help: A slightly stronger starting QK gain can sharpen attention sooner, which may be useful when the training budget is extremely short.
- experiments tried: `56c1845aeb081de3d7c6968d7524121f9c979b47` raised the default `QK_GAIN_INIT` from `1.5` to `1.75` under the fast proxy harness and improved `val_bpb` from `2.44043721` to `2.41122899`, while also improving compressed size from `11262005` to `11239422`; `a48087dd91db1c2ea6e41f8bd0d0cd09320cbdb8` reran the same configuration and came back at `2.43602055` with a larger compressed artifact `11293354`, so the apparent win did not reproduce cleanly.
- status: inconclusive

## Muon Momentum Higher

- hypothesis: Raising `MUON_MOMENTUM` modestly above the default may improve early optimizer stability and lower short-budget proxy `val_bpb`.
- why it might help: A slightly higher momentum can smooth updates in the matrix optimizer and help the model use a short training budget more efficiently.
- experiments tried: `a3037b4d32ae19fae8b890d9afba9366f42309d0` raised the default `MUON_MOMENTUM` from `0.95` to `0.97` under the fast proxy harness and came back at `2.44366839` versus the fast baseline `2.44043721`, while also increasing compressed artifact size from `11262005` to `11295550`; on H100, `h100_muon_momentum_0975` improved to `1.33240794`, and the clean confirmation `h100_muon_momentum_0975_seed1337_r2` improved further to `1.31752605` with `1333116.35` tok/s; combined with the promoted throughput recipe, `h100_tp_gradacc4_gc10_muon0975_seed1337` reached the raw-lowest `1.29971188`, but the incremental gain was only `0.00043011` bpb and came with a larger `15513390` byte artifact.
- status: weak support

## Scalar LR Lower

- hypothesis: Lowering `SCALAR_LR` below `MATRIX_LR` may regularize scalar parameters and improve short-budget proxy `val_bpb`.
- why it might help: Biases, norms, and other scalar parameters can overreact early; a lower scalar LR may let matrix updates do more of the representation shaping without destabilizing auxiliary parameters.
- experiments tried: `79fd0bb11c8de4c282b2777553ca30f40e63aa80` lowered the default `SCALAR_LR` from `0.04` to `0.02` under the fast proxy harness and improved `val_bpb` from `2.44043721` to `2.40609621`; `2ee241a6c3a5c765b840e681dcda14af259df425` reran the same configuration and improved further to `2.38628306`, while both runs carried a larger compressed artifact than the default fast baseline.
- status: supported

## Grad Clip Norm

- hypothesis: Enabling moderate gradient clipping may improve early stability and reduce short-budget proxy `val_bpb` on top of the current best fast-proxy optimizer settings.
- why it might help: A small clip can prevent occasional large updates from destabilizing the short run, especially once the scalar LR is reduced and the optimization path becomes smoother.
- experiments tried: `8f9d3247fc3162527d6ca58f732c1544ec8a4380` enabled `GRAD_CLIP_NORM=1.0` on top of the confirmed `SCALAR_LR=0.02` fast-proxy base and improved `val_bpb` from `2.38628306` to `2.37371169`, while also reducing compressed artifact size from `11330554` to `11189754`; `c7e28da9fe1bbf210c1b4b49ecf9d36158047358` reran the same configuration and improved further to `2.35927199` with compressed artifact `11100841`; on H100, `h100_grad_clip_10` improved to `1.330568`, and the combined throughput confirmation `h100_tp_gradacc4_gc10_seed1337_r2` reached `1.30014199`.
- status: supported

## H100 Throughput: Lower Grad Accumulation

- hypothesis: Reducing gradient accumulation while keeping total batch tokens controlled can improve useful H100 throughput and short-budget quality.
- why it might help: Fewer accumulation microsteps reduce optimizer overhead and keep the GPU busier; if the batch schedule remains stable, this can improve both `tok_s` and the amount of useful training completed inside the 10-minute wall clock.
- experiments tried: `h100_tp_gradacc4_gc10` was the first promoted throughput result at `1.30813214` bpb and `1472016.17` tok/s; clean confirmation `h100_tp_gradacc4_gc10_seed1337_r2` improved to `1.30014199` bpb, `2.19523530` val loss, `1626354.81` tok/s, `85.01%` mean GPU utilization, and `26.02%` mean VRAM use; second-seed confirmation `h100_tp_gradacc4_gc10_seed2024` landed at `1.30794507` bpb and `1472595.01` tok/s.
- status: supported

## Logit Softcap Lower

- hypothesis: Lowering `LOGIT_SOFTCAP` below the default may reduce overly sharp logits early and improve short-budget proxy `val_bpb`.
- why it might help: Softer capped logits can regularize the output distribution during the brief training window and may compress better if activations stay in a narrower range.
- experiments tried: `383428f52bded56429d041b415bd2cf1a462a3a3` lowered the default `LOGIT_SOFTCAP` from `30.0` to `20.0` on top of the confirmed `SCALAR_LR=0.02` and `GRAD_CLIP_NORM=1.0` fast-proxy base, improving `val_bpb` from `2.35927199` to `2.34026166` and reducing compressed artifact size from `11100841` to `10900773`; `6ab3c0a783055d69c2a5efc640f949bb980dbd71` reran the same configuration and improved further to `2.33443439` with compressed artifact `10829872`.
- status: supported
