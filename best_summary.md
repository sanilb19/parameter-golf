# Current Best Summary

- current best commit: `208bbc14d25064bd7f03644bd997b76fd08123e3`
- current best H100 run: `h100_tp_gradacc4_gc10_seed1337_r2`
- current best val_bpb: `1.30014199`
- current best val_loss: `2.19523530`
- current best artifact_int8_bytes: `14785037`
- current best tok_s: `1626354.81`
- short description of why this candidate is currently best: clean 1xH100 confirmation of `GRAD_ACCUM_STEPS=4 + GRAD_CLIP_NORM=1.0`, supported by second-seed run `h100_tp_gradacc4_gc10_seed2024` at `1.30794507` bpb. The raw-lowest combined run is not promoted because the gain is below the noise gate and costs more artifact size.
- current Pareto alternates for size and speed: raw-loss alternate `h100_tp_gradacc4_gc10_muon0975_seed1337` at `1.29971188` bpb, `15513390` bytes, and `1610864.29` tok/s; second-seed robustness alternate `h100_tp_gradacc4_gc10_seed2024` at `1.30794507` bpb, `14455311` bytes, and `1472595.01` tok/s; optimizer alternate `h100_muon_momentum_0975_seed1337_r2` at `1.31752605` bpb, `14857373` bytes, and `1333116.35` tok/s.
