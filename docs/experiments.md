# Experiments and failure analysis

## Training configuration

| Parameter | Value |
|---|---|
| Policy | π0.5 |
| Steps | 30,000 |
| GPUs | 4 × RTX 4090 |
| Per-rank microbatch | 1 |
| Gradient accumulation | 16 |
| Effective batch | 64 |
| Precision | bfloat16 |
| Checkpoints | 10k, 20k, 30k |
| EMA | disabled |
| Trainable modules | action expert + vision encoder |
| Normalization | quantile state/action, identity image |

Training was interrupted after the last durable 10k checkpoint and resumed from 10k. The plotted curve therefore uses the original 0–10k segment and the resumed 10k–30k segment.

## Checkpoint/action-chunk ablation

Eight fixed task/seed pairs were evaluated for each cell.

| Checkpoint | `n_action_steps=1` | `n_action_steps=10` |
|---|---:|---:|
| 10k | 0/8 (0%) | 2/8 (25%) |
| 20k | 5/8 (62.5%) | 6/8 (75%) |
| 30k | 4/8 (50%) | **7/8 (87.5%)** |

This ablation selected the 30k checkpoint and `n_action_steps=10`. The 87.5% value is an exploratory single-seed-per-task result, not the headline benchmark.

## Paired multi-seed Base vs FT benchmark

Protocol: eight tasks × ten new environment seeds (`12000..12009`) × two policies. Both policies receive identical environment seeds and identical deterministic policy-seed schedules. Horizon is 300 and `n_action_steps=10`.

| Policy | Success | Grasp | Episodes |
|---|---:|---:|---:|
| Base | 0/80 (0%) | 0/80 (0%) | 80 |
| FT 30k | **43/80 (53.75%)** | **54/80 (67.5%)** | 80 |

### Fine-tuned success by task

| Task | Success |
|---|---:|
| green cube → blue bin | 8/10 (80%) |
| green cube → red bin | 6/10 (60%) |
| small box → blue bin | 6/10 (60%) |
| small box → red bin | 5/10 (50%) |
| cylinder → blue bin | 5/10 (50%) |
| cylinder → red bin | 5/10 (50%) |
| red cube → blue bin | 4/10 (40%) |
| red cube → red bin | 4/10 (40%) |

Raw lightweight results are in [`results/base_vs_ft_10seeds.json`](../results/base_vs_ft_10seeds.json) and [`results/checkpoint_ablation.json`](../results/checkpoint_ablation.json).

## Typical failures

1. **Base fails before grasping.** Across all 80 episodes Base recorded no successful grasp. This indicates a substantial visual/control-domain gap rather than only a placement problem. See the [Base failure video](../assets/videos/base_red_cube_blue_bin_failure.mp4).
2. **FT grasps but misses placement.** FT grasped in 54 episodes but completed only 43, leaving 11 grasp-without-success cases. The cylinder example reaches the object but fails to finish placement within the horizon. See the [placement failure video](../assets/videos/ft_cylinder_place_failure.mp4).
3. **Red-cube tasks are weakest.** Both red-cube variants reach only 40%, suggesting object-appearance or layout coverage should be expanded.
4. **Action clipping remains frequent.** Successful and failed trajectories can both contain clipped commands. Future work should quantify clipping by phase and compare action scaling/resampling.

## Interpretation

The paired benchmark demonstrates that fine-tuning learned task-relevant behavior: absolute success increased by 53.75 percentage points and grasp success by 67.5 points. It does not establish real-robot transfer or generalization outside the configured object/bin family.
