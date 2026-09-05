# Published results

- `base_vs_ft_10seeds.json`: headline paired evaluation (8 tasks × 10 seeds × 2 policies).
- `checkpoint_ablation.json`: 10k/20k/30k and `n_action_steps=1/10` exploratory evaluation, including episode rows.
- `task_success.csv`: task-level counts used by the public figure.
- `training_loss.csv`: loss values parsed every ten optimizer steps from the initial and resumed logs.

Definitions:

- **grasp success**: the target object was detected as grasped at least once during the episode.
- **place success**: the target object satisfied the target-bin placement condition.
- **task success**: the complete target-object/target-bin command was satisfied before the 300-step horizon.

The Base and FT headline comparison uses paired environment seeds and the same deterministic policy-seed schedule. Intermediate 87.5% results in `checkpoint_ablation.json` use only one seed per task and should not be reported as the main benchmark.
