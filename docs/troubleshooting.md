# Troubleshooting

## PaliGemma tokenizer returns 401

π0.5 preprocessing may request `google/paligemma-3b-pt-224`. Accept the model terms and authenticate once, or point `PI05_TOKENIZER_PATH` to a complete local snapshot:

```bash
export PI05_TOKENIZER_PATH=/path/to/models--google--paligemma-3b-pt-224/snapshots/<revision>
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

Do not commit Hugging Face tokens.

## EGL device assertion

If MuJoCo reports that `MUJOCO_EGL_DEVICE_ID` is not in `CUDA_VISIBLE_DEVICES`, expose the physical EGL GPU explicitly:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 MUJOCO_EGL_DEVICE_ID=3 MUJOCO_GL=egl python ...
```

Remember that long-lived shell exports can leak from a policy process into the next environment launch. The evaluation worker scopes policy and server variables in separate subshells.

## TorchCodec warnings

The tested environment could not load TorchCodec's FFmpeg shared libraries and fell back to PyAV. This warning did not prevent rollout video generation. Install a TorchCodec/FFmpeg combination compatible with the active PyTorch version if native TorchCodec decoding is required.

## Robot does not move or rollout is only a few kilobytes

Check, in order:

1. The server log contains `Waiting for policy client` and later step records.
2. State shape is 8D at the policy boundary. Convert simulator `xyz + xyzw quaternion + gripper2` to `xyz + rotation-vector + gripper2`.
3. Model output is interpreted as 7D delta EEF action, not an absolute pose.
4. Gripper sign and scaling match the expert action convention.
5. Dataset quantile statistics are supplied to both preprocessing and postprocessing.
6. The policy and environment use explicitly recorded seeds.

## `n_action_steps`

π0.5 predicts an action chunk. `n_action_steps=1` executes only its first action before replanning; `n_action_steps=10` executes ten predicted actions before obtaining a new observation. In this benchmark, 30k/n10 outperformed 30k/n1 (87.5% vs 50% in the exploratory ablation).
