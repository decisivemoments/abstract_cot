# Abstract-CoT

Minimal reproduction scaffold for Abstract Chain-of-Thought.

## Layout

- `configs/`: experiment and runtime configs
- `scripts/`: thin CLI entrypoints
- `src/abstract_cot/`: package code
- `tests/`: unit tests for core invariants

## Initial scope

This scaffold focuses on the warm-up path:

- abstract vocabulary construction
- tokenizer extension metadata
- constrained abstract decoding
- bottleneck attention mask

RL and full training loops are intentionally staged on top of these primitives.

## Remote Workflow

This project is designed for:

- local git management on your laptop
- one-way code sync to `cisl113` via `mutagen`
- server environment managed by `conda`, with dependencies installed by `uv pip`
- model and dataset downloads only on `cisl113`

### Mutagen

In the project root `abstract_cot/`, start sync directly with Mutagen:

```bash
cd abstract_cot
mutagen project start
```

List project sessions:

```bash
cd abstract_cot
mutagen project list
```

Flush pending changes:

```bash
cd abstract_cot
mutagen project flush
```

Terminate and recreate if needed:

```bash
cd abstract_cot
mutagen project terminate
mutagen project start
```

The sync is intentionally `one-way-safe` from local to remote. The remote copy is a runtime mirror, not the git source of truth.

The sync definition now lives in:

- [mutagen.yml](/Users/zhangjunyi/project/abstrcot/abstract_cot/mutagen.yml:1)

### Server Bootstrap

After the first sync, SSH into `cisl113` and run:

```bash
cd ~/workspace/abstract_cot
bash scripts/bootstrap_server.sh
```

This creates:

- a conda environment named `abstract-cot`
- dependencies installed inside that conda environment via `uv pip`
- cache directories under `~/workspace/.cache/abstract_cot`
- local asset directories under `server_assets/`

The runtime defaults are defined in:

- [cisl113.yaml](/Users/zhangjunyi/project/abstrcot/abstract_cot/configs/runtime/cisl113.yaml:1)

### Model And Dataset Downloads

Only download assets on `cisl113`.

Baseline download:

```bash
cd ~/workspace/abstract_cot
bash scripts/download_qwen_and_data.sh
```

This defaults to:

- model: `Qwen/Qwen3-0.6B`
- warm-up data: `dolcivallone/dolci-think-sft`
- RL data: `dolcivallone/dolci-think-rl`

The download path uses `HF_ENDPOINT=https://hf-mirror.com` by default.

Additional model configs are available in:

- [qwen3_0p6b.yaml](/Users/zhangjunyi/project/abstrcot/abstract_cot/configs/model/qwen3_0p6b.yaml:1)
- [qwen3_1p7b.yaml](/Users/zhangjunyi/project/abstrcot/abstract_cot/configs/model/qwen3_1p7b.yaml:1)
- [qwen3_4b.yaml](/Users/zhangjunyi/project/abstrcot/abstract_cot/configs/model/qwen3_4b.yaml:1)

### First Warm-Up Run

After sync, bootstrap, and downloads complete on `cisl113`, the first smoke run is:

```bash
cd ~/workspace/abstract_cot
bash scripts/run_server_warmup.sh
```

The current MVP script:

- loads the local Qwen model from `server_assets/models/Qwen3-0.6B`
- extends the tokenizer with abstract tokens
- resizes embeddings
- loads the local Hugging Face parquet dataset via `datasets.load_dataset(...)`
- runs a minimal bottleneck/distillation SFT pass

Expected dataset path in the current config:

`server_assets/datasets/Dolci-Think-SFT-7B`

If the mirrored dataset layout differs after download, update [mvp_warmup.yaml](/Users/zhangjunyi/project/abstrcot/abstract_cot/configs/experiment/mvp_warmup.yaml:1) before running.

### Experimental Continuous Latent Branch

An experimental branch is now included for continuous latent reasoning.

Idea:

- run the prompt through the model
- take the last-layer hidden state of the final position
- if the predicted stop token is not `<endabstract>`, feed that hidden state back as the next `inputs_embeds`
- repeat for a bounded number of reasoning steps

This is intentionally separate from the discrete abstract-token path.

Current entry points:

- [continuous_latent_mvp.yaml](/Users/zhangjunyi/project/abstrcot/abstract_cot/configs/experiment/continuous_latent_mvp.yaml:1)
- [recurrent_latent.py](/Users/zhangjunyi/project/abstrcot/abstract_cot/src/abstract_cot/continuous/recurrent_latent.py:1)
- [continuous_experiment.py](/Users/zhangjunyi/project/abstrcot/abstract_cot/src/abstract_cot/training/continuous_experiment.py:1)
- [inspect_continuous_latent.py](/Users/zhangjunyi/project/abstrcot/abstract_cot/scripts/inspect_continuous_latent.py:1)

Current support level:

- continuous latent trace generation
- composing `prompt + latent states + answer` for SFT-style forward passes
- collecting latent trace + generated answer for RL-style trajectories

This branch is experimental and not yet wired into the main warm-up runner.

### Notes

- The remote environment was not auto-probed from this session, so `~/workspace/abstract_cot` is an explicit convention, not a verified existing directory.
- `server_assets/` should be kept out of Mutagen-managed sync, even if it remains inside the project directory on the server.
