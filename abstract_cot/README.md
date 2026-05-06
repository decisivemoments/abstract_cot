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
- one-way code sync to the server via `mutagen`
- server environment managed by `conda`, with dependencies installed by `uv pip`
- model and dataset downloads only on the server

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

After the first sync, SSH into the server and run:

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

- [server.yaml](/Users/zhangjunyi/project/abstrcot/abstract_cot/configs/runtime/server.yaml:1)

Dependency management now uses:

- [requirements.txt](/Users/zhangjunyi/project/abstrcot/abstract_cot/requirements.txt:1)
- [requirements-dev.txt](/Users/zhangjunyi/project/abstrcot/abstract_cot/requirements-dev.txt:1)

The remaining [pyproject.toml](/Users/zhangjunyi/project/abstrcot/abstract_cot/pyproject.toml:1) is kept only as minimal packaging metadata for the `src/` layout. It is no longer the primary dependency manifest.

### Model And Dataset Downloads

Only download assets on the server.

Baseline download:

```bash
cd ~/workspace/abstract_cot
bash scripts/download_qwen_and_data.sh
```

This downloads:

- model: `Qwen/Qwen3-0.6B`
- warm-up data: `allenai/Dolci-Think-SFT-7B`
- RL data: `allenai/Dolci-Think-RL-7B`

Warm-up preprocessing is now a separate step:

```bash
cd ~/workspace/abstract_cot
bash scripts/preprocess_warmup_data.sh
```

This produces a preprocessed Hugging Face dataset directory:

`server_assets/datasets/Dolci-Think-SFT-7B-cot`

The `-cot` dataset is saved with `datasets.save_to_disk(...)` and contains columns:

- `sample_id`
- `prompt`
- `cot`
- `answer`
- `task_type`

The download path uses `HF_ENDPOINT=https://hf-mirror.com` by default.

Additional model configs are available in:

- [qwen3_0p6b.yaml](/Users/zhangjunyi/project/abstrcot/abstract_cot/configs/model/qwen3_0p6b.yaml:1)
- [qwen3_1p7b.yaml](/Users/zhangjunyi/project/abstrcot/abstract_cot/configs/model/qwen3_1p7b.yaml:1)
- [qwen3_4b.yaml](/Users/zhangjunyi/project/abstrcot/abstract_cot/configs/model/qwen3_4b.yaml:1)

### First Warm-Up Run

After sync, bootstrap, downloads, and warm-up preprocessing complete on the server, the first smoke run is:

```bash
cd ~/workspace/abstract_cot
bash scripts/run_server_warmup.sh
```

Single-node 8 GPU FSDP smoke run:

```bash
cd ~/workspace/abstract_cot
USE_FSDP=true NPROC_PER_NODE=8 bash scripts/run_server_warmup.sh
```

The current MVP script:

- loads the local Qwen model from `server_assets/models/Qwen3-0.6B`
- extends the tokenizer with abstract tokens
- resizes embeddings
- loads the local Hugging Face parquet dataset via `datasets.load_dataset(...)`
- loads the preprocessed `-cot` parquet dataset directly
- partitions data into `D_t,1` and `D_t,2` for each warm-up round
- runs Stage 1 policy-iteration warm-up round by round
- treats `warmup.batch_size` as per-rank batch size in distributed runs

Expected dataset path in the current config:

`server_assets/datasets/Dolci-Think-SFT-7B-cot`

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
- `build/lib/abstract_cot` and `src/*.egg-info` are Python packaging build artifacts. They appear when setuptools-based installation/build steps run. They are not source directories and should remain ignored.
