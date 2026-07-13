# On-policy distillation with Megatron and vLLM Ascend

This guide covers full-parameter on-policy distillation (OPD) on Ascend NPUs
with Megatron training and vLLM Ascend inference. The canonical recipes are:

- `run_qwen2_5_0_5b_megatron_vllm_ascend.sh`: Qwen2.5-0.5B student and
  Qwen2.5-3B-Instruct teacher on GSM8K.
- `run_qwen3_vl_4b_megatron_vllm_ascend.sh`: Qwen3-VL-4B-Instruct student and
  Qwen3-VL-8B-Instruct teacher on Geo3K.

The recipes use tensor parallelism only. Their default single-node placement is
two NPUs for the colocated student/rollout pool and two NPUs for the teacher
pool. Both pools use TP=2 and PP=1.

## Software compatibility

Start from an Ascend image whose CANN, PyTorch, torch-npu, vLLM, and
vLLM Ascend versions are mutually compatible. Do not install a CUDA-focused
verl image on top of an Ascend runtime.

For CANN 9.0, the Ascend Dockerfiles in this repository use MindSpeed and
Megatron Core from their `core_r0.16.0` branches. Keep those two dependencies
on the same Megatron Core line. The legacy mbridge backend is selected because
the NVIDIA Megatron-Bridge package imports CUDA-only Transformer Engine
components in this environment.

Qwen3-VL support requires an mbridge revision containing the `qwen3_vl`
bridge. The recipes were validated with mbridge commit
`a61943d7fcb34a190471cfeb0a0eb8bbda621ddf`. The PyPI package named `0.15.1`
may not contain that bridge even though it has the same package version. Until
a newer mbridge release includes it, install the tested revision explicitly:

```bash
pip install --no-deps \
  'git+https://github.com/ISEEKYAN/mbridge.git@a61943d7fcb34a190471cfeb0a0eb8bbda621ddf'
```

Verify the bridge before allocating NPUs:

```bash
python - <<'PY'
from mbridge import AutoBridge

assert "qwen3_vl" in AutoBridge.list_supported_models()
PY
```

## Data and models

Prepare GSM8K with the repository utility:

```bash
python examples/data_preprocess/gsm8k.py \
  --local_save_dir "$HOME/data/gsm8k"
```

Prepare Geo3K with the multimodal preprocessing utility used by the canonical
Qwen3-VL examples. The resulting parquet files must contain the `images`
column, and the referenced image paths must be visible from every Ray worker.

Hugging Face model IDs are the defaults. For an offline cluster, set
`STUDENT_MODEL` and `TEACHER_MODEL` to local snapshot directories. ModelScope
can be used as the download source; no recipe changes are required after the
snapshots are stored locally.

## Launch

The text recipe uses the official `forward_kl_topk` loss with top-k 64 and
direct supervised distillation. It does not add a task-reward or policy-gradient
term:

```bash
ray stop --force
bash examples/on_policy_distillation_trainer/\
run_qwen2_5_0_5b_megatron_vllm_ascend.sh
```

The vision-language recipe uses the official `k1` definition as a detached
policy-gradient advantage:

```bash
ray stop --force
bash examples/on_policy_distillation_trainer/\
run_qwen3_vl_4b_megatron_vllm_ascend.sh
```

Both recipes run 100 optimizer steps by default. Override paths and batch
sizes with environment variables rather than editing the scripts:

```bash
STUDENT_MODEL=/models/student \
TEACHER_MODEL=/models/teacher \
TRAIN_FILE=/data/train.parquet \
VAL_FILE=/data/test.parquet \
TRAIN_BATCH_SIZE=64 \
PPO_MINI_BATCH_SIZE=64 \
bash examples/on_policy_distillation_trainer/\
run_qwen2_5_0_5b_megatron_vllm_ascend.sh
```

For long runs, launch in a persistent terminal such as tmux and keep Ray's
temporary directory, model caches, checkpoints, and logs on persistent storage.

## Tuning order

Tune one dimension at a time and retain at least ten steady-state steps after
warmup before comparing throughput.

1. Confirm that the actor, rollout, and teacher pools use all four NPUs and
   that no worker is placed on the wrong pool.
2. Increase `TRAIN_BATCH_SIZE` until generation is continuously batched.
3. Increase `PPO_MINI_BATCH_SIZE` while keeping it no larger than the train
   batch size.
4. Raise `PPO_MAX_TOKEN_LEN_PER_GPU` until memory is well utilized, then reduce
   it if long-tail batches cause out-of-memory failures.
5. Tune rollout and teacher memory utilization independently. Leave headroom
   for weight synchronization and multimodal preprocessing.
6. Enable graph capture only after an eager run succeeds. Graph capture has a
   significant one-time startup cost and should not be included in steady-state
   throughput comparisons.

The end-to-end global token throughput is:

```text
sum(prompt tokens + generated response tokens across the global batch)
-----------------------------------------------------------------------
                         optimizer-step wall time
```

Use the trainer's `perf/total_num_tokens` and `perf/time_per_step` metrics. This
definition includes rollout, teacher inference, log-probability computation,
Megatron forward/backward, optimizer update, and weight synchronization. Do not
report vLLM decode-only throughput as the end-to-end result.

## Correctness checks

Before a long run:

- run a one-step eager smoke test;
- confirm finite distillation loss and gradient norm;
- check that `training/global_step` advances exactly once per optimizer step;
- verify that student and teacher tokenizers use the same vocabulary and chat
  template for top-k distillation;
- for Qwen3-VL, inspect several decoded samples to confirm that images are
  loaded and visual tokens are present;
- compare a short fixed batch before and after any performance-only change.

The Megatron `forward_kl_topk` implementation is vocabulary-parallel. Its loss
and gradients should match the full-vocabulary reference under TP before an
environment is accepted for training.

## Known limitations

- The Qwen3-VL mbridge implementation is still marked experimental upstream.
  Pin and test the dependency instead of tracking its moving main branch.
- The recipes cover TP only. Pipeline and expert parallelism are not part of
  this configuration.
- A two-NPU one-step smoke validates connectivity but does not represent the
  four-NPU steady-state performance target.
- Startup time includes Ray worker creation, model conversion, and optional NPU
  graph capture. Exclude it from steady-state throughput, but record it
  separately when operational startup time matters.
- Version strings alone do not guarantee compatibility: some vendor images
  carry locally patched builds. Record package versions and dependency commit
  hashes with every benchmark.
