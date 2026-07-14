# On-policy distillation with Megatron and vLLM Ascend

Last updated: 07/14/2026.

This guide covers full-parameter on-policy distillation (OPD) on Ascend NPUs
with Megatron training and vLLM Ascend inference. The canonical recipes are:

- `run_qwen2_5_0_5b_megatron.sh`: Qwen2.5-0.5B student and
  Qwen2.5-3B-Instruct teacher on GSM8K.
- `run_qwen3_vl_4b_megatron.sh`: Qwen3-VL-4B-Instruct student and
  Qwen3-VL-8B-Instruct teacher on Geo3K.

The recipes use tensor parallelism only. Their default single-node placement is
one NPU for the colocated student/rollout pool and one NPU for the teacher
pool. Both pools use TP=1 and PP=1. A four-NPU TP=2+2 placement is documented
as a supplementary topology below.

## Software compatibility

Start from an Ascend image whose CANN, PyTorch, torch-npu, vLLM, and
vLLM Ascend versions are mutually compatible. Do not install a CUDA-focused
verl image on top of an Ascend runtime.

Load both the CANN and NNAL/ATB runtime environments before starting Ray or
vLLM Ascend. Loading CANN alone is insufficient: the vLLM worker fails while
registering its ATB extensions when `libatb.so` is not on the library path.
For the standard Ascend installation layout, run:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
```

Launch wrappers that enable Bash `nounset` (`set -u`) may need to disable it
temporarily while sourcing vendor environment scripts, then enable it again.
Some NNAL releases also append ATB example and test directories to
`LD_LIBRARY_PATH`. If a Ray actor exits before processing its first request
with an `ld.so` TLS-allocation assertion, keep the variables exported by the
NNAL script but restrict the ATB library entry to its runtime directory:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cann_ld_library_path=$LD_LIBRARY_PATH
source /usr/local/Ascend/nnal/atb/set_env.sh
export LD_LIBRARY_PATH="$ATB_HOME_PATH/lib:$cann_ld_library_path"
```

Verify the runtime before allocating NPUs:

```bash
python - <<'PY'
import ctypes

ctypes.CDLL("libatb.so")
from torch_npu.op_plugin.atb._atb_ops import _register_atb_extensions

_register_atb_extensions()
print("ATB runtime is ready")
PY
```

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

Prepare Geo3K with the repository utility:

```bash
python examples/data_preprocess/geo3k.py \
  --local_save_dir "$HOME/data/geo3k"
```

The resulting parquet files must contain the `images` column. Images may be
embedded in the parquet as bytes; if a dataset stores paths instead, those
paths must be visible from every Ray worker.

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
run_qwen2_5_0_5b_megatron.sh
```

The recipe defaults use two NPUs (one for student/rollout and one for the
teacher, TP=1 in each pool). The validated GSM8K defaults are batch 12, prompt
256, and response 1024, so the basic launch above needs no topology overrides.
To run the supplementary four-NPU topology, assign TP=2 to both pools:

```bash
NGPUS_PER_NODE=2 \
TEACHER_NGPUS_PER_NODE=2 \
ACTOR_TP=2 \
ROLLOUT_TP=2 \
TEACHER_TP=2 \
bash examples/on_policy_distillation_trainer/\
run_qwen2_5_0_5b_megatron.sh
```

Treat these batch and sequence values as a safe starting point, not a portable
performance claim. Re-tune them for the model pair, task-length distribution,
and HBM capacity. A short probe can miss a later long-tail batch, so preserve
memory headroom before committing to a long run.

The vision-language recipe uses the official `k1` definition as a detached
policy-gradient advantage:

```bash
ray stop --force
bash examples/on_policy_distillation_trainer/\
run_qwen3_vl_4b_megatron.sh
```

Qwen3-VL full-parameter training uses Megatron's CPU optimizer by default when
the student pool has one 64 GiB NPU. The default two-NPU topology assigns one
NPU to the student/rollout pool and one to the teacher pool. The validated
defaults use batch 12, prompt/response limits 1024/1024, and a 4096-token
dynamic microbatch budget. Rollout memory utilization defaults to 0.20 because
0.25 exhausted physical HBM while waking rollout weights after several
optimizer steps on the validated two-NPU setup. A ten-step probe at 0.20
completed nine sleep/wake transitions. Treat these values as starting points
for short-form multimodal reasoning, not portable defaults for every dataset.

With colocated Megatron training and vLLM rollout, the trainer returns cached
allocator pages immediately before vLLM restores sleeping weights. This
handoff is required even when the trainer has no live tensors in those cached
pages: otherwise the rollout process can fail its physical-memory allocation
after an optimizer step. A probe must include the rollout wake-up for the next
step; ending exactly after an optimizer step does not exercise this boundary.

`OPTIMIZER_CPU_OFFLOAD=true` selects Megatron's CPU-resident optimizer and is
the recipe default. It uses a full offload fraction with the precision-aware
optimizer. This still updates every model parameter; it changes optimizer-state
placement, not the training scope. Override `OPTIMIZER_OFFLOAD_FRACTION` or
`USE_PRECISION_AWARE_OPTIMIZER` only after validating the resulting optimizer
state and memory use.

The processor-expanded Geo3K prompts in the validated dataset reached 771
tokens in train and 613 in test, so the 1024 prompt limit preserves every
sample. Size multimodal prompts after image/video token expansion rather than
from raw text alone. Unlike response clipping, truncating a multimodal prompt
can break placeholder-to-feature alignment and should be treated as a
configuration error.

For Geo3K, a deterministic 601-sample checkpoint comparison found that a
512-token response limit allowed later checkpoints to drift below the initial
student, while changing only the response limit to 1024 recovered the loss.
The 1024-token recipe default is therefore a quality setting, not a requirement
of the OPD loss. A 512-token limit remains a valid memory-constrained fallback,
but compare fixed checkpoint evaluations before adopting it for a long run.

For the supplementary four-NPU topology, set both pools and all three TP values
to two, as in the Qwen2.5 example above. Re-tune batch size, token budget, and
CPU optimizer placement instead of assuming the two-NPU settings are optimal.

This optimizer setting is different from
`actor_rollout_ref.actor.megatron.optimizer_offload`. The latter moves
already-created optimizer state between worker stages, but it cannot prevent
the first optimizer step from allocating Adam state on the NPU. Use the CPU
optimizer when that initial allocation is the memory bottleneck.

Both recipes run 100 optimizer steps and disable periodic validation by default
so the reported step time covers training only. Set `TEST_FREQ` to a positive
value when validation is required. Override paths and batch sizes with
environment variables rather than editing the scripts:

```bash
STUDENT_MODEL=/models/student \
TEACHER_MODEL=/models/teacher \
TRAIN_FILE=/data/train.parquet \
VAL_FILE=/data/test.parquet \
TRAIN_BATCH_SIZE=64 \
PPO_MINI_BATCH_SIZE=64 \
bash examples/on_policy_distillation_trainer/\
run_qwen2_5_0_5b_megatron.sh
```

Choose `MAX_RESPONSE_LENGTH` for the task. The response clip ratio is a
diagnostic, not a universal quality or acceptance threshold: OPD still
supplies token-level supervision on the retained response prefix, and short
mathematical tasks can often use a shorter limit than open-ended generation.
Record the ratio and inspect clipped samples. A base student that repeats until
the limit is different from a dataset whose valid answers genuinely require
more context. Compare fixed downstream evaluations when changing the limit;
use a shorter setting for memory headroom and throughput only when it does not
degrade the selected checkpoint.

For long runs, launch in a persistent terminal such as tmux and keep Ray's
temporary directory, model caches, checkpoints, and logs on persistent storage.

## Tuning order

Tune one dimension at a time and retain at least ten steady-state steps after
warmup before comparing throughput.

1. Confirm that the actor, rollout, and teacher pools use the selected NPU
   topology and that no worker is placed on the wrong pool.
2. Increase `TRAIN_BATCH_SIZE` until generation is continuously batched.
3. Increase `PPO_MINI_BATCH_SIZE` while keeping it no larger than the train
   batch size.
4. Raise `PPO_MAX_TOKEN_LEN_PER_GPU` until memory is well utilized, then reduce
   it if long-tail batches cause out-of-memory failures. This budget controls
   dynamic actor and log-probability microbatching independently of the global
   train batch size, so reducing it can remove an activation or logits peak
   without reducing rollout concurrency.
5. Tune rollout and teacher memory utilization independently. Leave headroom
   for weight synchronization and multimodal preprocessing. Test enough steps
   to exercise repeated rollout sleep/wake transitions; a probe that ends
   immediately after an optimizer step can miss the next weight wake-up.
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
report vLLM decode-only throughput as the end-to-end result. The trainer's
`perf/throughput` metric is normalized per accelerator, so calculate the ratio
above explicitly when reporting global throughput.

## Correctness checks

Before a long run:

- run a one-step eager smoke test;
- confirm finite distillation loss and gradient norm;
- check that `training/global_step` advances exactly once per optimizer step;
- verify that student and teacher tokenizers map the same token IDs to the same
  vocabulary entries; different chat templates are acceptable because the
  teacher scores the student-rendered token sequence directly;
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
- Two-NPU TP=1 and four-NPU TP=2 use different model-parallel topologies;
  validate both independently instead of extrapolating throughput between
  them.
- Startup time includes Ray worker creation, model conversion, and optional NPU
  graph capture. Exclude it from steady-state throughput, but record it
  separately when operational startup time matters.
- Version strings alone do not guarantee compatibility: some vendor images
  carry locally patched builds. Record package versions and dependency commit
  hashes with every benchmark.
