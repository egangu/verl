# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from verl.workers.engine_workers import ActorRolloutRefWorker


class _FakeActorEngine:
    is_param_offload_enabled = False

    def get_per_tensor_param(self, **_kwargs):
        return [("weight", object())], None


class _FakeRollout:
    def __init__(self, events):
        self.events = events

    async def resume(self, tags):
        self.events.append(f"resume:{tags[0]}")

    async def update_weights(self, *_args, **_kwargs):
        self.events.append("update_weights")


def test_update_weights_releases_cached_pages_before_rollout_resume():
    events = []
    worker = ActorRolloutRefWorker.__new__(ActorRolloutRefWorker)
    worker.config = SimpleNamespace(
        rollout=SimpleNamespace(
            checkpoint_engine=SimpleNamespace(backend="naive"),
            free_cache_engine=True,
        )
    )
    worker.actor = SimpleNamespace(engine=_FakeActorEngine())
    worker.rollout = _FakeRollout(events)
    worker.layered_summon = False
    worker.peft_merge = True
    worker.base_sync_done = True

    with (
        patch(
            "verl.workers.engine_workers.aggressive_empty_cache",
            side_effect=lambda **_kwargs: events.append("cleanup"),
        ),
        patch("verl.workers.engine_workers.log_gpu_memory_usage"),
        patch("verl.workers.engine_workers.set_expandable_segments"),
    ):
        asyncio.run(ActorRolloutRefWorker.update_weights.__wrapped__(worker, global_steps=1))

    assert events == [
        "cleanup",
        "resume:weights",
        "update_weights",
        "cleanup",
        "resume:kv_cache",
    ]


def test_update_weights_does_not_add_pre_wake_cleanup_without_cache_sleep():
    events = []
    worker = ActorRolloutRefWorker.__new__(ActorRolloutRefWorker)
    worker.config = SimpleNamespace(
        rollout=SimpleNamespace(
            checkpoint_engine=SimpleNamespace(backend="naive"),
            free_cache_engine=False,
        )
    )
    worker.actor = SimpleNamespace(engine=_FakeActorEngine())
    worker.rollout = _FakeRollout(events)
    worker.layered_summon = False
    worker.peft_merge = True
    worker.base_sync_done = True

    with (
        patch(
            "verl.workers.engine_workers.aggressive_empty_cache",
            side_effect=lambda **_kwargs: events.append("cleanup"),
        ),
        patch("verl.workers.engine_workers.log_gpu_memory_usage"),
        patch("verl.workers.engine_workers.set_expandable_segments"),
    ):
        asyncio.run(ActorRolloutRefWorker.update_weights.__wrapped__(worker, global_steps=1))

    assert events == ["update_weights", "cleanup"]
