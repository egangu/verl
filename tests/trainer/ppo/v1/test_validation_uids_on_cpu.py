# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import numpy as np
import pytest

from verl.trainer.ppo.v1.trainer_base import build_validation_uids


def test_build_validation_uids_from_stable_dataset_field():
    batch = {"raw_prompt": np.array(["a", "b"], dtype=object), "index": np.array([7, 11], dtype=object)}

    first = build_validation_uids(batch, uid_key="index")
    second = build_validation_uids(batch, uid_key="index")

    assert first.tolist() == ["index:7", "index:11"]
    assert second.tolist() == first.tolist()


def test_build_validation_uids_random_by_default():
    batch = {"raw_prompt": np.array(["a", "b"], dtype=object)}

    first = build_validation_uids(batch)
    second = build_validation_uids(batch)

    assert len(set(first)) == 2
    assert first.tolist() != second.tolist()


def test_build_validation_uids_rejects_missing_field():
    batch = {"raw_prompt": np.array(["a"], dtype=object)}

    with pytest.raises(KeyError, match="sample_id"):
        build_validation_uids(batch, uid_key="sample_id")
