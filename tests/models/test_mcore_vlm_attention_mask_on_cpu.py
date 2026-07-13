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

import torch

from verl.models.mcore import util


def test_build_vlm_attn_mask_thd_keeps_valid_token_mask_on_npu(monkeypatch):
    """VLM bridges need the 2D mask to prepare MRoPE and packed sequences."""
    monkeypatch.setattr(util, "is_npu_available", True)
    input_ids = torch.nested.nested_tensor(
        [torch.tensor([11, 12, 13]), torch.tensor([21, 22])],
        layout=torch.jagged,
    )

    padded_input_ids, attention_mask = util.build_vlm_attn_mask_thd(input_ids, pad_token_id=0)

    torch.testing.assert_close(padded_input_ids, torch.tensor([[11, 12, 13], [21, 22, 0]]))
    torch.testing.assert_close(
        attention_mask,
        torch.tensor([[True, True, True], [True, True, False]]),
    )
