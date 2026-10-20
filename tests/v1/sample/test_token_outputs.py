# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The token-output channel's worker side (`vllm/v1/token_outputs.py`)."""

from types import SimpleNamespace

import numpy as np
import torch

from vllm.v1.token_outputs import (
    AUX_READOUTS,
    LAST_HIDDEN_STATES,
    TokenOutputs,
    allowed,
    requested,
)


def _batch(num_logits: list[int], logits_indices: list[int]):
    cu = np.concatenate([[0], np.cumsum(num_logits)]).astype(np.int32)
    return SimpleNamespace(
        req_ids=[f"r{i}" for i in range(len(num_logits))],
        cu_num_logits_np=cu,
        logits_indices=torch.tensor(logits_indices),
    )


def _asked(result):
    """The gathered outputs of a step in which a request asked for something."""
    assert result is not None
    return result


def _params(asks: bool, readouts: list[str] | None = None):
    return SimpleNamespace(return_last_hidden_states=asks, aux_readouts=readouts)


def _config(hidden: bool, readouts: bool):
    return SimpleNamespace(
        enable_return_last_hidden_states=hidden, enable_aux_readouts=readouts
    )


def test_entries_follow_the_engine_flags_and_the_request():
    assert allowed(_config(False, False)) == []
    assert allowed(_config(True, False)) == [LAST_HIDDEN_STATES]
    assert allowed(_config(False, True)) == [AUX_READOUTS]
    assert allowed(_config(True, True)) == [LAST_HIDDEN_STATES, AUX_READOUTS]
    assert requested(_params(False)) == []
    assert requested(_params(True)) == [LAST_HIDDEN_STATES]
    assert requested(_params(False, ["decision"])) == [AUX_READOUTS]
    assert requested(_params(True, ["decision"])) == [LAST_HIDDEN_STATES, AUX_READOUTS]
    assert requested(_params(False, None)) == [] and requested(_params(False, [])) == []


def test_a_request_gets_only_the_entries_it_asked_for():
    hidden = torch.arange(12, dtype=torch.float32).view(6, 2)
    batch = _batch([1, 1], [1, 4])
    outputs = TokenOutputs([LAST_HIDDEN_STATES, AUX_READOUTS])
    outputs.add_request("r0", _params(True))
    outputs.add_request("r1", _params(False, ["decision"]))
    out = _asked(outputs.gather(hidden, batch, {AUX_READOUTS: lambda rows: rows * 2}))
    assert list(out[LAST_HIDDEN_STATES]) == ["r0"]
    assert list(out[AUX_READOUTS]) == ["r1"]
    assert torch.equal(out[AUX_READOUTS]["r1"], hidden[[4]] * 2)
    assert outputs.num_requests() == 2
    outputs.remove_request("r1")
    assert outputs.num_requests() == 1


def test_gather_returns_the_rows_compute_logits_reads_for_the_requests_that_asked():
    hidden = torch.arange(10 * 4, dtype=torch.float32).view(10, 4)
    # r0 asked (1 row, position 3), r1 did not (position 5), r2 asked (2 rows).
    batch = _batch([1, 1, 2], [3, 5, 7, 9])
    outputs = TokenOutputs([LAST_HIDDEN_STATES])
    outputs.add_request("r0", _params(True))
    outputs.add_request("r1", _params(False))
    outputs.add_request("r2", _params(True))

    out = _asked(outputs.gather(hidden, batch))

    assert list(out) == [LAST_HIDDEN_STATES]
    rows = out[LAST_HIDDEN_STATES]
    assert list(rows) == ["r0", "r2"]
    assert torch.equal(rows["r0"], hidden[[3]])
    assert torch.equal(rows["r2"], hidden[[7, 9]])
    assert rows["r0"].dtype == hidden.dtype


def test_nothing_is_gathered_when_no_scheduled_request_asked():
    hidden = torch.zeros(4, 2)
    batch = _batch([1, 1], [1, 3])
    outputs = TokenOutputs([LAST_HIDDEN_STATES])
    assert outputs.gather(hidden, batch) is None
    outputs.add_request("r9", _params(True))  # asked, but not scheduled in this step
    assert outputs.gather(hidden, batch) is None
    outputs.add_request("r0", _params(True))
    assert outputs.gather(hidden, batch) is not None
    outputs.remove_request("r0")
    assert outputs.gather(hidden, batch) is None


def test_a_request_with_no_logits_rows_gets_no_entry():
    hidden = torch.arange(8, dtype=torch.float32).view(4, 2)
    batch = _batch([0, 1], [2])
    outputs = TokenOutputs([LAST_HIDDEN_STATES])
    outputs.add_request("r0", _params(True))
    outputs.add_request("r1", _params(True))
    rows = _asked(outputs.gather(hidden, batch))[LAST_HIDDEN_STATES]
    assert list(rows) == ["r1"] and torch.equal(rows["r1"], hidden[[2]])


def test_a_compute_function_is_applied_once_to_all_the_asking_rows():
    hidden = torch.arange(12, dtype=torch.float32).view(6, 2)
    batch = _batch([1, 2], [1, 4, 5])
    seen = []

    def compute(rows: torch.Tensor) -> torch.Tensor:
        seen.append(rows.shape)
        return rows.sum(dim=1, keepdim=True)

    outputs = TokenOutputs([LAST_HIDDEN_STATES])
    outputs.add_request("r0", _params(True))
    outputs.add_request("r1", _params(True))
    rows = _asked(outputs.gather(hidden, batch, {LAST_HIDDEN_STATES: compute}))[
        LAST_HIDDEN_STATES
    ]
    assert seen == [(3, 2)]
    assert torch.equal(rows["r0"], hidden[[1]].sum(dim=1, keepdim=True))
    assert torch.equal(rows["r1"], hidden[[4, 5]].sum(dim=1, keepdim=True))
