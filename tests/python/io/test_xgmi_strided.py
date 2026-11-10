# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Tests for IOEngine.batch_read_strided on the XGMI backend.

Entry j of a strided read is rows[j] rows of sizes[j] bytes; row r is read from
remote_offsets[j] + r * remote_pitches[j] into local_offsets[j] + r * local_pitches[j].
The XGMI backend copies an entry as one contiguous block when it has one row or both
pitches equal the row size, and otherwise as a 2D copy on one of ROCr's SDMA blit queues
(hsa_amd_memory_async_copy_rect), merging entries of one shape spaced evenly on both sides
into one 3D copy, and falling back to hipMemcpy2DAsync for what ROCr can't take (unaligned
addresses or pitches, pitches too large for its packets, one GPU). An entry that merges with
no other and has rows over 1 MiB goes row by row as 1D copies on the pair's link engine. Every
test compares the whole destination buffer against the same copies done on the host, so
bytes written outside the requested rows are caught too.
"""

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tests.python.utils import TorchDistContext, get_free_port
from mori.io import (
    IOEngineConfig,
    BackendType,
    IOEngine,
    EngineDesc,
    MemoryDesc,
    StatusCode,
    XgmiBackendConfig,
    set_log_level,
)

pytestmark = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="requires at least 2 GPUs"
)

SRC_GPU = 0
DST_GPU = 1
KiB = 1024


def _make_engine(key):
    set_log_level("info")
    engine = IOEngine(key=key, config=IOEngineConfig(host="", port=0))
    engine.create_backend(
        BackendType.XGMI, XgmiBackendConfig(num_streams=8, num_events=8)
    )
    return engine


@pytest.fixture(scope="module")
def xgmi_engine():
    yield _make_engine("strided_engine")


def _alloc(engine, src_gpu, dst_gpu, src_bytes, dst_bytes, seed=0):
    """A source of nonzero random bytes and a zeroed destination, both registered."""
    g = torch.Generator().manual_seed(seed)
    src = torch.randint(1, 256, (src_bytes,), dtype=torch.uint8, generator=g).to(
        torch.device("cuda", src_gpu)
    )
    dst = torch.zeros(
        dst_bytes, dtype=torch.uint8, device=torch.device("cuda", dst_gpu)
    )
    torch.cuda.synchronize(src_gpu)
    return (
        src,
        dst,
        engine.register_torch_tensor(src),
        engine.register_torch_tensor(dst),
    )


def _expected(src, dst_bytes, entries):
    """Apply (local_off, remote_off, size, rows, local_pitch, remote_pitch) on the host."""
    s = src.cpu()
    out = torch.zeros(dst_bytes, dtype=torch.uint8)
    for lo, ro, size, rows, lp, rp in entries:
        for r in range(rows):
            out[lo + r * lp : lo + r * lp + size] = s[ro + r * rp : ro + r * rp + size]
    return out


def _read(engine, dst_mem, src_mem, entries):
    """One batch_read_strided of `entries` from src_mem into dst_mem; returns the status."""
    cols = list(zip(*entries)) if entries else [()] * 6
    lo, ro, sz, nr, lp, rp = ([list(c)] for c in cols)
    uid = engine.allocate_transfer_uid()
    (status,) = engine.batch_read_strided(
        [dst_mem], lo, [src_mem], ro, sz, nr, lp, rp, [uid]
    )
    status.Wait()
    return status


def _check(engine, src, dst, src_mem, dst_mem, entries):
    status = _read(engine, dst_mem, src_mem, entries)
    assert status.Succeeded(), f"batch_read_strided failed: {status.Message()}"
    torch.cuda.synchronize(dst.device)
    want = _expected(src, dst.numel(), entries)
    got = dst.cpu()
    bad = (got != want).nonzero()
    assert bad.numel() == 0, (
        f"{bad.numel()} bytes differ, first at offset {int(bad[0])}: "
        f"got {int(got[bad[0]])}, want {int(want[bad[0]])}"
    )


# ---------------------------------------------------------------------------
# Single-process, GPU 0 -> GPU 1
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("half", [0, 1])
def test_short_rows_column_half(xgmi_engine, half):
    """A tensor-parallel column shard: one half of every 8 KiB row, packed on arrival."""
    rows, row = 512, 4 * KiB
    src, dst, src_mem, dst_mem = _alloc(
        xgmi_engine, SRC_GPU, DST_GPU, rows * 2 * row, rows * row, seed=half
    )
    _check(
        xgmi_engine,
        src,
        dst,
        src_mem,
        dst_mem,
        [(0, half * row, row, rows, row, 2 * row)],
    )


def test_short_rows_scatter(xgmi_engine):
    """Packed source rows spread out on the destination (local pitch > row size)."""
    rows, row, pitch = 300, 1 * KiB + 256, 3 * KiB
    src, dst, src_mem, dst_mem = _alloc(
        xgmi_engine, SRC_GPU, DST_GPU, rows * row, rows * pitch + 512
    )
    _check(xgmi_engine, src, dst, src_mem, dst_mem, [(512, 0, row, rows, pitch, row)])


def test_long_rows(xgmi_engine):
    """Few long rows: grouped blocks such as one query block per attention group."""
    rows, row, src_pitch = 8, 256 * KiB, 320 * KiB
    src, dst, src_mem, dst_mem = _alloc(
        xgmi_engine, SRC_GPU, DST_GPU, rows * src_pitch, rows * row
    )
    _check(
        xgmi_engine,
        src,
        dst,
        src_mem,
        dst_mem,
        [(0, 64 * KiB, row, rows, row, src_pitch)],
    )


def test_contiguous_forms(xgmi_engine):
    """One row, and many rows whose pitches equal the row size, are plain copies."""
    n = 1 << 20
    src, dst, src_mem, dst_mem = _alloc(xgmi_engine, SRC_GPU, DST_GPU, 4 * n, 4 * n)
    _check(
        xgmi_engine,
        src,
        dst,
        src_mem,
        dst_mem,
        [
            (0, n, 3 * KiB, 1, 7, 13),  # one row: the pitches are ignored
            (n, 2 * n, 4 * KiB, 64, 4 * KiB, 4 * KiB),
        ],
    )


def test_mixed_batch(xgmi_engine):
    """Every kind of entry in one call, plus empty ones that must copy nothing."""
    m = 1 << 20
    src, dst, src_mem, dst_mem = _alloc(xgmi_engine, SRC_GPU, DST_GPU, 16 * m, 16 * m)
    _check(
        xgmi_engine,
        src,
        dst,
        src_mem,
        dst_mem,
        [
            (0, 3 * m, 2 * KiB, 128, 2 * KiB, 4 * KiB),  # short rows, 2D
            (m, 0, 128 * KiB, 4, 160 * KiB, 128 * KiB),  # long rows
            (2 * m, 5 * m, 64 * KiB, 1, 0, 0),  # one row
            (3 * m, 6 * m, 8 * KiB, 16, 8 * KiB, 8 * KiB),  # packed both sides
            (4 * m, 7 * m, 0, 10, 1 * KiB, 1 * KiB),  # zero bytes per row
            (5 * m, 8 * m, 1 * KiB, 0, 1 * KiB, 1 * KiB),  # zero rows
            (6 * m, 9 * m, 4 * KiB - 4, 33, 4 * KiB, 12 * KiB),  # odd row size, 2D
        ],
    )


def test_rows_out_of_order(xgmi_engine):
    """Entries listed in descending source order land at their own offsets."""
    n, row = 32, 16 * KiB
    src, dst, src_mem, dst_mem = _alloc(xgmi_engine, SRC_GPU, DST_GPU, n * row, n * row)
    _check(
        xgmi_engine,
        src,
        dst,
        src_mem,
        dst_mem,
        [(i * row, (n - 1 - i) * row, row, 1, row, row) for i in range(n)],
    )


def test_empty_batch(xgmi_engine):
    """No entries succeeds and leaves the destination untouched."""
    src, dst, src_mem, dst_mem = _alloc(xgmi_engine, SRC_GPU, DST_GPU, 4 * KiB, 4 * KiB)
    _check(xgmi_engine, src, dst, src_mem, dst_mem, [])


def test_same_gpu(xgmi_engine):
    """Source and destination on one GPU: hipMemcpyAsync, and hipMemcpy2DAsync for strided
    entries."""
    src, dst, src_mem, dst_mem = _alloc(xgmi_engine, SRC_GPU, SRC_GPU, 4 << 20, 4 << 20)
    _check(
        xgmi_engine,
        src,
        dst,
        src_mem,
        dst_mem,
        [
            (0, 0, 4 * KiB, 256, 4 * KiB, 8 * KiB),
            (1 << 20, 2 << 20, 96 * KiB, 4, 96 * KiB, 128 * KiB),
        ],
    )


def test_two_batches_one_call(xgmi_engine):
    """Two (destination, source) pairs in one engine call, each with its own status."""
    a = _alloc(xgmi_engine, SRC_GPU, DST_GPU, 1 << 20, 1 << 20, seed=1)
    b = _alloc(xgmi_engine, DST_GPU, SRC_GPU, 1 << 20, 1 << 20, seed=2)
    ea = [(0, 4 * KiB, 4 * KiB, 64, 4 * KiB, 8 * KiB)]
    eb = [(8 * KiB, 0, 100 * KiB, 3, 100 * KiB, 200 * KiB)]
    uids = [xgmi_engine.allocate_transfer_uid() for _ in range(2)]
    statuses = xgmi_engine.batch_read_strided(
        [a[3], b[3]],
        [[e[0] for e in ea], [e[0] for e in eb]],
        [a[2], b[2]],
        [[e[1] for e in ea], [e[1] for e in eb]],
        [[e[2] for e in ea], [e[2] for e in eb]],
        [[e[3] for e in ea], [e[3] for e in eb]],
        [[e[4] for e in ea], [e[4] for e in eb]],
        [[e[5] for e in ea], [e[5] for e in eb]],
        uids,
    )
    for (src, dst, _, _), entries, st in zip((a, b), (ea, eb), statuses):
        st.Wait()
        assert st.Succeeded(), st.Message()
        torch.cuda.synchronize(dst.device)
        assert torch.equal(dst.cpu(), _expected(src, dst.numel(), entries))


@pytest.mark.parametrize(
    "size,rows,local_pitch,remote_pitch",
    [
        (384, 2048, 384, 1536),  # expert down_proj shard
        (
            768 * KiB,
            2,
            768 * KiB,
            3 * 1024 * KiB,
        ),  # wider than one packet (16 Ki elements)
        (
            4 * 1024 * KiB,
            8,
            4 * 1024 * KiB,
            6 * 1024 * KiB,
        ),  # Q blocks of a fused QKV: rows over 1 MiB go 1D on the link engine
        (4 * KiB, 20000, 4 * KiB, 8 * KiB),  # more rows than one packet (16 Ki)
        (1, 100, 1, 4),  # 1-byte elements
        (3, 100, 3, 8),  # odd width, 4-byte pitch: 1-byte elements
        (386, 50, 386, 772),  # pitches not 4-byte aligned: hipMemcpy2DAsync
        (16, 4, 16, 16 << 20),  # pitch too large for 16-byte elements in the packet
        (4, 4, 4, 4 << 20),  # pitch too large at any element size: hipMemcpy2DAsync
    ],
)
def test_shapes(xgmi_engine, size, rows, local_pitch, remote_pitch):
    """SDMA packet tiling limits and element sizes in ROCr, and the fallback for what its
    packets can't express."""
    src, dst, src_mem, dst_mem = _alloc(
        xgmi_engine, SRC_GPU, DST_GPU, rows * remote_pitch, rows * local_pitch + 64
    )
    _check(
        xgmi_engine,
        src,
        dst,
        src_mem,
        dst_mem,
        [(64, 0, size, rows, local_pitch, remote_pitch)],
    )


def test_unaligned_offsets(xgmi_engine):
    """Entries starting off a 4-byte boundary fall back; aligned ones in the same call don't."""
    src, dst, src_mem, dst_mem = _alloc(xgmi_engine, SRC_GPU, DST_GPU, 1 << 20, 1 << 20)
    _check(
        xgmi_engine,
        src,
        dst,
        src_mem,
        dst_mem,
        [
            (1, 0, 256, 64, 256, 1024),
            (64 * KiB, 3, 256, 64, 256, 1024),
            (128 * KiB, 128 * KiB, 256, 64, 256, 1024),
        ],
    )


@pytest.mark.parametrize(
    "entry",
    [
        (0, 0, 4 * KiB, 4, 4 * KiB, 2 * KiB),  # remote pitch < row, SDMA-expressible
        (0, 0, 4 * KiB, 4, 2 * KiB, 8 * KiB),  # local pitch < row: rows would overlap
        (0, 0, 4098, 4, 4098, 2050),  # unaligned, would have gone to hipMemcpy2DAsync
    ],
)
def test_pitch_smaller_than_row_rejected(xgmi_engine, entry):
    """Whichever copy would have taken it, such an entry fails before anything is copied."""
    src, dst, src_mem, dst_mem = _alloc(xgmi_engine, SRC_GPU, DST_GPU, 1 << 20, 1 << 20)
    status = _read(xgmi_engine, dst_mem, src_mem, [entry])
    assert status.Code() == StatusCode.ERR_INVALID_ARGS, status.Message()
    assert "pitch smaller than row size" in status.Message()
    torch.cuda.synchronize(dst.device)
    assert not dst.cpu().any()


def test_mismatched_entry_lists_rejected(xgmi_engine):
    """Per-entry lists of different lengths fail that batch instead of reading past the end."""
    src, dst, src_mem, dst_mem = _alloc(xgmi_engine, SRC_GPU, DST_GPU, 1 << 20, 1 << 20)
    uid = xgmi_engine.allocate_transfer_uid()
    (status,) = xgmi_engine.batch_read_strided(
        [dst_mem],
        [[0, 8 * KiB]],
        [src_mem],
        [[0, 64 * KiB]],
        [[4 * KiB, 4 * KiB]],
        [[2]],
        [[4 * KiB, 4 * KiB]],
        [[8 * KiB, 8 * KiB]],
        [uid],
    )
    status.Wait()
    assert status.Code() == StatusCode.ERR_INVALID_ARGS, status.Message()
    torch.cuda.synchronize(dst.device)
    assert not dst.cpu().any()


def test_mismatched_batch_lists_raise(xgmi_engine):
    src, dst, src_mem, dst_mem = _alloc(xgmi_engine, SRC_GPU, DST_GPU, 4 * KiB, 4 * KiB)
    uid = xgmi_engine.allocate_transfer_uid()
    with pytest.raises(ValueError):
        xgmi_engine.batch_read_strided(
            [dst_mem, dst_mem],
            [[0]],
            [src_mem],
            [[0]],
            [[KiB]],
            [[2]],
            [[KiB]],
            [[KiB]],
            [uid],
        )


def _unmergeable(n, size, rows, pitch):
    """n entries whose destinations run backwards against their sources, so none merge."""
    return [
        ((n - 1 - i) * rows * size, i * rows * pitch, size, rows, size, pitch)
        for i in range(n)
    ]


def test_many_calls_in_flight(xgmi_engine):
    """Overlapping calls with one ROCr copy per entry, about 40,000 copies on one queue."""
    n, size, rows, pitch = 8192, 256, 2, 1 * KiB
    src, dst, src_mem, dst_mem = _alloc(
        xgmi_engine, SRC_GPU, DST_GPU, n * rows * pitch, n * rows * size
    )
    entries = _unmergeable(n, size, rows, pitch)
    for _ in range(5):
        pending = []
        for b in range(0, n, 62):
            chunk = entries[b : b + 62]
            lo, ro, sz, nr, lp, rp = ([list(c)] for c in zip(*chunk))
            uid = xgmi_engine.allocate_transfer_uid()
            (st,) = xgmi_engine.batch_read_strided(
                [dst_mem], lo, [src_mem], ro, sz, nr, lp, rp, [uid]
            )
            pending.append(st)
        for st in pending:
            st.Wait()
            assert st.Succeeded(), st.Message()
    torch.cuda.synchronize(dst.device)
    assert torch.equal(dst.cpu(), _expected(src, dst.numel(), entries))


def test_large_batch(xgmi_engine):
    """One call of 20,000 entries that cannot merge: 20,000 ROCr copies on one signal."""
    n, size, rows, pitch = 20000, 256, 2, 1 * KiB
    src, dst, src_mem, dst_mem = _alloc(
        xgmi_engine, SRC_GPU, DST_GPU, n * rows * pitch, n * rows * size
    )
    _check(xgmi_engine, src, dst, src_mem, dst_mem, _unmergeable(n, size, rows, pitch))


def _spaced(n, size, rows, lp, rp, local_step, remote_step, local_base=0, remote_base=0):
    return [
        (local_base + i * local_step, remote_base + i * remote_step, size, rows, lp, rp)
        for i in range(n)
    ]


@pytest.mark.parametrize(
    "entries",
    [
        # Back to back on both sides: one taller 2D copy.
        _spaced(64, 1 * KiB, 8, 1 * KiB, 4 * KiB, 8 * KiB, 32 * KiB),
        # Gaps on both sides (the same slice of each stacked expert): one 3D copy.
        _spaced(48, 384, 32, 384, 1536, 16 * KiB, 64 * KiB, 4 * KiB, 128),
        # Even locally, uneven remotely: runs break where the remote step changes.
        _spaced(8, 2 * KiB, 4, 2 * KiB, 8 * KiB, 8 * KiB, 32 * KiB)
        + _spaced(8, 2 * KiB, 4, 2 * KiB, 8 * KiB, 8 * KiB, 40 * KiB, 64 * KiB, 512 * KiB),
        # Two shapes interleaved in the input.
        [
            e
            for pair in zip(
                _spaced(16, 1 * KiB, 4, 1 * KiB, 2 * KiB, 4 * KiB, 8 * KiB),
                _spaced(16, 512, 8, 512, 2 * KiB, 4 * KiB, 16 * KiB, 64 * KiB, 256 * KiB),
            )
            for e in pair
        ],
        # Source blocks overlapping the next one's rows: must not merge.
        _spaced(16, 1 * KiB, 8, 1 * KiB, 2 * KiB, 8 * KiB, 8 * KiB),
        # Steps not 4-byte aligned: ROCr refuses, entries fall back one by one.
        _spaced(16, 256, 4, 256, 1 * KiB, 4 * KiB + 2, 8 * KiB + 2),
    ],
    ids=["adjacent", "gaps", "remote_step_changes", "two_shapes", "overlapping", "odd_step"],
)
def test_merged_entries(xgmi_engine, entries):
    """Entries merged into one ROCr copy land exactly where each entry says, shuffled or not."""
    import random

    src_bytes = max(ro + rows * rp for _, ro, _, rows, _, rp in entries)
    dst_bytes = max(lo + rows * lp for lo, _, _, rows, lp, _ in entries)
    src, dst, src_mem, dst_mem = _alloc(xgmi_engine, SRC_GPU, DST_GPU, src_bytes, dst_bytes)
    shuffled = list(entries)
    random.Random(3).shuffle(shuffled)
    _check(xgmi_engine, src, dst, src_mem, dst_mem, shuffled)


_SDMA_PROBE = """
from tests.python.io.test_xgmi_strided import _make_engine, _alloc, _check, SRC_GPU, DST_GPU
engine = _make_engine("strided_engine_probe")
src, dst, src_mem, dst_mem = _alloc(engine, SRC_GPU, DST_GPU, 512 * 8192, 512 * 4096)
_check(engine, src, dst, src_mem, dst_mem, [(0, 0, 4096, 512, 4096, 8192)])
"""


@pytest.mark.skipif(
    torch.cuda.device_count() < 2
    or not torch.cuda.get_device_properties(DST_GPU).gcnArchName.startswith("gfx9"),
    reason="SDMA rect copies need gfx9",
)
@pytest.mark.parametrize(
    "sdma_env,expected_log",
    [
        (None, "strided SDMA copies for"),
        ("0", "MORI_IO_XGMI_SDMA_RECT=0; strided copies use hipMemcpy2DAsync"),
    ],
)
def test_uses_sdma_queue(sdma_env, expected_log):
    """Strided entries reach the pair's SDMA queue rather than silently falling back, unless
    MORI_IO_XGMI_SDMA_RECT=0. The queue logs once, when the process creates it, hence a fresh
    process. The level comes from the environment because set_log_level does not reach the
    loggers inside libmori_io."""
    import os
    import subprocess
    import sys

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    env = {**os.environ, "MORI_IO_LOG_LEVEL": "info"}
    env.pop("MORI_IO_XGMI_SDMA_RECT", None)
    if sdma_env is not None:
        env["MORI_IO_XGMI_SDMA_RECT"] = sdma_env
    out = subprocess.run(
        [sys.executable, "-c", _SDMA_PROBE],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    log = out.stdout + out.stderr
    assert out.returncode == 0, log[-2000:]
    assert expected_log in log, log[-2000:]


# ---------------------------------------------------------------------------
# Cross-process: the target's memory is opened over IPC, as between a trainer and
# an inference process.
# ---------------------------------------------------------------------------

XP_SRC_BYTES = 8 << 20
XP_DST_BYTES = 8 << 20
XP_ENTRIES = [
    (0, 4 * KiB, 4 * KiB, 512, 4 * KiB, 8 * KiB),  # 2D
    (2 << 20, 4 << 20, 192 * KiB, 4, 192 * KiB, 256 * KiB),  # long rows
    (4 << 20, 1 << 20, 1 << 20, 1, 1 << 20, 1 << 20),  # one row
    (5 << 20, 0, 1536 * KiB, 2, 1536 * KiB, 2 << 20),  # rows over 1 MiB: 1D on the link engine
]


def _exchange(obj, rank):
    out = [None, None]
    dist.all_gather_object(out, obj)
    return out[1 - rank]


def _worker(rank, master_port, result_queue, gpus):
    """rank 0 reads on gpus[0]; rank 1 holds the source on gpus[1]."""
    try:
        with TorchDistContext(
            rank=rank,
            world_size=2,
            master_addr="localhost",
            master_port=str(master_port),
            device_id=gpus[rank],
            backend="gloo",
        ):
            engine = _make_engine(f"strided_xp_{rank}")
            engine.register_remote_engine(
                EngineDesc.unpack(_exchange(engine.get_engine_desc().pack(), rank))
            )
            g = torch.Generator().manual_seed(7)
            src = torch.randint(1, 256, (XP_SRC_BYTES,), dtype=torch.uint8, generator=g)
            if rank == 1:
                buf = src.to(torch.device("cuda", gpus[rank]))
            else:
                buf = torch.zeros(
                    XP_DST_BYTES, dtype=torch.uint8, device=torch.device("cuda", gpus[rank])
                )
            torch.cuda.synchronize()
            mem = engine.register_torch_tensor(buf)
            remote = MemoryDesc.unpack(_exchange(mem.pack(), rank))

            ok, detail = True, ""
            if rank == 0:
                status = _read(engine, mem, remote, XP_ENTRIES)
                torch.cuda.synchronize()
                if not status.Succeeded():
                    ok, detail = False, f"batch_read_strided failed: {status.Message()}"
                elif not torch.equal(
                    buf.cpu(), _expected(src, XP_DST_BYTES, XP_ENTRIES)
                ):
                    ok, detail = (
                        False,
                        "cross-process strided read returned wrong bytes",
                    )
            dist.barrier()
            result_queue.put(("PASS", "") if ok else ("FAIL", detail))
    except Exception as e:
        import traceback

        result_queue.put(("FAIL", f"{e}\n{traceback.format_exc()}"))


@pytest.mark.parametrize("src_gpu", [1, 2, 3])
def test_cross_process(src_gpu):
    """Over IPC, with sources whose pairs land on the reader's and on the source's ROCr
    queues."""
    if torch.cuda.device_count() <= src_gpu:
        pytest.skip(f"needs {src_gpu + 1} GPUs")
    master_port = get_free_port()
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    procs = [
        ctx.Process(target=_worker, args=(rank, master_port, result_queue, (0, src_gpu)))
        for rank in range(2)
    ]
    for p in procs:
        p.start()
    results = [result_queue.get(timeout=180) for _ in procs]
    for p in procs:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()
            p.join()
            pytest.fail(f"worker {p.pid} timed out")
    for status, msg in results:
        assert status == "PASS", f"worker failed: {msg}"
    for p in procs:
        assert p.exitcode == 0, f"worker exited with code {p.exitcode}"
