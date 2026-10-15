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
pitches equal the row size, row by row when rows are at least
XgmiBackendConfig.strided_2d_max_row bytes, and with one hipMemcpy2DAsync otherwise.
Every test compares the whole destination buffer against the same copies done on the
host, so bytes written outside the requested rows are caught too.
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
    XgmiBackendConfig,
    set_log_level,
)

pytestmark = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="requires at least 2 GPUs"
)

SRC_GPU = 0
DST_GPU = 1
KiB = 1024


def _make_engine(key, **xgmi_kwargs):
    set_log_level("info")
    engine = IOEngine(key=key, config=IOEngineConfig(host="", port=0))
    engine.create_backend(
        BackendType.XGMI, XgmiBackendConfig(num_streams=8, num_events=8, **xgmi_kwargs)
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
    """A tensor-parallel column shard: one half of every 8 KiB row, packed on arrival
    (rows below strided_2d_max_row, so one hipMemcpy2DAsync)."""
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


def test_long_rows_row_by_row(xgmi_engine):
    """Rows of at least strided_2d_max_row (64 KiB default): grouped blocks such as one
    query block per attention group, copied row by row."""
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
            (m, 0, 128 * KiB, 4, 160 * KiB, 128 * KiB),  # long rows, row by row
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
    """Source and destination on one GPU take the hipMemcpyAsync path."""
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


@pytest.mark.parametrize("max_row", [0, 1 << 40])
def test_strided_2d_max_row(max_row):
    """The threshold only picks the copy method: every row by row (0) or every strided
    entry as one 2D copy (huge) gives the same bytes."""
    engine = _make_engine(
        f"strided_engine_max_row_{max_row}", strided_2d_max_row=max_row
    )
    m = 1 << 20
    src, dst, src_mem, dst_mem = _alloc(engine, SRC_GPU, DST_GPU, 8 * m, 8 * m)
    _check(
        engine,
        src,
        dst,
        src_mem,
        dst_mem,
        [
            (0, 0, 4 * KiB, 64, 4 * KiB, 8 * KiB),
            (m, 2 * m, 128 * KiB, 8, 128 * KiB, 160 * KiB),
        ],
    )


def test_config_field():
    assert XgmiBackendConfig().strided_2d_max_row == 64 * KiB
    c = XgmiBackendConfig(strided_2d_max_row=4 * KiB)
    assert c.strided_2d_max_row == 4 * KiB
    c.strided_2d_max_row = 1
    assert c.strided_2d_max_row == 1


# ---------------------------------------------------------------------------
# Cross-process: the target's memory is opened over IPC, as between a trainer and
# an inference process.
# ---------------------------------------------------------------------------

XP_SRC_BYTES = 8 << 20
XP_DST_BYTES = 8 << 20
XP_ENTRIES = [
    (0, 4 * KiB, 4 * KiB, 512, 4 * KiB, 8 * KiB),  # 2D
    (2 << 20, 4 << 20, 192 * KiB, 4, 192 * KiB, 256 * KiB),  # row by row
    (4 << 20, 1 << 20, 1 << 20, 1, 1 << 20, 1 << 20),  # one row
]


def _exchange(obj, rank):
    out = [None, None]
    dist.all_gather_object(out, obj)
    return out[1 - rank]


def _worker(rank, master_port, result_queue):
    """rank 0 reads (GPU 0); rank 1 holds the source (GPU 1)."""
    try:
        with TorchDistContext(
            rank=rank,
            world_size=2,
            master_addr="localhost",
            master_port=str(master_port),
            device_id=rank,
            backend="gloo",
        ):
            engine = _make_engine(f"strided_xp_{rank}")
            engine.register_remote_engine(
                EngineDesc.unpack(_exchange(engine.get_engine_desc().pack(), rank))
            )
            g = torch.Generator().manual_seed(7)
            src = torch.randint(1, 256, (XP_SRC_BYTES,), dtype=torch.uint8, generator=g)
            if rank == 1:
                buf = src.to(torch.device("cuda", rank))
            else:
                buf = torch.zeros(
                    XP_DST_BYTES, dtype=torch.uint8, device=torch.device("cuda", rank)
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


def test_cross_process():
    master_port = get_free_port()
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    procs = [
        ctx.Process(target=_worker, args=(rank, master_port, result_queue))
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
