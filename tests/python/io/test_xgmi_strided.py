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

Each entry is rows rows of size bytes, local_pitch bytes apart locally and remote_pitch
bytes apart remotely. Strided entries go as 2D copies: between two GPUs as one
hipMemcpy3DBatchAsync per batch, within one GPU as hipMemcpy2DAsync. Entries with one row,
or with both pitches equal to the row size, go through batch_read's contiguous path on the
same stream. Every test compares the whole destination buffer against the same copies done
on the host, so bytes written outside the requested rows are caught too.

Blocks are written as (local_off, remote_off, size, rows, local_pitch, remote_pitch).
"""

import random

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
    # MORI copies on its own streams, which don't wait for the fill or the zeroing.
    torch.cuda.synchronize(src_gpu)
    torch.cuda.synchronize(dst_gpu)
    return (
        src,
        dst,
        engine.register_torch_tensor(src),
        engine.register_torch_tensor(dst),
    )


def _expected(src, dst_bytes, blocks):
    s = src.cpu()
    out = torch.zeros(dst_bytes, dtype=torch.uint8)
    for lo, ro, size, rows, lp, rp in blocks:
        for r in range(rows):
            out[lo + r * lp : lo + r * lp + size] = s[ro + r * rp : ro + r * rp + size]
    return out


def _columns(blocks):
    """The six per-entry lists of batch_read_strided for one batch."""
    return [list(c) for c in zip(*blocks)] if blocks else [[] for _ in range(6)]


def _read(engine, dst_mem, src_mem, blocks):
    """One batch_read_strided of `blocks` from src_mem into dst_mem."""
    lo, ro, sz, nr, lp, rp = _columns(blocks)
    (status,) = engine.batch_read_strided(
        [dst_mem], [lo], [src_mem], [ro], [sz], [nr], [lp], [rp],
        [engine.allocate_transfer_uid()],
    )
    status.Wait()
    return status


def _extent(blocks, side):
    """Bytes a buffer needs to hold every row of `blocks` on one side (0 local, 1 remote)."""
    return max(
        (b[side] + (b[3] - 1) * b[4 + side] + b[2] for b in blocks if b[2] and b[3]),
        default=0,
    )


def _check_blocks(engine, blocks, src_gpu=SRC_GPU, dst_gpu=DST_GPU, seed=0):
    src, dst, src_mem, dst_mem = _alloc(
        engine, src_gpu, dst_gpu, max(_extent(blocks, 1), 1), max(_extent(blocks, 0), 1), seed
    )
    status = _read(engine, dst_mem, src_mem, blocks)
    assert status.Succeeded(), f"transfer failed: {status.Message()}"
    torch.cuda.synchronize(dst.device)
    want = _expected(src, dst.numel(), blocks)
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
    _check_blocks(xgmi_engine, [(0, half * 4 * KiB, 4 * KiB, 512, 4 * KiB, 8 * KiB)])


def test_short_rows_scatter(xgmi_engine):
    """Packed source rows spread out on the destination (local pitch > row size)."""
    _check_blocks(xgmi_engine, [(512, 0, 1 * KiB + 256, 300, 3 * KiB, 1 * KiB + 256)])


def test_long_rows(xgmi_engine):
    """Few long rows: grouped blocks such as one query block per attention group."""
    _check_blocks(xgmi_engine, [(0, 64 * KiB, 256 * KiB, 8, 256 * KiB, 320 * KiB)])


def test_mixed_batch(xgmi_engine):
    """Strided blocks, contiguous entries and empty entries in one call, in any order."""
    m = 1 << 20
    blocks = [
        (0, 3 * m, 2 * KiB, 128, 2 * KiB, 4 * KiB),  # short rows
        (m, 0, 128 * KiB, 4, 160 * KiB, 128 * KiB),  # long rows, scattered
        (2 * m, 5 * m, 64 * KiB, 1, 0, 0),  # one row
        (3 * m, 6 * m, 8 * KiB, 16, 8 * KiB, 8 * KiB),  # contiguous both sides
        (6 * m, 9 * m, 4 * KiB - 4, 33, 4 * KiB, 12 * KiB),  # odd row size
        (4 * m, 7 * m, 0, 4, 4 * KiB, 4 * KiB),  # zero size
        (5 * m, 8 * m, 4 * KiB, 0, 4 * KiB, 8 * KiB),  # zero rows
    ]
    random.Random(1).shuffle(blocks)
    _check_blocks(xgmi_engine, blocks)


def test_contiguous_only(xgmi_engine):
    """Entries of mixed sizes at random offsets, none strided: batch_read's path alone."""
    rng = random.Random(5)
    blocks, d = [], 0
    for _ in range(500):
        size = rng.choice([256, 1 * KiB, 4 * KiB, 12 * KiB + 4])
        blocks.append((d, rng.randrange(0, 4 << 20, 4), size, 1, size, size))
        d += size + rng.choice([0, 4, 64, 4 * KiB])
    _check_blocks(xgmi_engine, blocks)


def test_empty_batch(xgmi_engine):
    """No entries succeeds and leaves the destination untouched."""
    _check_blocks(xgmi_engine, [])


def test_same_gpu(xgmi_engine):
    """Source and destination on one GPU stay on HIP's copies."""
    _check_blocks(
        xgmi_engine,
        [
            (0, 0, 4 * KiB, 256, 4 * KiB, 8 * KiB),
            (1 << 20, 2 << 20, 96 * KiB, 4, 96 * KiB, 128 * KiB),
            (2 << 20, 3 << 20, 64 * KiB, 1, 0, 0),
        ],
        dst_gpu=SRC_GPU,
    )


def test_two_batches_one_call(xgmi_engine):
    """Two (destination, source) pairs in one engine call, each with its own status."""
    a = _alloc(xgmi_engine, SRC_GPU, DST_GPU, 1 << 20, 1 << 20, seed=1)
    b = _alloc(xgmi_engine, DST_GPU, SRC_GPU, 1 << 20, 1 << 20, seed=2)
    ba = [(0, 4 * KiB, 4 * KiB, 64, 4 * KiB, 8 * KiB)]
    bb = [(8 * KiB, 0, 100 * KiB, 3, 100 * KiB, 200 * KiB), (512 * KiB, 0, 4 * KiB, 1, 0, 0)]
    (lo_a, ro_a, sz_a, nr_a, lp_a, rp_a), (lo_b, ro_b, sz_b, nr_b, lp_b, rp_b) = (
        _columns(ba),
        _columns(bb),
    )
    statuses = xgmi_engine.batch_read_strided(
        [a[3], b[3]],
        [lo_a, lo_b],
        [a[2], b[2]],
        [ro_a, ro_b],
        [sz_a, sz_b],
        [nr_a, nr_b],
        [lp_a, lp_b],
        [rp_a, rp_b],
        [xgmi_engine.allocate_transfer_uid() for _ in range(2)],
    )
    for (src, dst, _, _), blocks, st in zip((a, b), (ba, bb), statuses):
        st.Wait()
        assert st.Succeeded(), st.Message()
        torch.cuda.synchronize(dst.device)
        assert torch.equal(dst.cpu(), _expected(src, dst.numel(), blocks))


@pytest.mark.parametrize(
    "size,rows,local_pitch,remote_pitch",
    [
        (384, 2048, 384, 1536),  # expert down_proj shard
        (768 * KiB, 2, 768 * KiB, 3 * 1024 * KiB),  # wider than one SDMA packet (16 Ki elements)
        (4 * 1024 * KiB, 8, 4 * 1024 * KiB, 6 * 1024 * KiB),  # Q blocks of a fused QKV
        (4 * KiB, 20000, 4 * KiB, 8 * KiB),  # more rows than one SDMA packet (16 Ki)
        (1, 100, 1, 4),  # 1-byte elements
        (3, 100, 3, 8),  # odd width, 4-byte pitch: 1-byte elements
        (386, 50, 386, 772),  # pitches not 4-byte aligned
        (16, 4, 16, 16 << 20),  # pitch too large for 16-byte elements in a packet
        (4, 4, 4, 4 << 20),  # pitch too large at any element size
        (4 * KiB, 16, 4 * KiB, 2 * KiB),  # source rows overlap: read more than once
    ],
)
def test_shapes(xgmi_engine, size, rows, local_pitch, remote_pitch):
    """Shapes at and past what ROCr's SDMA rect packets express, and what HIP does with them."""
    _check_blocks(xgmi_engine, [(64, 0, size, rows, local_pitch, remote_pitch)])


def test_unaligned_offsets(xgmi_engine):
    """Blocks starting off a 4-byte boundary, mixed with aligned ones in the same call."""
    _check_blocks(
        xgmi_engine,
        [
            (1, 0, 256, 64, 256, 1024),
            (64 * KiB, 3 + 64 * KiB, 256, 64, 256, 1024),
            (128 * KiB, 128 * KiB, 256, 64, 256, 1024),
        ],
    )


def _backwards(n, size, rows, pitch):
    """n blocks whose destinations run backwards against their sources."""
    return [
        ((n - 1 - i) * rows * size, i * rows * pitch, size, rows, size, pitch)
        for i in range(n)
    ]


def test_many_calls_in_flight(xgmi_engine):
    """Overlapping calls of 62 blocks each, about 40,000 2D copies in flight."""
    n, size, rows, pitch = 8192, 256, 2, 1 * KiB
    src, dst, src_mem, dst_mem = _alloc(
        xgmi_engine, SRC_GPU, DST_GPU, n * rows * pitch, n * rows * size
    )
    blocks = _backwards(n, size, rows, pitch)
    for _ in range(5):
        pending = []
        for b in range(0, n, 62):
            lo, ro, sz, nr, lp, rp = _columns(blocks[b : b + 62])
            (st,) = xgmi_engine.batch_read_strided(
                [dst_mem], [lo], [src_mem], [ro], [sz], [nr], [lp], [rp],
                [xgmi_engine.allocate_transfer_uid()],
            )
            pending.append(st)
        for st in pending:
            st.Wait()
            assert st.Succeeded(), st.Message()
    torch.cuda.synchronize(dst.device)
    assert torch.equal(dst.cpu(), _expected(src, dst.numel(), blocks))


def test_large_batch(xgmi_engine):
    """One call of 20,000 blocks: 20,000 2D copies in one batch."""
    _check_blocks(xgmi_engine, _backwards(20000, 256, 2, 1 * KiB))


_PROBE = """
from tests.python.io.test_xgmi_strided import _make_engine, _check_blocks
_check_blocks(_make_engine("strided_engine_probe"), [(0, 0, 4096, 512, 4096, 8192)])
"""


@pytest.mark.parametrize(
    "rect_env,expected_log",
    [
        (None, "Batch transfer via 2D copies"),
        ("0", "MORI_IO_XGMI_SDMA_RECT=0; strided entries are copied row by row"),
    ],
)
def test_uses_2d_copies(rect_env, expected_log):
    """Strided entries go as 2D copies, unless MORI_IO_XGMI_SDMA_RECT=0. Checked from the
    backend's log in a fresh process, since the switch is read once per process; the level
    comes from the environment because set_log_level does not reach the loggers inside
    libmori_io."""
    import os
    import subprocess
    import sys

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    env = {**os.environ, "MORI_IO_LOG_LEVEL": "trace"}
    env.pop("MORI_IO_XGMI_SDMA_RECT", None)
    if rect_env is not None:
        env["MORI_IO_XGMI_SDMA_RECT"] = rect_env
    out = subprocess.run(
        [sys.executable, "-c", _PROBE],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    log = out.stdout + out.stderr
    assert out.returncode == 0, log[-2000:]
    assert expected_log in log, log[-2000:]
    if rect_env == "0":
        assert "via 2D copies" not in log, log[-2000:]


# ---------------------------------------------------------------------------
# Cross-process: the source memory is opened over IPC, as when an inference
# process reads from a trainer.
# ---------------------------------------------------------------------------

XP_SRC_BYTES = 8 << 20
XP_DST_BYTES = 8 << 20
XP_BLOCKS = [
    (0, 4 * KiB, 4 * KiB, 512, 4 * KiB, 8 * KiB),  # short rows
    (2 << 20, 4 << 20, 192 * KiB, 4, 192 * KiB, 256 * KiB),  # long rows
    (4 << 20, 1 << 20, 1 << 20, 1, 1 << 20, 1 << 20),  # one row
    (5 << 20, 0, 1536 * KiB, 2, 1536 * KiB, 2 << 20),  # rows over 1 MiB
]


def _exchange(obj, rank):
    out = [None, None]
    dist.all_gather_object(out, obj)
    return out[1 - rank]


def _worker(rank, master_port, result_queue, gpus):
    """rank 0 reads from rank 1's buffer."""
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
                status = _read(engine, mem, remote, XP_BLOCKS)
                if not status.Succeeded():
                    ok, detail = False, f"transfer failed: {status.Message()}"
                else:
                    torch.cuda.synchronize()
                    if not torch.equal(buf.cpu(), _expected(src, XP_DST_BYTES, XP_BLOCKS)):
                        ok, detail = False, "cross-process strided read returned wrong bytes"
            dist.barrier()
            result_queue.put(("PASS", "") if ok else ("FAIL", detail))
    except Exception as e:
        import traceback

        result_queue.put(("FAIL", f"{e}\n{traceback.format_exc()}"))


@pytest.mark.parametrize("src_gpu", [1, 2, 3])
def test_cross_process(src_gpu):
    """Over IPC, with the source memory opened from another process."""
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
