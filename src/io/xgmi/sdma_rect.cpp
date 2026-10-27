// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include "src/io/xgmi/sdma_rect.hpp"

#include <sched.h>

#include <algorithm>
#include <atomic>
#include <climits>
#include <cstdio>
#include <cstring>
#include <map>
#include <optional>
#include <utility>

#include "mori/io/logging.hpp"

namespace mori {
namespace io {
namespace {

// SDMA packets, gfx9 layout (ROCr's amd_blit_sdma.cpp and sdma_registers.h).
constexpr uint32_t kOpCopy = 1;
constexpr uint32_t kSubOpCopyLinearRect = 4;
constexpr uint32_t kOpFence = 5;
constexpr size_t kRectDwords = 13;
constexpr size_t kFenceDwords = 4;
constexpr uint32_t kRectMaxPitch = 1u << 19;  // in elements
constexpr uint32_t kRectMaxXY = 1u << 14;     // in elements (x) and rows (y)

constexpr uint64_t kRingBytes = 1 << 20;
// Large submissions are split so that one piece never needs more than a quarter of the ring.
constexpr size_t kPieceDwords = (kRingBytes / 4 / 4) / kRectDwords * kRectDwords;

bool OpenKfd() {
  static const bool ok = []() {
    if (hsaKmtOpenKFD() != HSAKMT_STATUS_SUCCESS) return false;
    HsaSystemProperties sys{};
    return hsaKmtAcquireSystemProperties(&sys) == HSAKMT_STATUS_SUCCESS;
  }();
  return ok;
}

std::optional<HsaNodeProperties> NodeProperties(uint32_t node) {
  HsaNodeProperties props{};
  if (hsaKmtGetNodeProperties(node, &props) != HSAKMT_STATUS_SUCCESS) return std::nullopt;
  return props;
}

std::vector<HsaIoLinkProperties> IoLinks(uint32_t node) {
  auto props = NodeProperties(node);
  if (!props || props->NumIOLinks == 0) return {};
  std::vector<HsaIoLinkProperties> links(props->NumIOLinks);
  if (hsaKmtGetNodeIoLinkProperties(node, props->NumIOLinks, links.data()) !=
      HSAKMT_STATUS_SUCCESS)
    return {};
  return links;
}

// KFD node of the GPU with PCI bus ID "dddd:bb:dd.f".
std::optional<uint32_t> GpuNode(const std::string& busId) {
  unsigned domain = 0, bus = 0, dev = 0, fn = 0;
  if (std::sscanf(busId.c_str(), "%x:%x:%x.%x", &domain, &bus, &dev, &fn) != 4) {
    return std::nullopt;
  }
  const uint32_t location = (bus << 8) | (dev << 3) | fn;
  HsaSystemProperties sys{};
  if (hsaKmtAcquireSystemProperties(&sys) != HSAKMT_STATUS_SUCCESS) return std::nullopt;
  for (uint32_t node = 0; node < sys.NumNodes; ++node) {
    auto props = NodeProperties(node);
    if (props && props->KFDGpuID != 0 && props->Domain == domain &&
        props->LocationId == location)
      return node;
  }
  return std::nullopt;
}

uint32_t NearestCpuNode(uint32_t gpuNode) {
  uint32_t best = 0, bestWeight = UINT32_MAX;
  for (const auto& link : IoLinks(gpuNode)) {
    auto to = NodeProperties(link.NodeTo);
    if (to && to->NumCPUCores != 0 && to->NumFComputeCores == 0 && link.Weight < bestWeight) {
      best = link.NodeTo;
      bestWeight = link.Weight;
    }
  }
  return best;
}

}  // namespace

SdmaRectQueue* SdmaRectQueue::ForPeer(const std::string& localBusId,
                                      const std::string& remoteBusId) {
  if (!OpenKfd()) {
    static std::once_flag warned;
    std::call_once(warned, []() {
      MORI_IO_WARN("XGMI: cannot open KFD; strided copies use hipMemcpy2DAsync");
    });
    return nullptr;
  }
  auto localNode = GpuNode(localBusId), remoteNode = GpuNode(remoteBusId);
  if (!localNode || !remoteNode || *localNode == *remoteNode) return nullptr;
  auto props = NodeProperties(*localNode);
  if (!props || props->EngineId.ui32.Major != 9) {
    MORI_IO_INFO("XGMI: strided SDMA copies need gfx9 (GPU {}); using hipMemcpy2DAsync",
                 localBusId);
    return nullptr;
  }
  uint32_t engineMask = 0;
  for (const auto& link : IoLinks(*localNode)) {
    if (link.NodeTo == *remoteNode) engineMask = link.RecSdmaEngIdMask;
  }
  if (engineMask == 0) {
    MORI_IO_INFO("XGMI: no recommended SDMA engine for {} -> {}; using hipMemcpy2DAsync",
                 remoteBusId, localBusId);
    return nullptr;
  }
  const uint32_t engine = static_cast<uint32_t>(__builtin_ctz(engineMask));

  // One queue per (GPU, engine), kept for the life of the process.
  static std::mutex mu;
  static auto* queues = new std::map<std::pair<uint32_t, uint32_t>, SdmaRectQueue*>();
  std::lock_guard<std::mutex> lock(mu);
  auto& q = (*queues)[{*localNode, engine}];
  if (q == nullptr) {
    auto* created = new SdmaRectQueue();
    if (!created->Init(*localNode, engine)) {
      MORI_IO_WARN("XGMI: failed to create SDMA queue on {} engine {}; using hipMemcpy2DAsync",
                   localBusId, engine);
      delete created;
      queues->erase({*localNode, engine});
      return nullptr;
    }
    MORI_IO_INFO("XGMI: strided SDMA copies for {} -> {} on engine {}", remoteBusId, localBusId,
                 engine);
    q = created;
  }
  return q;
}

bool SdmaRectQueue::Init(uint32_t gpuNode, uint32_t engine) {
  // Ring and fence live in system memory, as ROCr's do: CPU writes through a VRAM BAR mapping
  // are not guaranteed to reach the engine without an HDP flush, which hangs the engine.
  uint32_t cpuNode = NearestCpuNode(gpuNode);
  HsaMemFlags flags{};
  flags.ui32.HostAccess = 1;
  flags.ui32.PageSize = HSA_PAGE_SIZE_4KB;
  flags.ui32.ExecuteAccess = 1;
  flags.ui32.Uncached = 1;
  auto alloc = [&](size_t bytes, void** ptr) {
    HSAuint64 gpuVa = 0;
    return hsaKmtAllocMemory(cpuNode, bytes, flags, ptr) == HSAKMT_STATUS_SUCCESS &&
           hsaKmtMapMemoryToGPUNodes(*ptr, bytes, &gpuVa, HsaMemMapFlags{}, 1, &gpuNode) ==
               HSAKMT_STATUS_SUCCESS;
  };
  void* fenceMem = nullptr;
  if (!alloc(kRingBytes, &ring) || !alloc(4096, &fenceMem)) return false;
  std::memset(ring, 0, kRingBytes);
  fence = static_cast<volatile uint32_t*>(fenceMem);
  *fence = 0;
  if (hsaKmtCreateQueueExt(gpuNode, HSA_QUEUE_SDMA_BY_ENG_ID, 100, HSA_QUEUE_PRIORITY_MAXIMUM,
                           engine, ring, kRingBytes, nullptr, &queue) != HSAKMT_STATUS_SUCCESS)
    return false;
  writeIndex = *queue.Queue_write_ptr_aql;
  return true;
}

bool SdmaRectQueue::AppendCopy(std::vector<uint32_t>& packets, uintptr_t dst, size_t dstPitch,
                               uintptr_t src, size_t srcPitch, size_t width, size_t rows) {
  // Tiled as in ROCr's BuildCopyRectCommand. Elements are 1-16 bytes (log2 in `element`); the
  // pitches must be whole elements, and each tile's base addresses 4-byte aligned.
  auto log2Align = [](size_t v) { return static_cast<int>(__builtin_ctzll(v | 16)); };
  if (dst % 4 || src % 4 || dstPitch % 4 || srcPitch % 4) return false;
  const int maxElement = std::min(log2Align(srcPitch), log2Align(dstPitch));
  const int minElement = std::min(maxElement, log2Align(width));
  if ((srcPitch >> minElement) > kRectMaxPitch || (dstPitch >> minElement) > kRectMaxPitch)
    return false;
  for (size_t y = 0; y < rows; y += kRectMaxXY) {
    const uint32_t tileRows = static_cast<uint32_t>(std::min<size_t>(rows - y, kRectMaxXY));
    size_t x = 0;
    while (x < width) {
      const size_t remaining = width - x;
      const int alignedElement = std::min(log2Align(x % 4), maxElement);
      int element = std::min(log2Align(remaining), alignedElement);
      size_t count = remaining >> element;
      if (count > kRectMaxXY) {
        element = alignedElement;
        count = std::min<size_t>(remaining >> element, kRectMaxXY);
      }
      uintptr_t s = src + x + y * srcPitch, d = dst + x + y * dstPitch;
      const uint32_t srcOffsetX = (s % 4) >> element, dstOffsetX = (d % 4) >> element;
      s &= ~uintptr_t{3};
      d &= ~uintptr_t{3};
      x += count << element;
      const uint32_t pkt[kRectDwords] = {
          kOpCopy | (kSubOpCopyLinearRect << 8) | (static_cast<uint32_t>(element) << 29),
          static_cast<uint32_t>(s),
          static_cast<uint32_t>(s >> 32),
          srcOffsetX,
          static_cast<uint32_t>((srcPitch >> element) - 1) << 13,
          0,  // source slice pitch, unused for one slice
          static_cast<uint32_t>(d),
          static_cast<uint32_t>(d >> 32),
          dstOffsetX,
          static_cast<uint32_t>((dstPitch >> element) - 1) << 13,
          0,  // destination slice pitch
          static_cast<uint32_t>(count - 1) | ((tileRows - 1) << 16),
          0,  // one slice
      };
      packets.insert(packets.end(), pkt, pkt + kRectDwords);
    }
  }
  return true;
}

uint32_t SdmaRectQueue::Submit(const std::vector<uint32_t>& packets) {
  uint32_t ticket = 0;
  for (size_t off = 0; off < packets.size(); off += kPieceDwords) {
    ticket = SubmitPiece(packets.data() + off, std::min(kPieceDwords, packets.size() - off));
  }
  return ticket;
}

uint32_t SdmaRectQueue::SubmitPiece(const uint32_t* packets, size_t dwords) {
  std::lock_guard<std::mutex> lock(mu);
  const uint32_t ticket = ++lastTicket;
  const uint64_t bytes = (dwords + kFenceDwords) * 4;
  // A submission does not wrap: pad to the end of the ring with NOPs (zero dwords) instead.
  const uint64_t pad =
      (writeIndex % kRingBytes + bytes > kRingBytes) ? kRingBytes - writeIndex % kRingBytes : 0;
  auto readIndex = [this]() { return *static_cast<volatile uint64_t*>(queue.Queue_read_ptr_aql); };
  while (writeIndex + pad + bytes - readIndex() >= kRingBytes) sched_yield();

  auto* base = static_cast<volatile uint32_t*>(ring);
  for (uint64_t i = 0; i < pad / 4; ++i) base[writeIndex % kRingBytes / 4 + i] = 0;
  writeIndex += pad;
  volatile uint32_t* p = base + writeIndex % kRingBytes / 4;
  for (size_t i = 0; i < dwords; ++i) p[i] = packets[i];
  const auto fenceAddr = reinterpret_cast<uintptr_t>(fence);
  p[dwords + 0] = kOpFence;
  p[dwords + 1] = static_cast<uint32_t>(fenceAddr);
  p[dwords + 2] = static_cast<uint32_t>(fenceAddr >> 32);
  p[dwords + 3] = ticket;
  writeIndex += bytes;

  std::atomic_thread_fence(std::memory_order_seq_cst);
  *static_cast<volatile uint64_t*>(queue.Queue_write_ptr_aql) = writeIndex;
  std::atomic_thread_fence(std::memory_order_seq_cst);
  *static_cast<volatile uint64_t*>(queue.Queue_DoorBell_aql) = writeIndex;
  return ticket;
}

bool SdmaRectQueue::Done(uint32_t ticket) const {
  return static_cast<int32_t>(*fence - ticket) >= 0;
}

}  // namespace io
}  // namespace mori
