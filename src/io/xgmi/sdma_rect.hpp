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
#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "hsakmt/hsakmt.h"

namespace mori {
namespace io {

// SDMA queue owned by MORI on one copy engine of one GPU, fed from the host with 2D
// (COPY_LINEAR_RECT) packets. HIP runs 2D copies between GPUs on the compute units, and ROCr's
// SDMA rect copy only uses the host-to-device and device-to-host queues; this queue runs on the
// xGMI engine the driver recommends for a GPU pair, like HIP's 1D peer copies.
//
// gfx9 only (MI300, MI350): the packet layout differs on gfx12, and gfx9 SDMA copies need no
// cache maintenance packets (ROCr's BlitSdmaV4 emits none either).
class SdmaRectQueue {
 public:
  // The queue, shared per process, on the engine recommended for copies on localBusId's GPU with
  // remoteBusId's GPU. nullptr if the pair has no recommended engine, the GPU is not gfx9, or
  // the queue cannot be created; callers then fall back to hipMemcpy2DAsync.
  static SdmaRectQueue* ForPeer(const std::string& localBusId, const std::string& remoteBusId);

  // Appends the packets for one copy of `rows` rows of `width` bytes. Returns false, appending
  // nothing, if the copy cannot be expressed (addresses or pitches not 4-byte aligned, or a
  // pitch too large for the packet).
  static bool AppendCopy(std::vector<uint32_t>& packets, uintptr_t dst, size_t dstPitch,
                         uintptr_t src, size_t srcPitch, size_t width, size_t rows);

  // Submits packets built by AppendCopy; returns a ticket for Done.
  uint32_t Submit(const std::vector<uint32_t>& packets);
  bool Done(uint32_t ticket) const;

  SdmaRectQueue(const SdmaRectQueue&) = delete;
  SdmaRectQueue& operator=(const SdmaRectQueue&) = delete;

 private:
  SdmaRectQueue() = default;
  bool Init(uint32_t gpuNode, uint32_t engine);
  uint32_t SubmitPiece(const uint32_t* packets, size_t dwords);

  std::mutex mu;
  void* ring{nullptr};
  volatile uint32_t* fence{nullptr};
  HsaQueueResource queue{};
  uint64_t writeIndex{0};
  uint32_t lastTicket{0};
};

}  // namespace io
}  // namespace mori
