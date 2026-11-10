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

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <cstddef>
#include <string>

namespace mori {
namespace io {

// 2D copies between two GPUs on ROCr's SDMA blit queues (hsa_amd_memory_async_copy_rect). HIP
// runs 2D copies between GPUs on the compute units instead.
//
// ROCr's rect copy takes no engine: it runs on the host-to-device or device-to-host blit queue of
// the agent it is given, about 60 GB/s each. Each GPU pair is given one of four queues, the
// destination or the source GPU's, either direction, picked from the two GPUs' KFD node numbers
// alone. Processes that never talk to each other then spread an all-pairs pattern (N readers by
// N sources, N <= 4) over distinct queues.
class SdmaRectCopier {
 public:
  // The copier, shared per process, for copies into dstBusId's GPU from srcBusId's GPU. srcBusId's
  // GPU may be invisible to this process, in which case only the destination's queues are used.
  // nullptr if MORI_IO_XGMI_SDMA_RECT=0, the destination GPU has no HSA agent, or it is not gfx9;
  // callers then use hipMemcpy2DAsync.
  static SdmaRectCopier* ForPair(const std::string& dstBusId, const std::string& srcBusId);

  // Issues one copy of `slices` blocks of `rows` rows of `width` bytes, block k starting
  // k * dstSlice / k * srcSlice bytes after the first; `signal` is incremented before the copy
  // is issued and decremented by ROCr when it completes. Returns false, issuing nothing, if ROCr
  // cannot take the copy (addresses, pitches or slices not 4-byte aligned, or too large).
  //
  // ROCr costs about 7 us of engine time per copy whatever its size, so callers merge what they
  // can into slices.
  bool Copy(void* dst, size_t dstPitch, const void* src, size_t srcPitch, size_t width, size_t rows,
            hsa_signal_t signal, size_t slices = 1, size_t dstSlice = 0, size_t srcSlice = 0) const;

  // Whether CopyRow can run: ROCr recommends an engine for the pair and both GPUs are visible.
  bool HasRowEngine() const { return rowEngine != 0; }

  // Issues one 1D copy on the SDMA engine ROCr recommends for the pair's link, counted on
  // `signal` as in Copy. These engines are per link, where the 2D queues are shared by every
  // pair given the same agent.
  bool CopyRow(void* dst, const void* src, size_t size, hsa_signal_t signal) const;

  SdmaRectCopier(const SdmaRectCopier&) = delete;
  SdmaRectCopier& operator=(const SdmaRectCopier&) = delete;

 private:
  SdmaRectCopier(hsa_agent_t agent, hsa_amd_copy_direction_t queue) : agent(agent), queue(queue) {}

  hsa_agent_t agent;
  // Only selects which of the agent's blit queues runs the copy.
  hsa_amd_copy_direction_t queue;
  hsa_agent_t dstAgent{}, srcAgent{};
  hsa_amd_sdma_engine_id_t rowEngine{static_cast<hsa_amd_sdma_engine_id_t>(0)};
};

}  // namespace io
}  // namespace mori
