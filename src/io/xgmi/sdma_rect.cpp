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

#include <cstdint>
#include <cstdio>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#include "mori/io/logging.hpp"
#include "mori/utils/env_utils.hpp"

namespace mori {
namespace io {
namespace {

// MORI_IO_XGMI_SDMA_RECT=0 sends every strided copy to hipMemcpy2DAsync.
bool SdmaRectEnabled() {
  static const bool enabled = []() {
    const char* v = mori::env::Get("MORI_IO_XGMI_SDMA_RECT");
    if (v == nullptr || v[0] == '\0') return true;
    return mori::env::detail::ParseBool(v).value_or(true);
  }();
  return enabled;
}

struct GpuAgent {
  hsa_agent_t agent;
  uint32_t domain;
  uint32_t bdf;
  uint32_t node;
  bool gfx9;
};

const std::vector<GpuAgent>& GpuAgents() {
  static const auto* agents = []() {
    auto* out = new std::vector<GpuAgent>();
    // HIP has already initialized ROCr; this only takes a reference, never released.
    if (hsa_init() != HSA_STATUS_SUCCESS) return out;
    hsa_iterate_agents(
        [](hsa_agent_t agent, void* data) {
          hsa_device_type_t type;
          if (hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type) != HSA_STATUS_SUCCESS ||
              type != HSA_DEVICE_TYPE_GPU)
            return HSA_STATUS_SUCCESS;
          GpuAgent a{agent, 0, 0, 0, false};
          char name[64] = {};
          hsa_agent_get_info(agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_DOMAIN),
                             &a.domain);
          hsa_agent_get_info(agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_BDFID),
                             &a.bdf);
          hsa_agent_get_info(
              agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_DRIVER_NODE_ID), &a.node);
          hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, name);
          a.gfx9 = name[0] == 'g' && name[1] == 'f' && name[2] == 'x' && name[3] == '9';
          static_cast<std::vector<GpuAgent>*>(data)->push_back(a);
          return HSA_STATUS_SUCCESS;
        },
        out);
    return out;
  }();
  return *agents;
}

// The agent of the GPU with PCI bus ID "dddd:bb:dd.f", if this process sees it.
std::optional<GpuAgent> AgentFor(const std::string& busId) {
  unsigned domain = 0, bus = 0, dev = 0, fn = 0;
  if (std::sscanf(busId.c_str(), "%x:%x:%x.%x", &domain, &bus, &dev, &fn) != 4) {
    return std::nullopt;
  }
  const uint32_t bdf = (bus << 8) | (dev << 3) | fn;
  for (const auto& a : GpuAgents()) {
    if (a.domain == domain && a.bdf == bdf) return a;
  }
  return std::nullopt;
}

}  // namespace

SdmaRectCopier* SdmaRectCopier::ForPair(const std::string& dstBusId, const std::string& srcBusId) {
  if (!SdmaRectEnabled()) {
    static std::once_flag logged;
    std::call_once(logged, []() {
      MORI_IO_INFO("XGMI: MORI_IO_XGMI_SDMA_RECT=0; strided copies use hipMemcpy2DAsync");
    });
    return nullptr;
  }
  auto dst = AgentFor(dstBusId);
  if (!dst) {
    MORI_IO_INFO("XGMI: no HSA agent for GPU {}; strided copies into it use hipMemcpy2DAsync",
                 dstBusId);
    return nullptr;
  }
  if (!dst->gfx9) {
    MORI_IO_INFO("XGMI: strided SDMA copies need gfx9 (GPU {}); using hipMemcpy2DAsync", dstBusId);
    return nullptr;
  }
  auto src = AgentFor(srcBusId);

  static std::mutex mu;
  static auto* copiers = new std::map<std::pair<std::string, std::string>, SdmaRectCopier*>();
  std::lock_guard<std::mutex> lock(mu);
  auto& c = (*copiers)[{dstBusId, srcBusId}];
  if (c == nullptr) {
    // Destination H2D, destination D2H, source H2D, source D2H. With N readers each pulling
    // from the same N sources (N <= 4), every queue gets one pair.
    const uint32_t pick =
        src ? (src->node - dst->node) % 4 : std::hash<std::string>()(srcBusId) % 2;
    const bool onSource = pick >= 2;
    const auto queue = pick % 2 == 0 ? hsaHostToDevice : hsaDeviceToHost;
    c = new SdmaRectCopier(onSource ? src->agent : dst->agent, queue);
    // ROCr runs a copy between GPUs on the source GPU, and recommends from that GPU's table, so
    // the engine is the one for this link (unlike HIP before 7.15, which ran it on the
    // destination with the source's engine number).
    uint32_t preferred = 0;
    if (src && src->node != dst->node &&
        hsa_amd_memory_get_preferred_copy_engine(dst->agent, src->agent, &preferred) ==
            HSA_STATUS_SUCCESS &&
        preferred != 0) {
      c->dstAgent = dst->agent;
      c->srcAgent = src->agent;
      c->rowEngine = static_cast<hsa_amd_sdma_engine_id_t>(preferred & (~preferred + 1));
    }
    MORI_IO_INFO(
        "XGMI: strided SDMA copies for {} -> {} on {}'s {} blit queue, long rows on engine "
        "mask {:#x}",
        srcBusId, dstBusId, onSource ? srcBusId : dstBusId,
        queue == hsaHostToDevice ? "host-to-device" : "device-to-host",
        static_cast<uint32_t>(c->rowEngine));
  }
  return c;
}

bool SdmaRectCopier::Copy(void* dst, size_t dstPitch, const void* src, size_t srcPitch,
                          size_t width, size_t rows, hsa_signal_t signal, size_t slices,
                          size_t dstSlice, size_t srcSlice) const {
  if (width > UINT32_MAX || rows > UINT32_MAX || slices > UINT32_MAX) return false;
  if (slices == 1) dstSlice = srcSlice = 0;
  hsa_pitched_ptr_t d{dst, dstPitch, dstSlice};
  hsa_pitched_ptr_t s{const_cast<void*>(src), srcPitch, srcSlice};
  const hsa_dim3_t zero{0, 0, 0};
  const hsa_dim3_t range{static_cast<uint32_t>(width), static_cast<uint32_t>(rows),
                         static_cast<uint32_t>(slices)};
  hsa_signal_add_relaxed(signal, 1);
  if (hsa_amd_memory_async_copy_rect(&d, &zero, &s, &zero, &range, agent, queue, 0, nullptr,
                                     signal) != HSA_STATUS_SUCCESS) {
    hsa_signal_subtract_relaxed(signal, 1);
    return false;
  }
  return true;
}

bool SdmaRectCopier::CopyRow(void* dst, const void* src, size_t size, hsa_signal_t signal) const {
  if (rowEngine == 0) return false;
  hsa_signal_add_relaxed(signal, 1);
  if (hsa_amd_memory_async_copy_on_engine(dst, dstAgent, src, srcAgent, size, 0, nullptr, signal,
                                          rowEngine, false) != HSA_STATUS_SUCCESS) {
    hsa_signal_subtract_relaxed(signal, 1);
    return false;
  }
  return true;
}

}  // namespace io
}  // namespace mori
