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
#include "mori/application/topology/system.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "mori/application/transport/rdma/rdma.hpp"
#include "mori/application/utils/check.hpp"
#include "mori/utils/mori_log.hpp"

namespace mori {
namespace application {

/* ---------------------------------------------------------------------------------------------- */
/*                                           TopoSystem                                           */
/* ---------------------------------------------------------------------------------------------- */
TopoSystem::TopoSystem() { Load(); }

TopoSystem::~TopoSystem() {}

void TopoSystem::Load() {
  gpu.reset(new TopoSystemGpu());
  pci.reset(new TopoSystemPci());
  net.reset(new TopoSystemNet());
}

namespace {

// The KFD sysfs node of every GPU on the host, keyed by PCI bus id. HIP numbers
// the GPUs it can see in this order, and sysfs lists all of them regardless of
// HIP_VISIBLE_DEVICES or of which GPUs a container exposes.
std::unordered_map<uint64_t, long> KfdNodeByBusId() {
  namespace fs = std::filesystem;
  std::unordered_map<uint64_t, long> nodes;
  std::error_code ec;
  for (const auto& entry : fs::directory_iterator("/sys/class/kfd/kfd/topology/nodes", ec)) {
    long node = std::strtol(entry.path().filename().c_str(), nullptr, 10);
    std::ifstream gpuId(entry.path() / "gpu_id");
    uint64_t id = 0;
    if (!(gpuId >> id) || id == 0) continue;

    std::ifstream props(entry.path() / "properties");
    std::string key;
    uint64_t value = 0, location = 0, domain = 0;
    bool hasLocation = false;
    while (props >> key >> value) {
      if (key == "location_id") {
        location = value;
        hasLocation = true;
      } else if (key == "domain") {
        domain = value;
      }
    }
    if (!hasLocation) continue;
    PciBusId bus(static_cast<uint16_t>(domain), static_cast<uint8_t>(location >> 8),
                 static_cast<uint8_t>((location >> 3) & 0x1f), static_cast<uint8_t>(location & 0x7));
    nodes[bus.packed] = node;
  }
  return nodes;
}

}  // namespace

struct Candidate {
  TopoPathPci* path{nullptr};
  TopoNodePci* node{nullptr};
  TopoNodeNic* nic{nullptr};
};

std::vector<Candidate> SortCandidates(TopoSystem* sys, TopoNodeGpu* dev) {
  assert(sys != nullptr && dev != nullptr);

  TopoSystemPci* pci = sys->GetTopoSystemPci();
  TopoSystemNet* net = sys->GetTopoSystemNet();

  TopoNodePci* devPci = pci->Node(dev->busId);
  if (devPci == nullptr) return {};
  NumaNodeId gpuNumaNodeId = devPci->NumaNode();

  // Collect nic candidates
  auto nics = net->GetNics();
  std::vector<Candidate> candidates;
  for (auto* nic : nics) {
    TopoNodePci* nicPci = pci->Node(nic->busId);
    if (!nicPci) continue;  // NIC BDF not present in the scanned PCI tree
    TopoPathPci* path = pci->Path(dev->busId, nic->busId);
    if (!path) continue;
    candidates.push_back({path, nicPci, nic});
  }

  // Sort by 1) speed 2) numa 3) hops 4) name
  std::sort(candidates.begin(), candidates.end(),
            [&gpuNumaNodeId](Candidate a, Candidate b) -> bool {
              bool tie = (a.nic->totalGbps == b.nic->totalGbps);
              if (!tie) return a.nic->totalGbps > b.nic->totalGbps;

              if ((a.node->NumaNode() == gpuNumaNodeId) && (b.node->NumaNode() != gpuNumaNodeId))
                return true;
              if ((a.node->NumaNode() != gpuNumaNodeId) && (b.node->NumaNode() == gpuNumaNodeId))
                return false;

              tie = (a.path->Hops() == b.path->Hops());
              if (!tie) return a.path->Hops() <= b.path->Hops();

              return a.nic->name <= b.nic->name;
            });

  return candidates;
}

std::vector<Candidate> CollectAndSortCandidates(TopoSystem* sys, int id) {
  assert(sys != nullptr);

  TopoNodeGpu* dev = sys->GetTopoSystemGpu()->GetGpuByLogicalId(id);
  if (dev == nullptr) {
    MORI_APP_WARN("Topology: GPU logical id {} not found; no NIC candidates", id);
    return {};
  }
  if (sys->GetTopoSystemPci()->Node(dev->busId) == nullptr) {
    MORI_APP_WARN("Topology: GPU {} bdf {} not in PCI tree; no NIC candidates", id,
                  dev->busId.String());
    return {};
  }
  return SortCandidates(sys, dev);
}

std::string TopoSystem::MatchGpuAndNic(int id) {
  std::vector<std::string> matches = MatchAllGpusAndNics();
  assert(id < matches.size());
  return matches[id];
}

std::vector<std::string> TopoSystem::MatchGpuAndNics(int id, int k) {
  if (k <= 0) return {};

  std::vector<Candidate> candidates = CollectAndSortCandidates(this, id);
  std::vector<std::string> matches;
  matches.reserve(std::min<int>(k, candidates.size()));

  std::unordered_set<std::string> seen;
  for (const auto& candidate : candidates) {
    if (candidate.nic == nullptr) continue;
    const std::string& name = candidate.nic->name;
    if (!seen.insert(name).second) continue;
    matches.push_back(name);
    if (static_cast<int>(matches.size()) >= k) break;
  }

  return matches;
}

std::vector<std::string> TopoSystem::MatchCpuNics(int numaNode, int k) {
  if (k <= 0) return {};

  TopoSystemPci* pci = GetTopoSystemPci();
  TopoSystemNet* net = GetTopoSystemNet();

  auto nics = net->GetNics();
  std::vector<TopoNodeNic*> candidates;
  candidates.reserve(nics.size());
  for (auto* nic : nics) {
    if (nic == nullptr) continue;
    candidates.push_back(nic);
  }

  std::sort(candidates.begin(), candidates.end(), [&](TopoNodeNic* a, TopoNodeNic* b) -> bool {
    if (numaNode >= 0) {
      TopoNodePci* aNode = pci->Node(a->busId);
      TopoNodePci* bNode = pci->Node(b->busId);
      const bool aLocal = (aNode != nullptr) && (aNode->NumaNode() == numaNode);
      const bool bLocal = (bNode != nullptr) && (bNode->NumaNode() == numaNode);
      if (aLocal != bLocal) return aLocal;
    }

    bool tie = (a->totalGbps == b->totalGbps);
    if (!tie) return a->totalGbps > b->totalGbps;

    return a->name <= b->name;
  });

  std::vector<std::string> matches;
  matches.reserve(std::min<int>(k, candidates.size()));
  std::unordered_set<std::string> seen;
  for (const auto* nic : candidates) {
    if (nic == nullptr) continue;
    if (!seen.insert(nic->name).second) continue;
    matches.push_back(nic->name);
    if (static_cast<int>(matches.size()) >= k) break;
  }

  return matches;
}

// NICs are handed out over every GPU on the host, in the order HIP numbers them
// when all are visible, not only over the GPUs this process can see. Otherwise a
// process limited by HIP_VISIBLE_DEVICES to one GPU always gets the first NIC of
// that GPU's NUMA node, so several such processes share one NIC while the NICs of
// their own GPUs sit idle.
std::vector<std::string> TopoSystem::MatchAllGpusAndNics() {
  std::vector<TopoNodeGpu*> hostGpus = gpu->GetGpus();
  std::unordered_map<uint64_t, long> kfdNode = KfdNodeByBusId();
  auto order = [&kfdNode](TopoNodeGpu* g) {
    auto it = kfdNode.find(g->busId.packed);
    return std::make_pair(it == kfdNode.end() ? std::numeric_limits<long>::max() : it->second,
                          g->busId.packed);
  };
  std::sort(hostGpus.begin(), hostGpus.end(),
            [&order](TopoNodeGpu* a, TopoNodeGpu* b) { return order(a) < order(b); });

  std::unordered_map<uint64_t, std::string> nicByBusId;
  std::unordered_set<std::string> matched;

  for (size_t i = 0; i < hostGpus.size(); i++) {
    std::vector<Candidate> candidates = SortCandidates(this, hostGpus[i]);
    if (candidates.empty()) continue;

    std::string& nic = nicByBusId[hostGpus[i]->busId.packed];
    for (auto& cand : candidates) {
      if (matched.insert(cand.nic->name).second) {
        nic = cand.nic->name;
        break;
      }
    }
    if (nic.empty()) nic = candidates[i % candidates.size()].nic->name;
  }

  int count;
  HIP_RUNTIME_CHECK(hipGetDeviceCount(&count));

  std::vector<std::string> matches;
  for (int i = 0; i < count; i++) {
    TopoNodeGpu* dev = gpu->GetGpuByLogicalId(i);
    auto it = (dev == nullptr) ? nicByBusId.end() : nicByBusId.find(dev->busId.packed);
    if (it == nicByBusId.end()) {
      MORI_APP_WARN("Topology: GPU logical id {} has no NIC candidates", i);
      matches.push_back("");
      continue;
    }
    matches.push_back(it->second);
  }

  return matches;
}

}  // namespace application
}  // namespace mori
