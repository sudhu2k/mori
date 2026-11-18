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
#include <dlfcn.h>
#include <unistd.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "mori/application/topology/topology.hpp"

// Prints "<bdf> <nic>" for logical GPU 0, as seen by a child that is limited to one GPU.
int PrintFirstGpuMatch() {
  mori::application::TopoSystem sys{};
  auto* gpu = sys.GetTopoSystemGpu()->GetGpuByLogicalId(0);
  std::string nic = sys.MatchGpuAndNic(0);
  printf("%s %s\n", gpu->busId.String().c_str(), nic.empty() ? "-" : nic.c_str());
  return 0;
}

// A process that sees only GPU i must get the NIC GPU i gets when every GPU is visible.
int TestMatchIgnoresVisibleDevices(mori::application::TopoSystem& sys,
                                   const std::vector<std::string>& matches) {
  if (std::getenv("HIP_VISIBLE_DEVICES") || std::getenv("CUDA_VISIBLE_DEVICES") ||
      std::getenv("ROCR_VISIBLE_DEVICES")) {
    printf("skip visible-device check: run with every GPU visible\n");
    return 0;
  }
  char self[4096] = {};
  if (readlink("/proc/self/exe", self, sizeof(self) - 1) <= 0) return 1;

  int failures = 0;
  for (int i = 0; i < static_cast<int>(matches.size()); i++) {
    std::string cmd = "HIP_VISIBLE_DEVICES=" + std::to_string(i) + " " + self + " --first-gpu";
    FILE* child = popen(cmd.c_str(), "r");
    if (child == nullptr) return 1;
    char bdf[64] = {}, nic[256] = {};
    char line[512] = {};
    while (fgets(line, sizeof(line), child)) sscanf(line, "%63s %255s", bdf, nic);
    int rc = pclose(child);

    std::string want = matches[i].empty() ? "-" : matches[i];
    std::string wantBdf = sys.GetTopoSystemGpu()->GetGpuByLogicalId(i)->busId.String();
    bool ok = rc == 0 && wantBdf == bdf && want == nic;
    printf("gpu %d (%s) alone matches %s, with all GPUs %s: %s\n", i, bdf, nic, want.c_str(),
           ok ? "ok" : "MISMATCH");
    failures += !ok;
  }
  return failures == 0 ? 0 : 1;
}

int TestTopoNodeGpu() {
  mori::application::TopoSystem sys{};
  auto* gpuSys = sys.GetTopoSystemGpu();
  auto* netSys = sys.GetTopoSystemNet();
  auto* pciSys = sys.GetTopoSystemPci();

  auto gpus = gpuSys->GetGpus();
  auto nics = netSys->GetNics();

  for (auto* gpu : gpus) {
    assert(pciSys->Node(gpu->busId));
    for (auto* nic : nics) {
      assert(pciSys->Node(nic->busId));
      auto* path = pciSys->Path(gpu->busId, nic->busId);
      auto* gpuPci = pciSys->Node(gpu->busId);
      auto* nicPci = pciSys->Node(nic->busId);
      if (!path) {
        printf("gpu %s nic %s no direct link\n", gpu->busId.String().c_str(),
               nic->busId.String().c_str());
      } else {
        printf("gpu %s numa %d, nic %s name %s hops %zu speed %f numa %d\n",
               gpu->busId.String().c_str(), gpuPci->NumaNode(), nic->busId.String().c_str(),
               nic->name.c_str(), path->Hops(), nic->totalGbps, nicPci->NumaNode());
      }
    }
  }

  std::vector<std::string> matches = sys.MatchAllGpusAndNics();
  for (int i = 0; i < matches.size(); i++) {
    auto* gpu = gpuSys->GetGpuByLogicalId(i);
    printf("gpu %d (%s) matches %s\n", i, gpu->busId.String().c_str(), matches[i].c_str());
  }

  return TestMatchIgnoresVisibleDevices(sys, matches);
}

int main(int argc, char** argv) {
  if (argc > 1 && std::strcmp(argv[1], "--first-gpu") == 0) return PrintFirstGpuMatch();
  return TestTopoNodeGpu();
}
