# Research — fusing the V340's two gfx900 dies into one logical GPU (verdict: no) — 2026-07

Companion to `BENCH_GFX900.md` / `PERF_GFX900.md` / `GFX900_SETUP.md`. Those
docs measure this host; this one records an external research sweep on whether
a **low-level "fuse two dies into one logical device" abstraction** for the
Radeon Pro V340 (2× Vega 10 / gfx900 per board) is feasible or worthwhile, and
cross-checks the findings against our own measurements.

**Method:** multi-agent web research sweep (22 sources fetched, 106 claims
extracted, top 25 adversarially verified by 3-vote panels: 23 confirmed,
2 refuted), 2026-07-23. AI-assisted; all load-bearing claims below carry their
source and cross-check against this branch's measured data where we have it.

## Verdict

**Do not build a fused-device abstraction.** No such mechanism exists in
ROCm/HIP/HSA, AMD explicitly rejected the idea for its own dual-die flagship
on interconnect-bandwidth grounds that apply ~60× more strongly here, and our
own benchmarks already show the winning alternative (PP-heavy multi-device
layouts, see `BENCH_GFX900.md`). The only piece of the "treat the card as a
unit" idea that survives is **card-aware TP2 pair placement** — an open
experiment, see [§Open experiment](#open-experiment-same-card-vs-cross-card-p2p).

## Prior art: AMD's own dual-die parts don't fuse

- MI250/MI250X (Aldebaran) packages two GCDs joined by four in-package
  Infinity Fabric links (200 GB/s peak unidirectional, ~140–150 GB/s
  measured). AMD still exposes **each die as a separate GPU device** to ROCr,
  HIP, Slurm, and `ROCR_VISIBLE_DEVICES` — on Frontier, 4 physical cards
  enumerate as 8 GPUs. Software must be explicitly multi-GPU aware.
  [ROCm MI250 docs](https://rocm.docs.amd.com/en/docs-7.0.2/conceptual/gpu-arch/mi250.html),
  [OLCF Frontier guide](https://docs.olcf.ornl.gov/systems/frontier_user_guide.html),
  [arXiv:2302.14827](https://arxiv.org/pdf/2302.14827).
- AMD's stated rationale (Hot Chips 34, via
  [Chips and Cheese](https://chipsandcheese.com/p/hot-chips-34-amds-instinct-mi200-architecture)):
  inter-die bandwidth couldn't match local HBM bandwidth, so a fused device
  would silently run at interconnect speed whenever data landed on the wrong
  die. MI250's gap is ~8× (200 GB/s link vs 1.6 TB/s HBM2e).
- (MI300X "SPX" partitioning is a firmware mode on a unified-HBM package, not
  a runtime fusion of separate-memory devices — not applicable here.)

**Our numbers make the case worse, not better.** Vega 10 has no XGMI; die-to-die
is PCIe 3.0 only. Measured on this host (`GFX900_SETUP.md`): intra-socket P2P
D2D **~8 GB/s** vs **~365 GB/s** measured local HBM (`PERF_GFX900.md`) — a
**~45–60× penalty** on every wrong-die access, vs the ~8× gap AMD already
judged disqualifying at 200 GB/s.

## The DIY fusion primitives exist but are conditional — and moot

For the record, the building blocks a homemade fusion layer would need:

| Primitive | Status | Gate |
|-----------|--------|------|
| HIP VMM (`hipMemCreate`/`hipMemAddressReserve`/`hipMemMap`/`hipMemSetAccess`) — map one GPU's memory into another's address space | Beta, Linux-only ([HIP docs](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/virtual_memory.html)) | `hipDeviceAttributeVirtualMemoryManagementSupported` per device — **untested on gfx900**; the API postdates gfx900's official support window |
| GPU-originated PCIe P2P writes | First-class in ROCm ([PCIe atomics doc](https://rocm.docs.amd.com/en/docs-6.2.4/conceptual/More-about-how-ROCm-uses-PCIe-Atomics.html)) | GFX9↔GFX9 P2P needs BAR below 2^44 (large-BAR — **confirmed enabled on this host**, full 8G BAR) plus AtomicOp routing through **every** switch in the path, incl. the V340's on-board switch (unverified) |
| RCCL P2P over PCIe | Works — our custom gfx900 RCCL runs TP=8 collectives (`BENCH_GFX900.md`) | `HSA_FORCE_FINE_GRAIN_PCIE=1` + reported peer access, else host-staged fallback ([RCCL usage tips](https://rocm.docs.amd.com/projects/rccl/en/develop/how-to/rccl-usage-tips.html)) |
| SDMA explicit copies | Capped ~51 GB/s/transfer even on MI250X fabric ([arXiv:2302.14827](https://arxiv.org/pdf/2302.14827)) | Irrelevant here: our ~8 GB/s PCIe path is the binding constraint, far below the SDMA cap |

Even if every gate passed, all cross-die traffic runs at ~8 GB/s. A fused
address space adds Beta-API risk on a deprecated architecture and offers no
scheduling benefit that two well-overlapped HIP streams (or vLLM's existing
multi-device engine) don't already provide.

**Refuted claim worth recording:** "kernel-driven implicit peer load/store is
the transfer method that saturates every interconnect" was rejected 0–3 in
verification. The fastest die-to-die copy mechanism on gfx900 (SDMA vs blit
kernels via `HSA_ENABLE_SDMA=0` vs kernel load/store) is **unresolved** — worth
a microbenchmark only if a custom P2P path is ever built.

## Cross-check: external guidance vs our measured data

| Research finding (external) | Our measurement | Agrees? |
|-----------------------------|-----------------|---------|
| AMD vLLM guide: PP "may perform better than TP" without NVLink/XGMI ([vLLM optimization](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html)) | TP2×PP2 beats TP4 by **+48–90%**; TP2×PP4 beats TP8 on 3/4 cells (`BENCH_GFX900.md` §PP grid) | Yes — and our data is far stronger than AMD's hedged "may" |
| llama.cpp defaults to layer-split PP as the interconnect-tolerant mode; its TP "meta device" targets fast interconnects/CUDA only ([multi-gpu.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md)) | Same conclusion independently: lowest-TP/highest-PP that fits is the throughput sweet spot | Yes |
| gfx900 dropped after ROCm 4.5.x but distro builds still ship gfx900 Tensile targets (verified in Arch rocBLAS 7.2.4) | Running ROCm 7.2.4 with custom gfx900 RCCL end-to-end | Yes |
| RCCL gfx900 multi-GPU collectives still functional (MI25 TP=4 report, ROCm 7.2.1) | TP=8 serving works; known PP async-scheduling broadcast bug → `--no-async-scheduling` (`BENCH_GFX900.md`) | Yes, incl. sharper caveat |

Net: the external sweep independently converged on exactly the layout this
branch already promoted (`GFX900_RECOMMENDED.md`). It found **no** missed
mechanism that would change the recommendation.

## Open experiment: same-card vs cross-card P2P

The one thing neither the sweep nor our docs answer. KFD reports all dies
uniformly PCIe-attached (weight 40, 2 hops — `GFX900_SETUP.md`), i.e. ROCm sees
no locality difference between two dies **on the same V340 board** (behind its
on-board PCIe switch) and dies on different boards. Physically, a same-card
pair could do P2P through the on-board switch without touching the root
complex.

- **Hypothesis:** same-card die pairs have higher P2P bandwidth and/or lower
  latency than cross-card pairs, and/or leave host PCIe lanes free.
- **Test:** pairwise D2D bandwidth/latency sweep across all die pairs
  (extend the existing P2P harness), plus `lspci -vvv` to identify the
  on-board switch and its AtomicOp routing capability.
- **Payoff if true:** card-aware TP2 pair placement (pair dies 2n/2n+1 per
  board) in the TP2×PP{2,4} layouts — a scheduling change, not an abstraction.
- **Payoff if false:** closes the last remnant of the fusion idea; pair
  placement stays free.

## Sources (verified subset)

Primary: ROCm MI250 arch docs, OLCF Frontier guide, HIP VMM docs, ROCm PCIe
atomics doc, RCCL usage tips, AMD vLLM optimization guide, llama.cpp
multi-gpu.md, arXiv:2302.14827, arXiv:2410.00801. Secondary: Chips and Cheese
HC34 coverage, ArchWiki MI25 page, ROCm GitHub issues #787/#6074, RCCL #80/#92.
