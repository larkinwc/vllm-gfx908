<!-- markdownlint-disable MD001 MD041 -->

> **MI100 Users**: This fork adds support for AMD Instinct MI100 (gfx908) GPUs.
> For full setup instructions -- BIOS, drivers, ROCm 7.12, native build, and launch config --
> see **[MI100_SETUP.md](MI100_SETUP.md)**.
> For performance benchmarks and optimization results, see **[BENCH.md](BENCH.md)**.

---

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

---

## What This Fork Adds

This is a fork of [vllm-project/vllm](https://github.com/vllm-project/vllm) (`v0.18.0`) with patches for MI100 (gfx908):

- **FP8 emulation kernel** (`MI100FP8ScaledMMLinearKernel`) -- dequantizes FP8 to FP16 and uses rocBLAS, since MI100 lacks native FP8 hardware
- **MI100 platform detection** -- `on_mi100()`, `_ON_MI100`, gfx908 added to `_ON_GFX9` family
- **Pixtral chunked attention** -- memory-efficient attention for vision transformer

### Quick Start

```bash
git clone https://github.com/larkinwc/vllm-gfx908.git
cd vllm-gfx908 && git checkout mi100-fixes

# See MI100_SETUP.md for full instructions
```

### Upstream vLLM

For the original project, see [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm).
