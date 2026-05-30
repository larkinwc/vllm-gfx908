# F-M2 — Marlin W4A16 autotune config seeding (gfx908 / MI100)

Per-shape autotune configs for the MI100 Marlin-style **W4A16** Triton GEMM
(`vllm/model_executor/kernels/linear/scaled_mm/mi100_w4a16_marlin.py`).

## Wiring (how seeded JSONs become active)

`mi100_w4a16_marlin_gemm()` -> `_select_marlin_config(M, N, K, group_size)`
calls the shared gfx908 loader:

```python
from vllm.model_executor.kernels.configs.gfx908.config_loader import (
    load_config as _load_mi100_autotune_config,
)
cfg = _load_mi100_autotune_config(
    "mi100_w4a16_marlin", M=M, N=N, K=K, group_size=group_size)
```

* **Kernel key:** `mi100_w4a16_marlin`
* **Filename convention** (from `config_loader._config_path`, group_size
  branch): `mi100_w4a16_marlin_M{M}_N{N}_K{K}_g{group_size}.json`
* `load_config` returns the inner **`config`** dict; `_select_marlin_config`
  merges it onto the heuristic default and maps `GROUP_SIZE_M` -> `GROUP_M`.
* **Default-off preserved:** the lookup only fires on the Marlin path; when no
  JSON matches, `_default_marlin_config(M)` is used unchanged. The global
  escape hatch `VLLM_MI100_DISABLE_AUTOTUNE_CONFIG=1` forces the heuristic.

> Schema note: the inner `config` MUST use the keys the GEMM reads
> (`BLOCK_M`, `BLOCK_N`, `BLOCK_K`, `GROUP_SIZE_M`, `num_warps`, `num_stages`),
> matching the legacy `mi100_w4a16_*` files — NOT a `BLOCK_SIZE_*` schema.

## Catalog (48 = M x N x K x g)

* M in {1, 32, 128, 512}  (4)
* N in {4096, 10240, 24576}  (3)
* K in {4096, 12288}  (2)
* g in {32, 128}  (2)

4 x 3 x 2 x 2 = **48 files**. Tile selection depends only on (M, g); N and K
do not change the tile (see decoupling note below).

## LDS budget (gfx908, 64 KiB)

LDS estimate per resident tile set (conservative):

```
LDS_bytes = num_stages * BLOCK_K * (BLOCK_M + BLOCK_N) * 2      # fp16 act term
          + num_stages * (BLOCK_K // 8) * BLOCK_N * 4           # packed int4 B
```

The first term counts the contraction footprint at fp16 width across
(BLOCK_M + BLOCK_N) columns (deliberately over-counts the B side); the second
adds the packed int4 B-tile stored as `[BLOCK_K//8, BLOCK_N]` int32. Tiles
> 64 KiB are pruned by dropping `BLOCK_N` 256 -> 128 before writing.

| role | BLOCK_M | BLOCK_N | BLOCK_K | GROUP_SIZE_M | num_warps | num_stages | LDS bytes | KiB | note |
|------|--------:|--------:|--------:|-------------:|----------:|-----------:|----------:|----:|------|
| decode (M=1, g=32) | 16 | 128 | 32 | 8 | 4 | 2 | 22528 | 22.0 | PASS |
| decode (M=1, g=128) | 16 | 128 | 64 | 8 | 4 | 2 | 45056 | 44.0 | PASS |
| small (M=32, g in {32,128}) | 32 | 128 | 32 | 8 | 4 | 2 | 24576 | 24.0 | PASS |
| prefill (M in {128,512}, g in {32,128}) | 128 | 256 | 32 | 8 | 8 | 2 | 57344 | 56.0 | PASS |

All four distinct tiles are <= 64 KiB, so no `BLOCK_N` pruning was triggered
(the largest, prefill 128x256x32, is 56.0 KiB).

### Tile rules applied

* decode `M=1`: BLOCK_M=16, BLOCK_N=128, BLOCK_K=min(64, g), GROUP_SIZE_M=8,
  num_warps=4, num_stages=2.
* small `M=32`: BLOCK_M=32, BLOCK_N=128, BLOCK_K=min(32, g), GROUP_SIZE_M=8,
  num_warps=4, num_stages=2.
* prefill `M in {128, 512}`: BLOCK_M=128, BLOCK_N=256 (->128 if over budget),
  BLOCK_K=min(32, g), GROUP_SIZE_M=8, num_warps=8, num_stages=2.

`BLOCK_K` is always a multiple of 8 and divides `group_size`
(g=32 -> 32; g=128 -> 64 for decode, 32 elsewhere).

## N=10240 producer-overflow does NOT recur here

`BENCH_M2_PRODUCER_WIRE_IN.md` §11 documents an int8 producer-path overflow at
N=10240 caused by `EMIT_INT8_NEXT` sizing its store tile as
`BLOCK_N = next_pow2(N)` (10240 -> 16384), which blows the LDS/register budget.

This **cannot** recur for the Marlin GEMM: its `BLOCK_N` is a fixed tile knob
(128 or 256) **decoupled from N**. The kernel iterates output columns in
`BLOCK_N`-wide tiles (`pid_n`, `offs_bn`), so N only changes the grid size, not
the tile footprint. The LDS estimate above is independent of N; every N in the
catalog reuses the same (M, g)-selected tile.

## File list + per-file SHA256

 1. `mi100_w4a16_marlin_M128_N10240_K12288_g128.json`  
    `cd71746dba3bd084adf4e28d1f7fb74655f0f19122b3502f07b3323809fd20b6`
 2. `mi100_w4a16_marlin_M128_N10240_K12288_g32.json`  
    `542bc809d8f8e97d80f7a9b63efcac1ecc24bb6845baf9a1067cf8910f1a9f0f`
 3. `mi100_w4a16_marlin_M128_N10240_K4096_g128.json`  
    `33934538657d0652203990c259231c5061c5e192beb2cbe66b4aaa9b6ec930c0`
 4. `mi100_w4a16_marlin_M128_N10240_K4096_g32.json`  
    `71dc286e7aa8547ea8345938102e2f3b3b8d2cdea493b326ba8c59deab52422e`
 5. `mi100_w4a16_marlin_M128_N24576_K12288_g128.json`  
    `dcc13a72e771b9729ea142036e601f8047a5e623eb008a1fdf50616e58ce9d71`
 6. `mi100_w4a16_marlin_M128_N24576_K12288_g32.json`  
    `9971b86811e75fdc74b1f0f22449bc8f79f8395e773d42fcf120f0925892a2c6`
 7. `mi100_w4a16_marlin_M128_N24576_K4096_g128.json`  
    `f2345bcaf201eb65d31e19978fcf9d35aa54cac756cf70164f7892c51054dbb8`
 8. `mi100_w4a16_marlin_M128_N24576_K4096_g32.json`  
    `9992c158b5faa8eacf40d42a857ec6bd35aa5e221356c4c523c18e2a637b8be9`
 9. `mi100_w4a16_marlin_M128_N4096_K12288_g128.json`  
    `ade9856955417db407e6c28a3ce9bf64ae978a4d0c8a8bf046ec69d84017f117`
10. `mi100_w4a16_marlin_M128_N4096_K12288_g32.json`  
    `f76d901e5937ee1a25d16583db131ee84c8b4cc2006fd22d280eedab1c43630d`
11. `mi100_w4a16_marlin_M128_N4096_K4096_g128.json`  
    `9fa484e56572bc58829e3ce5ca5ac458f892a0ab196ded3da2dd161998035987`
12. `mi100_w4a16_marlin_M128_N4096_K4096_g32.json`  
    `657511b173f7cf4f983cdaa6ee543cbf1db432804512ba6d70b70966d78bc1af`
13. `mi100_w4a16_marlin_M1_N10240_K12288_g128.json`  
    `4d531c16e88726737eee737eebb0c92c647d3016d65883a5a8e5ddf826012e98`
14. `mi100_w4a16_marlin_M1_N10240_K12288_g32.json`  
    `d99ab139ed42972ec06a49811abb3076cfb10951b2bf2b01d6398d1b29c19cac`
15. `mi100_w4a16_marlin_M1_N10240_K4096_g128.json`  
    `e77ecd95a841ea3c4240d1f4bf004ebf59159323cf1325b740a9fa2d400e5ea1`
16. `mi100_w4a16_marlin_M1_N10240_K4096_g32.json`  
    `dc4de96bae21692fb982eeef790657e9c34a5c30da0d209c57c1e23f623c236d`
17. `mi100_w4a16_marlin_M1_N24576_K12288_g128.json`  
    `19a4852f0b4f89e90202e8bd4f1a114dc32d88e9537d2cef40042e42c76d6f07`
18. `mi100_w4a16_marlin_M1_N24576_K12288_g32.json`  
    `2b764f1bd624ea693021578afaef6abd6d83d07ffda6748a342ea61fbdca2bcf`
19. `mi100_w4a16_marlin_M1_N24576_K4096_g128.json`  
    `37594a84bb4e7ef2fae57748a7998688956a893f121be1f399a4a90cfd10be90`
20. `mi100_w4a16_marlin_M1_N24576_K4096_g32.json`  
    `89294acc6705c59dfd0e4072a84310a074b7b747c2f730538171dd9ad486cdfa`
21. `mi100_w4a16_marlin_M1_N4096_K12288_g128.json`  
    `fed0331ee46f74c9c71554489c20bd1f542fd4860e20d3aeedae24e459a3c80b`
22. `mi100_w4a16_marlin_M1_N4096_K12288_g32.json`  
    `61da43ae1d194ae87d929a883b737fa6b2e91dc89ffa6f79e499130b1e77a773`
23. `mi100_w4a16_marlin_M1_N4096_K4096_g128.json`  
    `d163f1f2838ffc95b42cc239cf8988d87637a5f4d114ac6595c891c3e727af8c`
24. `mi100_w4a16_marlin_M1_N4096_K4096_g32.json`  
    `e81c65d5cde4751d24eff9a362fba5efe1e66792ad4f81530608f9b9e7569e46`
25. `mi100_w4a16_marlin_M32_N10240_K12288_g128.json`  
    `c808957137681960d9256597666b7615d2e0c2c675a04bd0a70e7a00764d3103`
26. `mi100_w4a16_marlin_M32_N10240_K12288_g32.json`  
    `31a95ccb8794a54822a1c809adc9cb63c85706383250ed5feaee685ce067efa3`
27. `mi100_w4a16_marlin_M32_N10240_K4096_g128.json`  
    `1b187ee9f61687b7d40b26a3784b87b570bc74f238cee9a29890d57dbf065544`
28. `mi100_w4a16_marlin_M32_N10240_K4096_g32.json`  
    `9aae63fff8c9970e56b43677d81f1e60474b1531fbd69457edeb32cc0a5bd2cc`
29. `mi100_w4a16_marlin_M32_N24576_K12288_g128.json`  
    `676c5941561cee7ed432137b8cdb2be89efcc111bdedf407f975f45919e9ae6b`
30. `mi100_w4a16_marlin_M32_N24576_K12288_g32.json`  
    `1311026c5f1948e4bf6cbfc9ee2dc7ed1f46987d19bf05227e244ae3d6aee76f`
31. `mi100_w4a16_marlin_M32_N24576_K4096_g128.json`  
    `11e5b49d4535f68631fc72902fc2eb2e67bf5d1321f3bb1065d7c9e1e1bb50b7`
32. `mi100_w4a16_marlin_M32_N24576_K4096_g32.json`  
    `87340f3a8005e33b28ac94c3599e83d4352032a9e1b6a5dfad198f367760456d`
33. `mi100_w4a16_marlin_M32_N4096_K12288_g128.json`  
    `9410f6e5a5066ff1a889572dc875d2b77a6dd8a23b4f18e9f8c8bb3ea87c584c`
34. `mi100_w4a16_marlin_M32_N4096_K12288_g32.json`  
    `233d5737c4a94f9af87b207390c7c64938877a90a871bc9d195673a69dcac411`
35. `mi100_w4a16_marlin_M32_N4096_K4096_g128.json`  
    `e933c30f1ec666d516f0e5371648b8a39b1c95116a45d9dfea64e7a816a0a092`
36. `mi100_w4a16_marlin_M32_N4096_K4096_g32.json`  
    `bd116b2c24e23078b71026050942b37ca2d7e820233f7d13c3b820f6d623e3a2`
37. `mi100_w4a16_marlin_M512_N10240_K12288_g128.json`  
    `37fb5b54490dd0106a08d7e980d0e8dfecc3a86573176a973a2766a8755bd6fb`
38. `mi100_w4a16_marlin_M512_N10240_K12288_g32.json`  
    `b61c149e295997d0329b95cf44922f9cd716f54fe49da1016048a0b038c1c882`
39. `mi100_w4a16_marlin_M512_N10240_K4096_g128.json`  
    `77c5f005043e78b7b958f043f00ed7703c1acfa7d082cdeac9d2ea7b3a9432ce`
40. `mi100_w4a16_marlin_M512_N10240_K4096_g32.json`  
    `120a6bce7503ac1b64e5e30ff492a95a3e0eb42ef63c79dfea93f23df36f4fe6`
41. `mi100_w4a16_marlin_M512_N24576_K12288_g128.json`  
    `f2371b7714a65c419437f944295ab4ce8060f47e53d8addb5310d3f0a77654d4`
42. `mi100_w4a16_marlin_M512_N24576_K12288_g32.json`  
    `89a7e5a792863f346f949c354a0fa845a5c84b71b8a885d1ed65aded31ae30ee`
43. `mi100_w4a16_marlin_M512_N24576_K4096_g128.json`  
    `38d3ce9e44df586b60bd96bd8f0e1ffe49490e25dae2b4534dd28b6a0a8b95b3`
44. `mi100_w4a16_marlin_M512_N24576_K4096_g32.json`  
    `fe2257046875a88b41311289277a5b34700f371f8a61532e2b42191640cb4450`
45. `mi100_w4a16_marlin_M512_N4096_K12288_g128.json`  
    `be11e270c3be763ffe18e0da37ceb1f75e668e1de73cd8b6cf2058cae82fc1bc`
46. `mi100_w4a16_marlin_M512_N4096_K12288_g32.json`  
    `14340213923cfa264edda2207ed2abe37eeb63e8f69f2a35aa0edcc98a382d92`
47. `mi100_w4a16_marlin_M512_N4096_K4096_g128.json`  
    `a6abeb8b8c0a5b2d509b0c134f4ef27ba7d65c948248de914f4d0f8a73df56c6`
48. `mi100_w4a16_marlin_M512_N4096_K4096_g32.json`  
    `1baf04fdb82fe30493838a75340ba27befd47bcc4b1848dc8918f40725c71af2`

## SHA256 manifest

Rolling manifest SHA256:

```
e2f0bd1b1d2840d4a2feb98caa0bda74644948889c1decf91517c6266ab4c79d
```

Regenerate (must match the value above):

```bash
cd vllm/model_executor/kernels/configs/gfx908 && \
  sha256sum mi100_w4a16_marlin_M*.json | sort -k2 | sha256sum
```
