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

## Catalog (80 = TP=1 + TP=4 shapes)

### TP=1 full-layer shapes (48 files)

* M in {1, 32, 128, 512}  (4)
* N in {4096, 10240, 24576}  (3)
* K in {4096, 12288}  (2)
* g in {32, 128}  (2)

4 x 3 x 2 x 2 = **48 files**.

### TP=4 shard shapes (32 files)

Derived from Qwen3.5-9B config.json (H=4096, I=12288, qkv=4096, kv_heads=1024).
At TP=4 each rank handles column-parallel (N-sharded) or row-parallel (K-sharded) GEMMs:

| layer | TP=1 (N, K) | TP=4 shard (N_shard, K) | sharding |
|-------|------------|------------------------|----------|
| qkv_proj | (4096, 4096) | **(1536, 4096)** | col-parallel N//(TP×3 heads) |
| gate_up_proj | (12288, 4096) | **(6144, 4096)** | col-parallel N//2 per rank |
| o_proj | (4096, 4096) | **(4096, 1024)** | row-parallel K//TP |
| down_proj | (4096, 12288) | **(4096, 3072)** | row-parallel K//TP |

* M in {1, 32, 128, 512}  (4)
* (N, K) in {(1536,4096), (6144,4096), (4096,1024), (4096,3072)}  (4)
* g in {32, 128}  (2)

4 x 4 x 2 = **32 files**.

**Total: 48 + 32 = 80 files**. Tile selection depends only on (M, g); N and K
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

### TP=1 full-layer shapes (48 files)

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

### TP=4 shard shapes (32 files, added commit 7df913bba)

49. `mi100_w4a16_marlin_M128_N1536_K4096_g128.json`  
    `d6964e8c4d454607f95fe5ce1d36b12d03ebb5d3e657ebc815e03fa6f2d9a9bc`
50. `mi100_w4a16_marlin_M128_N1536_K4096_g32.json`  
    `8e51b14c22cd07fc84536e872fdc69a82fa5e517f1968511745739488f55a888`
51. `mi100_w4a16_marlin_M128_N4096_K1024_g128.json`  
    `1d4fdc303c216d05ffd92695ab145993446b693e7ed87e5eee27854014bf5fbb`
52. `mi100_w4a16_marlin_M128_N4096_K1024_g32.json`  
    `babf5987b3240217d08fdf4b125b8ff0a576e76e40f5e7020ac74c2b7c1e9c80`
53. `mi100_w4a16_marlin_M128_N4096_K3072_g128.json`  
    `a1764f2237a8e35b15cd168b54cb8f3dd3eda2ca7a576b6c0a25b677434cc0b1`
54. `mi100_w4a16_marlin_M128_N4096_K3072_g32.json`  
    `94dd71cdce81760c552492b4a7aad8109d93f11064ed622513f713b8ce897449`
55. `mi100_w4a16_marlin_M128_N6144_K4096_g128.json`  
    `217bd1d6334418565822da552e8753400f42a51b59ebc8bf2d6844719575c576`
56. `mi100_w4a16_marlin_M128_N6144_K4096_g32.json`  
    `dc36bb7a38518f9249f095d2d6b094a5e065c71416bfbe6d8a39085ff9f356ba`
57. `mi100_w4a16_marlin_M1_N1536_K4096_g128.json`  
    `11a2375fdc873bb24ab8223d85daf3919c603a9701847a9d689601909a2fa605`
58. `mi100_w4a16_marlin_M1_N1536_K4096_g32.json`  
    `7924cf5be2122f328b3e38fc775aa174c79d2d2e3f368c74aafa69b4f466da1f`
59. `mi100_w4a16_marlin_M1_N4096_K1024_g128.json`  
    `ea8e1293137068cc424ceee8d4ed8ced99d507345a1863d761ed87064f01a8d6`
60. `mi100_w4a16_marlin_M1_N4096_K1024_g32.json`  
    `94a0a49a9aa5acfeb0e459d2bbac0a48ee3518555356aa93737d817754605f62`
61. `mi100_w4a16_marlin_M1_N4096_K3072_g128.json`  
    `ed9a28e304bf84443e574d4333cfd0a268710f357de34679fb9b2c374b97adb5`
62. `mi100_w4a16_marlin_M1_N4096_K3072_g32.json`  
    `f6b5efae8c4efdd2945adc019ecf6e0497f6bf268f8ada2a6450e110fd62bc1b`
63. `mi100_w4a16_marlin_M1_N6144_K4096_g128.json`  
    `f7b8605bec8104aa510cd317cf84c6e89628a095a271dce3cc89f243812739cd`
64. `mi100_w4a16_marlin_M1_N6144_K4096_g32.json`  
    `6fca9efbf7bafc6ddd95a427809f93915f3458a0d80894e46905ef6b992493af`
65. `mi100_w4a16_marlin_M32_N1536_K4096_g128.json`  
    `105cc8147133918743f299858b51a223cb0a388279d28b63ca21ed7bfee24b19`
66. `mi100_w4a16_marlin_M32_N1536_K4096_g32.json`  
    `b5d4f3bb31d89c85fcdbe29209e454480e7438c8de25eb049e29657818c87b93`
67. `mi100_w4a16_marlin_M32_N4096_K1024_g128.json`  
    `f7dd2d5734eb8d2759f20da67e790d7cb5dcd0f28f46a637d9c0a6b97d319377`
68. `mi100_w4a16_marlin_M32_N4096_K1024_g32.json`  
    `ff280da19f6c1bb359d5d79fc66b442127ed1ba24a3216976b5b488a259ecd7b`
69. `mi100_w4a16_marlin_M32_N4096_K3072_g128.json`  
    `de7cf4319ff714b8ce47f50e4b38e8af4425373c967c9c27088c94048f69b0f1`
70. `mi100_w4a16_marlin_M32_N4096_K3072_g32.json`  
    `7f927b530ec7dbce23bdf5a71f71a1e2af657f9909343c485f1bf5ee3cb19379`
71. `mi100_w4a16_marlin_M32_N6144_K4096_g128.json`  
    `50c0946928ca2635434b22fbbcee2d75cab943496c3094a683491640c6fbb905`
72. `mi100_w4a16_marlin_M32_N6144_K4096_g32.json`  
    `85dad9d3af065f656601a87c8c68a0988cd2c3b6580e7b443ed2bc9fc55e731c`
73. `mi100_w4a16_marlin_M512_N1536_K4096_g128.json`  
    `918735c0d0d834b51e43d82b96900457abf1d6292705302e25f473cb546a09b7`
74. `mi100_w4a16_marlin_M512_N1536_K4096_g32.json`  
    `8722bbfef4672b28951016c72523343bceb4f428ce85de4791dbdd26e959bdd4`
75. `mi100_w4a16_marlin_M512_N4096_K1024_g128.json`  
    `6f15df6506f21bc7b56bb023d202f2ec4dd05bb426b5e7caa1c8025818090e2f`
76. `mi100_w4a16_marlin_M512_N4096_K1024_g32.json`  
    `5ed50e8f087f3dd8a28fbcf7073fa6922fe15eb25cc2a99166342283d0ebfe72`
77. `mi100_w4a16_marlin_M512_N4096_K3072_g128.json`  
    `a2658865c3fa2713160bfab96344a2591a3b6432637bba67ae21ae1de6e66594`
78. `mi100_w4a16_marlin_M512_N4096_K3072_g32.json`  
    `c0451c6e254590ec16876e968c0089e116705273b3014029a802976ba3f24e22`
79. `mi100_w4a16_marlin_M512_N6144_K4096_g128.json`  
    `17e1589a664f48de1b60c6538ca3d5307c5166a32d14a75119a901ed3ea72da3`
80. `mi100_w4a16_marlin_M512_N6144_K4096_g32.json`  
    `00eb210f4dbb6adfcb449c7d81464db9d09000323c237c1651ec907bd3e4c324`

## SHA256 manifest

Rolling manifest SHA256 (80 files):

```
e6a933f108ac6e21261ab2e20bc53a3c856e2a8dc953543e1197cce3ec89ecce
```

Regenerate (must match the value above):

```bash
cd vllm/model_executor/kernels/configs/gfx908 && \
  sha256sum mi100_w4a16_marlin_M*.json | sort -k2 | sha256sum
```
