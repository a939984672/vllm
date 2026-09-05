# DeepSeek-V4-Flash-Vision-Exp 启动优化经过与结论

> 2026-09-05 · 2× RTX PRO 6000 (SM120) · vLLM v0.28.0
> **结论先行：经过多轮优化与实测，最终确认初始版（commit 3571742）最稳定，无需任何优化改动。**

---

## 一、版本脉络

| 版本 | 内容 | 结论 |
|---|---|---|
| **3571742（初始/最终版）** | 基础多模态 + visible-window + bias_vl 路由 + 动态 SWA 行宽 | ✅ **最稳定**（600K+多图实测全过） |
| 622dc5b（回退版） | 去掉 visible-window，回退原版 flashinfer | ❌ 引入"图+长文本"必崩 |
| 社区方案（block-size 256 等） | 参考 HF discussion 18 的 SM120 配方 | ❌ decode 113→19 tok/s，不适用 |

---

## 二、优化尝试记录（均被实测否决）

### 1. 社区方案参数（-83% 性能）
参考另一位 2×RTX PRO 6000 用户的成功配方（`--block-size 256`、`--disable-custom-all-reduce`、`--max-num-batched-tokens 4096`、`--cudagraph-capture-sizes 1 2 4 8`）实测：decode 113→**19.4 tok/s**、长文本 TTFT 0.25s→1.59s。
**否决原因**：对方实现路径不同（自组装 vision + 无 custom all-reduce），参数不通用；`disable-custom-all-reduce` 拖慢 TP2 通信，`block-size 256` 与 SWA 512 宽不匹配。

### 2. 回退初始版之前的 622dc5b（最严重的错误）
为规避"多图偶发 356/357 对位"问题，整体回退到 622dc5b（丢 visible-window + patched flashinfer）。
**后果**：原生 kernel 对图 token 行在长上下文的处理缺陷被暴露——**"图 + 长文本（>128K）必崩"**（CUDA illegal memory access，定位到 layer 2 indexer）。
**教训：已验证的全链路版本不要因局部问题整体回退**——局部问题应局部修。

### 3. 排查过程中排除的假设（调试设施全部保留）
- prefix-caching 关闭 → 仍崩（排除）
- CUDA graph / Marlin c_tmp（vLLM #43730）→ enforce-eager 仍崩（排除）
- cutedsl indexer → 切 Triton 仍崩（排除，问题在更底层的图行稀疏处理）
- **最终确认：问题根因是"缺失 patched flashinfer 对图 token 行的 sparse-MLA 支持"**，而非任何启动参数。

---

## 三、最终结论

**初始版 3571742 就是最优生产版**：
- 代码即 GitHub `dsv4-vision-exp` 分支（已打 tag `final-v1.0-3571742`）
- **600K + 真实截图（2048×814）实测：101s 完成、内容正确、0 崩溃**
- 600K + 2 图：105s ✅；500K + 截图：99s ✅；纯文本 600K：103s ✅
- 开启优化前后对比：**未经优化的初始配置即为最佳**

**为什么它是"最好的"**：
1. visible-window（图 token 双向 span）+ patched flashinfer（SM120 topk=512 dual-layout）——图 token 长上下文稳定的**前提**，回退即崩
2. 动态 SWA 行宽（含图步 512 / 纯文本步 128）——长文本 TTFT 10.45s→0.25s
3. 保留 custom all-reduce + block-size 默认 + `gpu-memory-utilization 0.96`（KV 实际 121 万 token）

---

## 四、生产中正确姿势

- 不要再尝试"优化"这个版本——**参数改动只可能变差**（三项实测均劣化）
- 部署与回滚见配套文档 `dsv4v-部署指南-v1.0-3571742.md`
- 已知低概率遗留：多图（≥3 张不同尺寸）偶发对位问题（vLLM 上游 bug），重启自动恢复
