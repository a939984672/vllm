# DeepSeek-V4-Flash-Vision-Exp 部署指南 v1.0（最终生产版 3571742）

> **版本定位**：本指南描述的 `3571742` 已实测为**最终生产基线**（600K+多图长上下文稳定），代码与 GitHub `a939984672/vllm` 分支 `dsv4-vision-exp`（commit `3571742`）完全一致。
> **更新日期**：2026-09-05
> **实测环境**：2× NVIDIA RTX PRO 6000 Blackwell（SM120, 96GB），Ubuntu 22.04/24.04，vLLM v0.28.0 镜像

---

## 1. 版本标识与备份

| 项 | 值 |
|---|---|
| 模型 | `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`（~285B，FP4+FP8 混合） |
| 推理镜像 | `vllm/vllm-openai:v0.28.0` |
| 代码基线 | git commit `3571742`（tag：`final-v1.0-3571742`） |
| 备份位置（服务器） | `~/backups/dsv4v-final-3571742/`（挂载文件 490K + patched flashinfer 源码 14M） |
| GitHub 同步 | `dsv4-vision-exp` 分支 = `3571742`（内容已验证一致） |

**挂载文件清单**（`3571742` 相对 vLLM v0.28.0 镜像的全部改动）：
- `vllm/models/deepseek_v4/`（模型实现：visible-window 双向注意力 + bias_vl 路由 + 动态 SWA 行宽）
- `vllm/v1/engine/input_processor.py`（多模态输入处理）
- `vllm/model_executor/models/registry.py`（模型注册）
- `vllm/v1/attention/backends/mla/sparse_swa.py`（稀疏 SWA：含图步 512 宽 / 纯文本步 128 宽）
- `vllm/v1/worker/gpu_model_runner.py`（步级图像标志）
- **flashinfer 补丁**（`docs/flashinfer_patches/flashinfer-sm120-tk512-dual.patch`）：SM120 sparse-MLA 增加 `topk=512` dual-layout 支持（图 token 长上下文必需）

---

## 2. 前置准备

### 2.1 模型权重
```bash
# 服务器路径（已就绪）
/mnt/ai-storage/models/DeepSeek-V4-Flash-Vision-Exp/
```

### 2.2 flashinfer 补丁（关键，图 token 长上下文必需）
补丁内容与 wheel 应用方法见分支内 `docs/flashinfer_patches/flashinfer-sm120-tk512-dual.patch` 及 `DSV4_VISION_README.md`。

本部署使用**免重编方案**（推荐）：
- 把 patched flashinfer 源码放 `/home/e/fi-src/flashinfer`（容器挂载覆盖）
- AOT 缓存目录用空目录挂载免疫：`/home/e/fi-src/aot_disabled/sparse_mla_sm120`（防止预编译 AOT 遮蔽 patched 源码）

> ⚠️ 不用 patched flashinfer 会导致"图 + 长文本（>128K）必崩"（原版 kernel 对图 token 行的处理缺陷）。本版已实测规避。

---

## 3. 部署（docker 启动命令）

### 3.1 标准启动（生产，6006 端口 / 模型名 dsv4v）

```bash
docker run -d \
  --gpus all --name vllm-dsv4-vision-exp -p 6006:8000 \
  --restart unless-stopped \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -v /mnt/ai-storage/models/DeepSeek-V4-Flash-Vision-Exp:/model \
  -v /home/e/dev/vllm/vllm/models/deepseek_v4:/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4 \
  -v /home/e/dev/vllm/vllm/v1/engine/input_processor.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/input_processor.py \
  -v /home/e/dev/vllm/vllm/model_executor/models/registry.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/registry.py \
  -v /home/e/dev/vllm/vllm/v1/attention/backends/mla/sparse_swa.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/sparse_swa.py \
  -v /home/e/dev/vllm/vllm/v1/worker/gpu_model_runner.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py \
  -v /home/e/fi-src/flashinfer:/usr/local/lib/python3.12/dist-packages/flashinfer \
  -v /home/e/fi-src/aot_disabled/sparse_mla_sm120:/usr/local/lib/python3.12/dist-packages/flashinfer_jit_cache/jit_cache/sparse_mla_sm120 \
  --log-driver json-file --log-opt max-size=100m --log-opt max-file=5 \
  --shm-size=100g vllm/vllm-openai:v0.28.0 \
  /model --served-model-name dsv4v \
  --hf-overrides '{"architectures":["DeepseekV4VForConditionalGeneration"],"is_mm_prefix_lm":true}' \
  --host 0.0.0.0 --port 8000 --trust-remote-code --tensor-parallel-size 2 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.96 --max-model-len 1000000 --max-num-seqs 4 \
  --enable-prefix-caching --enable-chunked-prefill \
  --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --structured-outputs-config '{"backend":"xgrammar"}' \
  --default-chat-template-kwargs.thinking=true \
  --default-chat-template-kwargs.reasoning_effort=max
```

### 3.2 关键参数说明

| 参数 | 说明 |
|---|---|
| 8 个 `-v` 挂载 | 5 个代码挂载 + patched flashinfer + AOT 免疫目录 + 模型（**缺一不可**，尤其 flashinfer 两个） |
| `--max-model-len 1000000` | 1M 上下文上限；实际 KV 121 万 token（实测 600K+ 稳定） |
| `--gpu-memory-utilization 0.96` | KV 自动分配（避开手动 `kv-cache-memory-bytes` 的上游容量计算 bug） |
| `--kv-cache-dtype fp8` | DeepSeek V4 的 fp8_ds_mla KV 布局 |
| `--restart unless-stopped` | 崩溃自动拉起（vLLM 偶发 EngineDead 兜底） |
| 日志轮转 | 100MB×5，崩溃现场不丢 |

### 3.3 启动耗时
约 8~10 分钟（TileLang/FlashInfer JIT 编译），`Application startup complete` 后 `:6006/health` 返回 200 即就绪。

---

## 4. 验证清单（实测通过）

| 场景 | 预期 | 实测 |
|---|---|---|
| 基础图文（单图颜色/内容） | 正确回答 | ✅ |
| 纯文本 600K | 完成（~103s） | ✅ 610,509 tokens |
| **600K + 真实截图（2048×814）** | 完成且内容正确 | ✅ 101s，0 崩溃 |
| 600K + 2 图 | 完成 | ✅ 105s |
| 500K + 截图 | 完成 | ✅ 99s |

调用示例（OpenAI 兼容）：
```bash
curl -X POST http://localhost:6006/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"dsv4v","messages":[{"role":"user","content":[{"type":"text","text":"图里有什么？"},{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}]}],"max_tokens":100}'
```

---

## 5. 已知限制与运维

### 5.1 已知限制（低优先级）
- **多图（≥3 张不同尺寸同会话）偶发** `Attempted to assign ... to N+1 placeholders`（vLLM 异步层对位问题，vLLM 上游 bug，频率低）——重启自动恢复，可临时把图分批发送规避
- 客户端请勿中断流式请求（vLLM 已知 `shm_broadcast cancelled` 引擎自杀 bug）——客户端超时调长即可

### 5.2 运维
- 日志：`docker logs vllm-dsv4-vision-exp`（轮转保留最近 500MB）
- 重启：`docker restart vllm-dsv4-vision-exp`（约 10 分钟就绪）
- 端口：`6006`（`--network host` 变体则为 `8000`，注意区分）

---

## 6. 回滚与恢复

- 代码恢复到本版：`cd ~/dev/vllm && git checkout final-v1.0-3571742 -- vllm/`
- 完整备份：`~/backups/dsv4v-final-3571742/`（挂载文件 + flashinfer 源码）
- GitHub：`dsv4-vision-exp` 分支即本版，随时可 fetch
