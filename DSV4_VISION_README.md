# DeepSeek-V4-Flash-Vision-Exp on vLLM (dsv4-vision-exp branch)

社区移植分支：让 **DeepSeek-V4-Flash-Vision-Exp**（DeepSeek 首个 V4 多模态实验模型）在 **vLLM v0.28.0** 上以 OpenAI 兼容 API 正常运行，已在与本 README 相同的软硬件配置（2× RTX PRO 6000 Blackwell，SM120，TP=2）上完整验证。

> 背景：该模型 2026-08-31 开源权重时，官方仅提供最小 PyTorch 参考推理实现（非服务引擎），vLLM/SGLang 主线尚不支持——直接启动会报
> `ValueError: There is no module or parameter named 'aligner' in DeepseekV4ForCausalLM`。
> 本分支补齐了多模态加载、hash 路由保护、模态分叉专家选择（bias_vl）等全部适配。

---

## 1. 功能与实现内容

在 vLLM v0.28.0 的 `DeepseekV4ForCausalLM`（纯文本）基础上新增视觉路径：

| 组件 | 说明 |
|---|---|
| `DeepseekV4VForConditionalGeneration` | 多模态包装类：ViT(32 层, 2D RoPE) + Aligner(3×3 合并) + 文本主干复用 |
| 哨兵 token 机制 | 每张图展开为 `vocab_size + {0..4}` 的块；embedding 屏蔽 + 引擎校验放宽 + hash 查表保护 |
| `bias_vl` 模态分叉路由 | 图像位置的专家 top-k 选择使用独立偏置（对齐官方参考实现，含 hash 层的打分路由） |
| hash-gate bias 兼容 | 视觉 checkpoint 的 hash 层携带未使用 bias，加载时显式跳过 |
| mm_prefix clamp | 图像 span 超出 SWA 窗口时保持双向区间存活（visible-window 进行时基础件） |

**未包含**（与官方参考实现的已知差异，详见"已知限制"）：
- 图像 span 内的双向注意力（当前为 causal；实测计数/属性/空间关系仍准确）
- visible-window kernel 已完成勘察与基础函数（`compute_vision_visible_window`），接线进行中

## 2. 硬件与软件要求

### 硬件（作者验证配置）

| 项目 | 要求 |
|---|---|
| GPU | 2× NVIDIA RTX PRO 6000 Blackwell 96GB（**SM120**），Tensor Parallel = 2 |
| 显存 | FP8 权重约 156 GiB，TP2 后每卡约 78 GiB + KV cache |
| 系统内存 | ≥ 128 GB（权重加载缓冲） |
| 磁盘 | 权重 168 GB + 建议预留（模型文件所在盘建议 NVMe） |

> SM121（如 DGX Spark）理论上同为 packed sparse 路径但未验证；SM100/103 走 TRTLLM-GEN 路径未验证。

### 软件

| 项目 | 版本 |
|---|---|
| OS | Ubuntu 22.04（作者环境） |
| NVIDIA 驱动 | ≥ 575（容器 CUDA 13.0.2） |
| Docker | 支持 `--gpus all` 的任意近期版本 |
| 基础镜像 | `vllm/vllm-openai:v0.28.0`（自动拉取，含全部 Python 依赖） |

**无需**本地编译：所有改动为纯 Python，通过挂载覆盖容器内 vLLM 包。

## 3. 获取模型权重

```bash
# 方式一：huggingface-cli（约 168 GB，48 个 safetensors 分片）
pip install -U "huggingface_hub[cli]"
hf download deepseek-ai/DeepSeek-V4-Flash-Vision-Exp \
  --local-dir /path/to/DeepSeek-V4-Flash-Vision-Exp

# 方式二：国内镜像
# HF_ENDPOINT=https://hf-mirror.com hf download ...
```

目录校验：48 个 `model-*.safetensors` + `config.json`（`architectures: ["DeepseekV4ForCausalLM"]`、`vision_n_layers: 32`）+ `tokenizer.json`。

## 4. 安装分支代码

### 路径 A：patch 方式（推荐，最小依赖）

```bash
# 1. 克隆 vLLM v0.28.0（GitHub 直连不通时可用镜像前缀）
git clone --depth 1 --branch v0.28.0 https://github.com/vllm-project/vllm.git
cd vllm
# 直连失败改用: git clone --depth 1 --branch v0.28.0 https://gh-proxy.com/https://github.com/vllm-project/vllm.git

# 2. 应用 patch（5 个序列 patch，或用 combined 单文件版）
#    git am 需要 git 身份，先配置：
git config user.email "you@example.com"
git config user.name "Your Name"

git am /path/to/patches/0001-*.patch
git am /path/to/patches/0002-*.patch
git am /path/to/patches/0003-*.patch
git am /path/to/patches/0004-*.patch
git am /path/to/patches/0005-*.patch
# 或: git apply dsv4-vision-exp-combined-v0.28.0.patch
```

### 路径 B：直接使用分支仓库

官方 fork（含本分支）：**https://github.com/a939984672/vllm** （分支 `dsv4-vision-exp`）

```bash
git clone --depth 1 --branch dsv4-vision-exp https://github.com/a939984672/vllm.git
```

## 5. 运行

以下假设代码在 `/opt/vllm`（按实际路径替换）、模型在 `/data/models/DeepSeek-V4-Flash-Vision-Exp`、对外端口 `6007`：

```bash
docker run -d \
  --gpus all \
  --name vllm-dsv4-vision-exp \
  -p 6007:8000 \
  -v /data/models/DeepSeek-V4-Flash-Vision-Exp:/model \
  -v /opt/vllm/vllm/models/deepseek_v4:/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4 \
  -v /opt/vllm/vllm/v1/engine/input_processor.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/input_processor.py \
  -v /opt/vllm/vllm/model_executor/models/registry.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/registry.py \
  -v /opt/vllm/vllm/v1/attention/backends/mla/sparse_swa.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/sparse_swa.py \
  --shm-size=100g \
  vllm/vllm-openai:v0.28.0 \
  /model \
  --served-model-name dsv4v \
  --hf-overrides '{"architectures":["DeepseekV4VForConditionalGeneration"],"is_mm_prefix_lm":true}' \
  --host 0.0.0.0 --port 8000 \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.96 \
  --max-model-len 1000000 \
  --max-num-seqs 4 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --structured-outputs-config '{"backend":"xgrammar"}' \
  --default-chat-template-kwargs.thinking=true \
  --default-chat-template-kwargs.reasoning_effort=max
```

启动需 3~6 分钟（权重 2 分钟 + CUDA graph / FlashInfer autotune 数分钟），直至日志出现：

```
Application startup complete.
```

> `--hf-overrides` 的两个开关缺一不可：
> ① 架构名重定向到多模态包装类；② `is_mm_prefix_lm` 启用图像区间数据源（visible-window 基础设施）。
>
> 若需节省 KV 显存，可将 `--max-model-len` 降到 262144。

## 6. 验证

```bash
# 健康
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:6007/health   # 期望 200

# 文本
curl -s http://localhost:6007/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"dsv4v","messages":[{"role":"user","content":"用一句话介绍 MoE"}],"max_tokens":200}' | head -c 400

# 图文（任意 JPEG/PNG，base64 内联）
python3 - <<'PY'
import base64, json, urllib.request
img = base64.b64encode(open("test.jpg", "rb").read()).decode()
req = urllib.request.Request("http://localhost:6007/v1/chat/completions",
    data=json.dumps({"model": "dsv4v", "max_tokens": 800, "messages": [{"role": "user", "content": [
        {"type": "text", "text": "描述这张图片。"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + img}}]}]}),
    headers={"Content-Type": "application/json"})
print(json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["message"]["content"])
PY
```

## 7. 已知限制与注意事项

1. **注意力为因果（causal）**：图像 span 内的双向注意力（官方 visible-window）接线进行中；实测计数、属性、空间关系不受影响。
2. **thinking 预算**：`thinking=true` 时 reasoning 先消耗 `max_tokens`，图文问答建议 `max_tokens ≥ 1024`（长思考链可到 2048），否则 `content` 可能为空。
3. **prefix caching**：默认开启且验证稳定；若在极端交错多图场景遇到 multimodal 区间错误，可加 `--no-enable-prefix-caching` 规避。
4. **SM120 sparse kernel 组合约束**：SWA/extra 段宽度枚举 `{128, 512, 1024}`，部分组合（如 512+512）被 kernel 拒绝——勿自行加宽 SWA 段。
5. **无 speculative decoding**：DSpark 投机与多模态组合未验证，本分支未启用。
6. **实验版模型**：模型本身为 DeepSeek 官方 Exp 版，生成质量以官方发布为准。

## 8. 性能参考（作者环境）

| 场景 | 数值 |
|---|---|
| 权重加载 | ~2 分钟（NVMe + 页缓存热） |
| 单流生成（文本，DSpark 思考模式） | ~135 tok/s |
| 图像 token 展开 | 与官方参考数学精确一致（313 / 109） |
| 多轮交错图文请求 | 15/15 稳定（prefix caching 开启） |

## 9. 目录与文件对照

```
vllm/models/deepseek_v4/vision.py          新增  ViT + Aligner
vllm/models/deepseek_v4/vision_model.py    新增  多模态包装类与 processor 注册
vllm/models/deepseek_v4/mm_preprocess.py   新增  图像预处理与位置依赖 token 展开
vllm/models/deepseek_v4/nvidia/model.py    修改  bias_vl 注册/哨兵保护/hash-gate skip
vllm/models/deepseek_v4/common/ops/cache_utils.py  修改  visible 窗口计算（基础件）
vllm/v1/engine/input_processor.py          修改  哨兵 id 校验放宽
vllm/model_executor/models/registry.py     修改  模型注册
vllm/v1/attention/backends/mla/sparse_swa.py (可选) visible-window 预计算接线
tests/models/registry.py                   修改  测试注册
docs/models/supported_models.md            修改  文档
```

## 10. 许可证

- 本分支对 vLLM 的修改遵循 vLLM 的 **Apache License 2.0**。
- DeepSeek-V4-Flash-Vision-Exp 模型权重与官方代码为 **MIT License**（模型卡见 Hugging Face）。
- 致谢：[tacos8me](https://github.com/tacos8me)（vLLM issue #54561 的初始多模态实现，本分支 0001 patch 的作者）；DeepSeek 团队（参考实现与开源权重）。
