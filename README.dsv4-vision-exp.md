# dsv4-vision-exp — DeepSeek-V4-Flash-Vision-Exp on vLLM

> 这是 **`vLLM` 的 `dsv4-vision-exp` 分支**，为在 **vLLM v0.28.0** 上运行 DeepSeek 首个 V4 多模态实验模型 **DeepSeek-V4-Flash-Vision-Exp** 而做。
>
> ⚠️ **本分支并非 vLLM 官方主线**。它是开源社区在 v0.28.0 基线上的多模态支持移植，用独立分支形式发布，已在与作者相同的硬件上完整验证。

## 这个分支解决什么问题？

DeepSeek-V4-Flash-Vision-Exp（MIT，2026-08-31 开源，约 168 GB 权重）发布时，vLLM/SGLang 主线尚未支持——直接用 vLLM 加载会报：

```
ValueError: There is no module or parameter named 'aligner' in DeepseekV4ForCausalLM
```

本分支补齐了多模态加载与推理所需的全部适配，让模型能以 OpenAI 兼容 API 正常运行。

## 已有功能

- 图片理解（ViT + Aligner 视觉前端接入）
- 图像 token 的哨兵机制（`vocab_size+{0..4}` 块展开）
- `bias_vl` 模态分叉专家路由（对齐官方参考实现）
- 图文 / 多图 / 计数 / 属性问答
- SM120 的 sparse-MLA 路径（`vllm/vllm-openai:v0.28.0` 镜像）

## 硬件与软件要求

- **GPU**：2× NVIDIA RTX PRO 6000 Blackwell 96GB（**SM120**），TP=2（作者验证配置）
- 系统内存 ≥128 GB、磁盘约 170 GB（权重）、NVIDIA 驱动 ≥575、Docker
- 基础镜像：`vllm/vllm-openai:v0.28.0`（免本地编译，纯 Python 挂载）

## 快速开始

```bash
# 1. 克隆本分支（已含全部改动）
git clone --depth 1 --branch dsv4-vision-exp https://github.com/a939984672/vllm.git
cd vllm
# 2. 下载模型权重（约 168 GB）
hf download deepseek-ai/DeepSeek-V4-Flash-Vision-Exp --local-dir /data/models/DeepSeek-V4-Flash-Vision-Exp
# 3. 运行容器（路径按实际替换）
docker run -d --gpus all --name dsv4-vision-exp -p 6007:8000 \
  -v /data/models/DeepSeek-V4-Flash-Vision-Exp:/model \
  -v $(pwd)/vllm/models/deepseek_v4:/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4 \
  -v $(pwd)/vllm/v1/engine/input_processor.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/input_processor.py \
  -v $(pwd)/vllm/model_executor/models/registry.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/registry.py \
  -v $(pwd)/vllm/v1/attention/backends/mla/sparse_swa.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/sparse_swa.py \
  --shm-size=100g vllm/vllm-openai:v0.28.0 /model \
  --served-model-name dsv4v \
  --hf-overrides '{"architectures":["DeepseekV4VForConditionalGeneration"],"is_mm_prefix_lm":true}' \
  --host 0.0.0.0 --port 8000 --trust-remote-code --tensor-parallel-size 2 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.96 --max-model-len 1000000 --max-num-seqs 4 \
  --enable-prefix-caching --enable-chunked-prefill \
  --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --structured-outputs-config '{"backend":"xgrammar"}' \
  --default-chat-template-kwargs.thinking=true --default-chat-template-kwargs.reasoning_effort=max
```

## 更多说明

- **完整文档（建议收藏）**：见上游代码库中本分支的 `DSV4_VISION_README.md`，或直接访问 [完整 README](https://github.com/a939984672/vllm/blob/dsv4-vision-exp/DSV4_VISION_README.md)
- **Patch 发布包**：`dsv4-vision-exp-release-20260901.tar.gz`（含 5 个 patch、LICENSE、测试脚本）
- **验证**：`curl http://localhost:6007/health`；图文请求 `python3 scripts/test_vision.py --port 6007 --image <图片>`

## 致谢与许可

- vLLM 代码改动遵循 **Apache License 2.0**；模型与官方参考代码为 **MIT License**
- 感谢 [tacos8me](https://github.com/tacos8me)（vLLM issue #54561 初始多模态实现）与 DeepSeek 团队

---

*本分支由社区维护，非 vLLM 官方发布。已知限制与 roadmap 见完整 README 的 "Known limitations" 小节。*
