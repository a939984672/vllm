# SPDX-License-Identifier: Apache-2.0
"""Multimodal wrapper for DeepSeek-V4-Flash-Vision-Exp.

Wraps the existing text-only DeepseekV4ForCausalLM with the checkpoint's ViT +
aligner and the sentinel-block embedding layout.

The image block is emitted as out-of-vocabulary token ids (vocab_size + type,
i.e. 129280..129284). Those ids never index the embedding table -- the LM reads
them for MoE routing and attention visibility -- so every position in the block
is a "multimodal" position and embed_multimodal returns the whole block: the
learned image_start/pad/newline/end vectors in their slots and aligner outputs
at IMAGE slots, permuted into the N-layout by `perm`.
"""

from collections.abc import Iterable, Mapping

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors

from vllm.models.deepseek_v4.mm_preprocess import (
    IMAGE,
    IMAGE_PLACEHOLDER,
    DeepseekV4VDummyInputsBuilder,
    DeepseekV4VMultiModalProcessor,
    DeepseekV4VProcessingInfo,
)
from vllm.models.deepseek_v4.vision import DeepseekV4Aligner, DeepseekV4ViT


@MULTIMODAL_REGISTRY.register_processor(
    DeepseekV4VMultiModalProcessor,
    info=DeepseekV4VProcessingInfo,
    dummy_inputs=DeepseekV4VDummyInputsBuilder,
)
class DeepseekV4VForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP):
    # The LM's MoE gate selects `bias_vl` over `bias` per position and the hash
    # layers index tid2eid by token id, so the raw ids must reach forward().
    requires_raw_input_tokens = True

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "layers.": "language_model.layers.",
            "embed.": "language_model.embed.",
            "norm.": "language_model.norm.",
            "hc_head": "language_model.hc_head",
            "mtp.": "language_model.mtp.",
        },
        orig_to_new_suffix={"head.weight": "language_model.head.weight"},
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return IMAGE_PLACEHOLDER
        return None

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config

        with self._mark_tower_model(vllm_config, "image"):
            # Plain nn.Modules: never handed a quant_config, so the BF16 tower
            # cannot be picked up by DeepseekV4FP8Config.
            self.vision = DeepseekV4ViT(config)
            self.aligner = DeepseekV4Aligner(config)
            hidden = config.hidden_size
            self.image_start = nn.Parameter(torch.empty(hidden))
            self.image_pad = nn.Parameter(torch.empty(hidden))
            self.image_newline = nn.Parameter(torch.empty(hidden))
            self.image_end = nn.Parameter(torch.empty(hidden))

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["DeepseekV4ForCausalLM"],
            )

        # 129280..129284 are >= vocab_size; this makes the default
        # embed_input_ids mask them to 0 before the text-embedding gather.
        vocab_size = config.vocab_size
        self.configure_mm_token_handling(
            vocab_size, [vocab_size + i for i in range(5)])

    def _encode_one(self, patches, n_vit_h, n_vit_w, types, perm):
        feats = self.vision(patches, n_vit_h, n_vit_w)
        embeds = self.aligner(feats, n_vit_h, n_vit_w)[perm]
        params = torch.stack([
            self.image_start, self.image_pad, self.image_pad,
            self.image_newline, self.image_end,
        ]).to(embeds.dtype)
        block = params[types]
        block[types == IMAGE] = embeds.to(block.dtype)
        return block

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings | None:
        patches = kwargs.get("patches")
        if patches is None:
            return None
        patch_sizes = kwargs["patch_sizes"]
        types = kwargs["types"]
        type_sizes = kwargs["type_sizes"]
        perm = kwargs["perm"]
        perm_sizes = kwargs["perm_sizes"]
        n_vit = kwargs["n_vit"]

        def _flat(x):
            return torch.cat([t.flatten() for t in x]) if isinstance(x, (list, tuple)) else x.flatten()

        patch_sizes = _flat(patch_sizes).tolist()
        type_sizes = _flat(type_sizes).tolist()
        perm_sizes = _flat(perm_sizes).tolist()
        if isinstance(patches, (list, tuple)):
            patches = torch.cat([p for p in patches], dim=0)
        if isinstance(types, (list, tuple)):
            types = torch.cat([t.flatten() for t in types], dim=0)
        if isinstance(perm, (list, tuple)):
            perm = torch.cat([p.flatten() for p in perm], dim=0)
        if isinstance(n_vit, (list, tuple)):
            n_vit = torch.cat([n.reshape(-1, 2) for n in n_vit], dim=0)
        n_vit = n_vit.reshape(-1, 2)

        device = next(self.vision.parameters()).device
        out, po, to, ro = [], 0, 0, 0
        for i, np_ in enumerate(patch_sizes):
            ts, rs = type_sizes[i], perm_sizes[i]
            out.append(self._encode_one(
                patches[po:po + np_].to(device),
                int(n_vit[i][0]), int(n_vit[i][1]),
                types[to:to + ts].to(device),
                perm[ro:ro + rs].to(device),
            ))
            po += np_; to += ts; ro += rs
        return out

    def forward(self, input_ids, positions, intermediate_tensors=None,
                inputs_embeds=None, **kwargs):
        return self.language_model(
            input_ids=input_ids, positions=positions,
            intermediate_tensors=intermediate_tensors, inputs_embeds=inputs_embeds)

    def compute_logits(self, hidden_states, *args, **kwargs):
        return self.language_model.compute_logits(hidden_states, *args, **kwargs)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="aligner",
            tower_model="vision",
        )
