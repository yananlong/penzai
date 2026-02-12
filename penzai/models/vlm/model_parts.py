# Copyright 2024 The Penzai Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Core components for Gemma 3 vision-language models."""

from __future__ import annotations

import dataclasses

from penzai import pz


@pz.pytree_dataclass
class ConcatenateVisualAndTextTokens(pz.nn.Layer):
  """Concatenates projected visual tokens with text tokens along "seq"."""

  axis_name: str = dataclasses.field(
      default="seq", metadata={"pytree_node": False}
  )

  def __call__(
      self,
      visual_tokens: pz.nx.NamedArray,
      text_tokens: pz.nx.NamedArray,
  ) -> pz.nx.NamedArray:
    return pz.nx.concatenate([visual_tokens, text_tokens], self.axis_name)


@pz.pytree_dataclass
class Gemma3VLM(pz.nn.Layer):
  """Top-level Gemma 3 vision-language model wrapper.

  This layer wires together a vision tower, multimodal projector, token
  combiner, and the Gemma language-model body into a single Pytree-friendly
  module.

  Attributes:
    vision_tower: Layer that consumes images or patch embeddings and produces
      visual tokens.
    multimodal_projector: Layer that maps vision-tower outputs into the LM
      embedding space.
    token_combiner: Layer that merges projected visual tokens with text tokens
      to produce inputs for the LM body.
    lm_body: The Gemma language-model body that consumes the combined tokens
      and produces logits.
  """

  vision_tower: pz.nn.Layer
  multimodal_projector: pz.nn.Layer
  token_combiner: pz.nn.Layer
  lm_body: pz.nn.Layer

  def __call__(
      self,
      image: pz.nx.NamedArray,
      tokens: pz.nx.NamedArray,
      *,
      token_positions: pz.nx.NamedArray | None = None,
      **side_inputs,
  ) -> pz.nx.NamedArray:
    """Runs the vision tower, projects, merges tokens, and scores logits."""
    visual_tokens = self.vision_tower(image, **side_inputs)
    projected_visual_tokens = self.multimodal_projector(visual_tokens)
    combined_tokens = self.token_combiner(projected_visual_tokens, tokens)
    if token_positions is None:
      token_positions = pz.nx.arange("seq", combined_tokens.named_shape["seq"])
    return self.lm_body(
        combined_tokens, token_positions=token_positions, **side_inputs
    )
