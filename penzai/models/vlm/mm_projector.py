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

"""Gemma 3 multi-modal projector and token insertion helpers."""

from __future__ import annotations

import dataclasses
from typing import Literal

import jax
import jax.numpy as jnp
from penzai import pz
from penzai.nn import linear_and_affine
from penzai.nn import parameters


GEMMA3_VISUAL_INSERTION_MODE: Literal["prepend"] = "prepend"


def _soft_emb_norm_initializer(
    key: jax.Array, *, embedding_dim: int, dtype: jax.typing.DTypeLike
) -> pz.nx.NamedArray:
  del key
  return pz.nx.full({"embedding": embedding_dim}, 1.0, dtype=dtype)


@pz.pytree_dataclass
class Gemma3MultiModalProjector(pz.nn.Layer):
  """Projects vision tower outputs into the LM embedding space.

  This module mirrors Gemma 3 multimodal checkpoints, which store parameters
  under the ``multi_modal_projector.*`` prefix. The projection weight is stored
  in ``mm_input_projection_weight`` and the post-projection RMS scaling weight
  is stored in ``mm_soft_emb_norm.weight``.
  """

  input_projection: pz.nn.Linear
  soft_emb_norm_scale: parameters.ParameterLike[pz.nx.NamedArray]
  epsilon: float = dataclasses.field(default=1e-6, metadata={"pytree_node": False})
  vision_axis: str = dataclasses.field(default="vision_embedding", metadata={"pytree_node": False})
  embedding_axis: str = dataclasses.field(default="embedding", metadata={"pytree_node": False})

  def __call__(
      self, vision_outputs: pz.nx.NamedArray, **_unused_side_inputs
  ) -> pz.nx.NamedArray:
    """Projects and normalizes vision outputs into LM embeddings."""
    if self.vision_axis not in vision_outputs.named_shape:
      raise ValueError(
          "Vision outputs must include the"
          f" {self.vision_axis!r} axis, got"
          f" {vision_outputs.named_shape}."
      )
    projected = self.input_projection(vision_outputs)
    if self.embedding_axis not in projected.named_shape:
      raise ValueError(
          "Projected vision outputs must include the"
          f" {self.embedding_axis!r} axis, got"
          f" {projected.named_shape}."
      )
    standardized = projected.untag(self.embedding_axis)

    @pz.nx.nmap
    def _rms_scale(values: jax.Array) -> jax.Array:
      mean_sq = jnp.mean(jnp.square(values), axis=-1, keepdims=True)
      return values * jax.lax.rsqrt(mean_sq + self.epsilon)

    normalized = _rms_scale(standardized).tag(self.embedding_axis)
    return normalized * self.soft_emb_norm_scale.value

  @classmethod
  def from_config(
      cls,
      *,
      init_base_rng: jax.Array | None,
      vision_embedding_dim: int,
      embedding_dim: int,
      dtype: jax.typing.DTypeLike = jnp.float32,
      epsilon: float = 1e-6,
      vision_axis: str = "vision_embedding",
      embedding_axis: str = "embedding",
  ) -> "Gemma3MultiModalProjector":
    """Constructs the projector with Gemma 3 parameter names."""
    projection_weights = parameters.make_parameter(
        "multi_modal_projector/mm_input_projection_weight",
        init_base_rng,
        linear_and_affine.xavier_uniform_initializer,
        input_axes={vision_axis: vision_embedding_dim},
        output_axes={embedding_axis: embedding_dim},
        parallel_axes={},
        convolution_spatial_axes={},
        dtype=dtype,
    )
    scale = parameters.make_parameter(
        "multi_modal_projector/mm_soft_emb_norm.weight",
        init_base_rng,
        _soft_emb_norm_initializer,
        embedding_dim=embedding_dim,
        dtype=dtype,
    )
    return cls(
        input_projection=pz.nn.Linear(
            weights=projection_weights,
            in_axis_names=(vision_axis,),
            out_axis_names=(embedding_axis,),
        ),
        soft_emb_norm_scale=scale,
        epsilon=epsilon,
        vision_axis=vision_axis,
        embedding_axis=embedding_axis,
    )


def insert_visual_tokens(
    text_embeddings: pz.nx.NamedArray,
    visual_embeddings: pz.nx.NamedArray,
    *,
    insertion: Literal["prepend", "concat"] = GEMMA3_VISUAL_INSERTION_MODE,
    text_axis: str = "seq",
    visual_axis: str = "vision_seq",
    embedding_axis: str = "embedding",
) -> pz.nx.NamedArray:
  """Combines visual and text token embeddings.

  Gemma 3 multimodal tokenization inserts a fixed block of visual tokens before
  the text tokens, so the default insertion mode prepends the visual tokens to
  the text sequence.
  """
  if text_axis not in text_embeddings.named_shape:
    raise ValueError(
        f"Text embeddings must include the {text_axis!r} axis, got"
        f" {text_embeddings.named_shape}."
    )
  if visual_axis not in visual_embeddings.named_shape:
    raise ValueError(
        f"Visual embeddings must include the {visual_axis!r} axis, got"
        f" {visual_embeddings.named_shape}."
    )
  if embedding_axis not in text_embeddings.named_shape:
    raise ValueError(
        f"Text embeddings must include the {embedding_axis!r} axis, got"
        f" {text_embeddings.named_shape}."
    )
  if embedding_axis not in visual_embeddings.named_shape:
    raise ValueError(
        f"Visual embeddings must include the {embedding_axis!r} axis, got"
        f" {visual_embeddings.named_shape}."
    )
  if (
      text_embeddings.named_shape[embedding_axis]
      != visual_embeddings.named_shape[embedding_axis]
  ):
    raise ValueError(
        "Visual and text embeddings must share the same embedding dimension,"
        f" got {text_embeddings.named_shape[embedding_axis]} and"
        f" {visual_embeddings.named_shape[embedding_axis]}."
    )

  if visual_axis != text_axis:
    visual_embeddings = visual_embeddings.untag(visual_axis).tag(text_axis)

  if insertion == "prepend":
    ordered = [visual_embeddings, text_embeddings]
  elif insertion == "concat":
    ordered = [text_embeddings, visual_embeddings]
  else:
    raise ValueError(f"Unsupported insertion mode: {insertion!r}.")
  return pz.nx.concatenate(ordered, text_axis)
