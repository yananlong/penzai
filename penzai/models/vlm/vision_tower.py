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

"""Vision tower components for Gemma-3 VLM models."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

from penzai import pz


@dataclasses.dataclass(frozen=True)
class Gemma3VisionConfig:
  """Configuration for the Gemma-3 vision encoder.

  Mirrors the Hugging Face Gemma-3 vision tower config fields.

  Attributes:
    image_size: Size (height/width) of the input image in pixels.
    patch_size: Size of each patch in pixels.
    num_channels: Number of input image channels.
    hidden_size: Embedding dimension for vision tokens.
    num_hidden_layers: Number of transformer blocks.
    num_attention_heads: Number of attention heads.
    intermediate_size: Hidden size of the MLP blocks.
    layer_norm_eps: Epsilon for layer normalization.
    parameter_dtype: Dtype used for parameters.
    activation_dtype: Dtype used for activations.
  """

  image_size: int
  patch_size: int
  num_channels: int
  hidden_size: int
  num_hidden_layers: int
  num_attention_heads: int
  intermediate_size: int
  layer_norm_eps: float = 1e-6
  parameter_dtype: jax.typing.DTypeLike = jnp.float32
  activation_dtype: jax.typing.DTypeLike = jnp.float32

  def __post_init__(self) -> None:
    if self.image_size % self.patch_size != 0:
      raise ValueError(
          "image_size must be divisible by patch_size ("
          f"{self.image_size} vs {self.patch_size})."
      )
    if self.hidden_size % self.num_attention_heads != 0:
      raise ValueError(
          "hidden_size must be divisible by num_attention_heads ("
          f"{self.hidden_size} vs {self.num_attention_heads})."
      )

  @property
  def num_patches(self) -> int:
    patches_per_side = self.image_size // self.patch_size
    return patches_per_side * patches_per_side

  @property
  def projection_dim(self) -> int:
    return self.hidden_size // self.num_attention_heads


@pz.pytree_dataclass
class VisionPatchEmbedding(pz.nn.Layer):
  """Patch embedding layer for vision inputs."""

  patch_size: int = dataclasses.field(metadata={"pytree_node": False})
  num_channels: int = dataclasses.field(metadata={"pytree_node": False})
  projection: pz.nn.Layer

  def __call__(
      self, images: pz.nx.NamedArray, **_unused_side_inputs
  ) -> pz.nx.NamedArray:
    height = images.named_shape.get("height")
    width = images.named_shape.get("width")
    channels = images.named_shape.get("channels")
    if height is None or width is None or channels is None:
      raise ValueError(
          "images must have named axes 'height', 'width', and 'channels'."
      )
    if channels != self.num_channels:
      raise ValueError(
          "channels axis does not match config num_channels: "
          f"{channels} vs {self.num_channels}."
      )
    if height % self.patch_size != 0 or width % self.patch_size != 0:
      raise ValueError(
          "Image dimensions must be divisible by patch_size ("
          f"height={height}, width={width}, patch_size={self.patch_size})."
      )
    patch_dim = self.patch_size * self.patch_size * self.num_channels

    def _to_patches(arr: jax.Array) -> jax.Array:
      patch_rows = height // self.patch_size
      patch_cols = width // self.patch_size
      arr = arr.reshape(
          patch_rows,
          self.patch_size,
          patch_cols,
          self.patch_size,
          self.num_channels,
      )
      arr = arr.transpose(0, 2, 1, 3, 4)
      return arr.reshape(patch_rows * patch_cols, patch_dim)

    patches = (
        pz.nx.nmap(_to_patches)(
            images.untag("height", "width", "channels")
        )
        .tag("seq", "patch")
    )
    return self.projection(patches)


@pz.pytree_dataclass
class AddPositionEmbedding(pz.nn.Layer):
  """Adds learnable position embeddings to a sequence."""

  lookup: pz.nn.EmbeddingLookup

  def __call__(
      self, tokens: pz.nx.NamedArray, **_unused_side_inputs
  ) -> pz.nx.NamedArray:
    positions = pz.nx.arange("seq", tokens.named_shape["seq"])
    pos_embeddings = self.lookup(positions)
    return tokens + pos_embeddings


@pz.pytree_dataclass
class Gemma3VisionBlock(pz.nn.Sequential):
  """Transformer block used in the Gemma-3 vision tower."""


def _build_attention(
    name: str,
    init_base_rng: jax.Array | None,
    config: Gemma3VisionConfig,
) -> pz.nn.Attention:
  projection_dim = config.projection_dim
  num_heads = config.num_attention_heads
  return pz.nn.Attention(
      input_to_query=pz.nn.Sequential([
          pz.nn.Affine.from_config(
              name=f"{name}/query",
              init_base_rng=init_base_rng,
              input_axes={"embedding": config.hidden_size},
              output_axes={"heads": num_heads, "projection": projection_dim},
              dtype=config.parameter_dtype,
          ),
          pz.nn.ConstantRescale(
              by=jnp.array(projection_dim**-0.5, dtype=config.activation_dtype)
          ),
      ]),
      input_to_key=pz.nn.Sequential([
          pz.nn.Affine.from_config(
              name=f"{name}/key",
              init_base_rng=init_base_rng,
              input_axes={"embedding": config.hidden_size},
              output_axes={"heads": num_heads, "projection": projection_dim},
              dtype=config.parameter_dtype,
          ),
      ]),
      input_to_value=pz.nn.Sequential([
          pz.nn.Affine.from_config(
              name=f"{name}/value",
              init_base_rng=init_base_rng,
              input_axes={"embedding": config.hidden_size},
              output_axes={"heads": num_heads, "projection": projection_dim},
              dtype=config.parameter_dtype,
          ),
      ]),
      query_key_to_attn=pz.nn.Sequential([
          pz.nn.NamedEinsum(
              (
                  {"seq": "tq", "heads": "h", "projection": "p"},
                  {"seq": "tkv", "heads": "h", "projection": "p"},
              ),
              {"seq": "tq", "heads": "h", "kv_seq": "tkv"},
          ),
          pz.nn.Softmax("kv_seq"),
      ]),
      attn_value_to_output=pz.nn.Sequential([
          pz.nn.NamedEinsum(
              (
                  {"seq": "tq", "heads": "h", "kv_seq": "tkv"},
                  {"seq": "tkv", "heads": "h", "projection": "p"},
              ),
              {"seq": "tq", "heads": "h", "projection": "p"},
          ),
          pz.nn.Affine.from_config(
              name=f"{name}/output",
              init_base_rng=init_base_rng,
              input_axes={"heads": num_heads, "projection": projection_dim},
              output_axes={"embedding": config.hidden_size},
              dtype=config.parameter_dtype,
          ),
      ]),
  )


def _build_mlp(
    name: str,
    init_base_rng: jax.Array | None,
    config: Gemma3VisionConfig,
) -> pz.nn.Sequential:
  return pz.nn.Sequential([
      pz.nn.Affine.from_config(
          name=f"{name}/in",
          init_base_rng=init_base_rng,
          input_axes={"embedding": config.hidden_size},
          output_axes={"neurons": config.intermediate_size},
          dtype=config.parameter_dtype,
      ),
      pz.nn.Elementwise(jax.nn.gelu),
      pz.nn.Affine.from_config(
          name=f"{name}/out",
          init_base_rng=init_base_rng,
          input_axes={"neurons": config.intermediate_size},
          output_axes={"embedding": config.hidden_size},
          dtype=config.parameter_dtype,
      ),
  ])


def _build_block(
    name: str,
    init_base_rng: jax.Array | None,
    config: Gemma3VisionConfig,
) -> Gemma3VisionBlock:
  return Gemma3VisionBlock([
      pz.nn.Residual(
          pz.nn.Sequential([
              pz.nn.LayerNorm.from_config(
                  name=f"{name}/pre_attn_norm",
                  init_base_rng=init_base_rng,
                  across_axes={"embedding": config.hidden_size},
                  epsilon=config.layer_norm_eps,
                  dtype=config.parameter_dtype,
              ),
              _build_attention(f"{name}/attention", init_base_rng, config),
          ])
      ),
      pz.nn.Residual(
          pz.nn.Sequential([
              pz.nn.LayerNorm.from_config(
                  name=f"{name}/pre_mlp_norm",
                  init_base_rng=init_base_rng,
                  across_axes={"embedding": config.hidden_size},
                  epsilon=config.layer_norm_eps,
                  dtype=config.parameter_dtype,
              ),
              _build_mlp(f"{name}/mlp", init_base_rng, config),
          ])
      ),
  ])


@pz.pytree_dataclass
class Gemma3VisionTower(pz.nn.Layer):
  """Vision tower for Gemma-3 VLMs."""

  patch_embed: VisionPatchEmbedding
  position_embed: AddPositionEmbedding
  blocks: pz.nn.Sequential
  final_norm: pz.nn.Layer
  config: Gemma3VisionConfig = dataclasses.field(metadata={"pytree_node": False})

  def __call__(
      self, images: pz.nx.NamedArray, **side_inputs
  ) -> pz.nx.NamedArray:
    height = images.named_shape.get("height")
    width = images.named_shape.get("width")
    if height != self.config.image_size or width != self.config.image_size:
      raise ValueError(
          "Input image size does not match config image_size: "
          f"{height}x{width} vs {self.config.image_size}."
      )
    hidden = self.patch_embed(images)
    hidden = pz.nn.CastToDType(self.config.activation_dtype)(hidden)
    hidden = self.position_embed(hidden)
    hidden = self.blocks(hidden, **side_inputs)
    return self.final_norm(hidden)

  @classmethod
  def from_config(
      cls,
      config: Gemma3VisionConfig,
      init_base_rng: jax.Array | None = None,
      name: str = "vision_tower",
  ) -> Gemma3VisionTower:
    patch_dim = config.patch_size * config.patch_size * config.num_channels
    patch_embed = VisionPatchEmbedding(
        patch_size=config.patch_size,
        num_channels=config.num_channels,
        projection=pz.nn.Affine.from_config(
            name=f"{name}/patch_embedding",
            init_base_rng=init_base_rng,
            input_axes={"patch": patch_dim},
            output_axes={"embedding": config.hidden_size},
            dtype=config.parameter_dtype,
        ),
    )
    position_table = pz.nn.EmbeddingTable.from_config(
        name=f"{name}/position_embedding",
        init_base_rng=init_base_rng,
        vocab_size=config.num_patches,
        embedding_axes={"embedding": config.hidden_size},
        vocabulary_axis="position",
        dtype=config.parameter_dtype,
    )
    position_embed = AddPositionEmbedding(pz.nn.EmbeddingLookup(position_table))
    blocks = pz.nn.Sequential([
        _build_block(f"{name}/block_{i}", init_base_rng, config)
        for i in range(config.num_hidden_layers)
    ])
    final_norm = pz.nn.LayerNorm.from_config(
        name=f"{name}/final_norm",
        init_base_rng=init_base_rng,
        across_axes={"embedding": config.hidden_size},
        epsilon=config.layer_norm_eps,
        dtype=config.parameter_dtype,
    )
    return cls(
        patch_embed=patch_embed,
        position_embed=position_embed,
        blocks=blocks,
        final_norm=final_norm,
        config=config,
    )
