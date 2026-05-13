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

"""Helpers for selecting stable transformer intervention sites.

These helpers operate on the model tree itself. For `LayerStack` models, block
local selections refer to the prototype stacked block in the tree, not one
independently selectable node per logical decoder layer.
"""

from __future__ import annotations

from typing import Any

from penzai import pz
from penzai.core import selectors
from . import model_parts

COMMON_DECODER_SITE_NAMES = (
    "embedder",
    "pre_attention_norm",
    "attention",
    "post_attention_norm",
    "pre_ffw_norm",
    "mlp",
    "post_ffw_norm",
    "final_norm",
    "lm_head",
    "final_logits",
)

COMMON_DECODER_BLOCK_SITE_NAMES = (
    "pre_attention_norm",
    "attention",
    "post_attention_norm",
    "pre_ffw_norm",
    "mlp",
    "post_ffw_norm",
)

COMMON_MULTIMODAL_SITE_NAMES = (
    "vision_encoder",
    "vision_projection",
    "stitch_embeddings",
    "truncate_vision_logits",
)

ALL_SITE_NAMES = COMMON_DECODER_SITE_NAMES + COMMON_MULTIMODAL_SITE_NAMES


def _validate_site_name(
    site_name: str, *, allowed_site_names: tuple[str, ...] = ALL_SITE_NAMES
) -> None:
  if site_name not in allowed_site_names:
    raise ValueError(
        f"Unknown transformer site name {site_name!r}. Known site names are:"
        f" {allowed_site_names!r}"
    )


def available_intervention_site_names(tree: Any) -> tuple[str, ...]:
  """Returns the known intervention-site names present in a model tree."""
  return tuple(
      site_name
      for site_name in ALL_SITE_NAMES
      if not select_intervention_site(tree, site_name).is_empty()
  )


def select_intervention_site(
    tree: Any, site_name: str
) -> selectors.Selection[pz.nn.NamedGroup]:
  """Selects every named intervention site with the requested name."""
  _validate_site_name(site_name)
  return (
      pz.select(tree)
      .at_instances_of(pz.nn.NamedGroup)
      .where(lambda group: group.name == site_name)
  )


def select_decoder_blocks(
    tree: Any,
) -> selectors.Selection[model_parts.TransformerBlock]:
  """Selects decoder blocks from a model tree."""
  return pz.select(tree).at_instances_of(model_parts.TransformerBlock)


def select_block_intervention_sites(
    tree: Any, site_name: str
) -> selectors.Selection[pz.nn.NamedGroup]:
  """Selects block-local intervention sites inside decoder blocks."""
  _validate_site_name(
      site_name, allowed_site_names=COMMON_DECODER_BLOCK_SITE_NAMES
  )
  return select_decoder_blocks(tree).refine(
      lambda block: select_intervention_site(block, site_name)
  )
