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

"""Selection helpers for common VLM components.

These helpers assume that VLMs wrap major components in ``pz.nn.NamedGroup``
with common names. If your model uses different names, pass ``names=...`` with
custom entries or wrap the relevant blocks with ``NamedGroup`` to make them
selectable.

Examples:
  >>> from penzai import pz
  >>> from penzai.models.vlm import selectors as vlm_selectors

  # Zero out visual tokens at the fusion boundary.
  >>> visual_slice, text_slice = vlm_selectors.fusion_token_slices(
  ...     num_visual_tokens=256
  ... )
  >>> activations = model(inputs)  # NamedArray with a "seq" axis.
  >>> ablated = (
  ...     pz.select(activations)
  ...     .at_instances_of(pz.nx.NamedArrayBase)
  ...     .apply(lambda x: x.at[visual_slice].set(0))
  ... )

  # Patch visual tokens with a fixed template.
  >>> template = pz.nx.zeros({"seq": 256, "embedding": 4096})
  >>> patched = (
  ...     pz.select(activations)
  ...     .at_instances_of(pz.nx.NamedArrayBase)
  ...     .apply(lambda x: x.at[visual_slice].set(template))
  ... )

  # Inspect cross-modal influence from text queries to visual keys.
  >>> attn_weights = captured_attn  # NamedArray with "seq" and "kv_seq".
  >>> cross_modal = attn_weights[
  ...     {
  ...         "seq": text_slice["seq"],
  ...         "kv_seq": visual_slice["seq"],
  ...     }
  ... ].mean()
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Callable

from penzai import pz
from penzai.core import named_axes
from penzai.core import selectors
from penzai.models.transformer import model_parts
from penzai.nn import attention as attention_layers
from penzai.nn import grouping

DEFAULT_VISION_TOWER_NAMES = (
    "vision_tower",
    "vision_encoder",
    "image_encoder",
)
DEFAULT_VISION_BLOCK_PREFIXES = (
    "block_",
    "layer_",
    "encoder_block_",
    "encoder_layer_",
)
DEFAULT_PROJECTOR_NAMES = (
    "multimodal_projector",
    "mm_projector",
    "projector",
)
DEFAULT_PROJECTOR_OUTPUT_NAMES = (
    "multimodal_projector_output",
    "mm_projector_output",
    "projector_output",
)
DEFAULT_LM_NAMES = (
    "language_model",
    "lm",
    "decoder",
    "text_model",
)
DEFAULT_LM_BLOCK_PREFIXES = (
    "block_",
    "layer_",
    "decoder_block_",
    "decoder_layer_",
)
DEFAULT_HEAD_AXIS_NAMES = (
    "heads",
    "head_groups",
    "query_heads",
)


def _as_selection(
    tree_or_selection: selectors.Selection | Any,
) -> selectors.Selection:
  if isinstance(tree_or_selection, selectors.Selection):
    return tree_or_selection
  return pz.select(tree_or_selection)


def _normalize_names(names: Iterable[str]) -> set[str]:
  return {name.lower() for name in names}


def _name_matches(candidate: str, names: set[str]) -> bool:
  return candidate.lower() in names


def _has_prefix(candidate: str, prefixes: Iterable[str]) -> bool:
  return any(candidate.startswith(prefix) for prefix in prefixes)


def _block_predicate(
    block_types: tuple[type[Any], ...] | None,
    block_name_prefixes: Iterable[str],
) -> Callable[[Any], bool]:
  def _predicate(subtree: Any) -> bool:
    if block_types and isinstance(subtree, block_types):
      return True
    if isinstance(subtree, grouping.NamedGroup):
      return _has_prefix(subtree.name, block_name_prefixes)
    return False

  return _predicate


def _head_axis_predicate(
    head_axis_names: Iterable[str],
) -> Callable[[Any], bool]:
  head_axis_set = set(head_axis_names)

  def _predicate(subtree: Any) -> bool:
    if not isinstance(subtree, named_axes.NamedArrayBase):
      return False
    return any(name in subtree.named_shape for name in head_axis_set)

  return _predicate


def vision_tower(
    tree_or_selection: selectors.Selection | Any,
    *,
    names: Iterable[str] = DEFAULT_VISION_TOWER_NAMES,
) -> selectors.Selection[grouping.NamedGroup]:
  """Selects the vision tower container(s) in a VLM."""
  selection = _as_selection(tree_or_selection)
  name_set = _normalize_names(names)
  return selection.at_instances_of(grouping.NamedGroup).where(
      lambda group: _name_matches(group.name, name_set)
  )


def vision_tower_layers(
    tree_or_selection: selectors.Selection | Any,
    *,
    names: Iterable[str] = DEFAULT_VISION_TOWER_NAMES,
    block_types: tuple[type[Any], ...] | None = None,
    block_name_prefixes: Iterable[str] = DEFAULT_VISION_BLOCK_PREFIXES,
) -> selectors.Selection:
  """Selects vision tower blocks by type and/or name prefix."""
  return vision_tower(tree_or_selection, names=names).at_subtrees_where(
      _block_predicate(block_types, block_name_prefixes)
  )


def vision_tower_attention_layers(
    tree_or_selection: selectors.Selection | Any,
    *,
    names: Iterable[str] = DEFAULT_VISION_TOWER_NAMES,
) -> selectors.Selection[attention_layers.Attention]:
  """Selects attention layers within the vision tower."""
  return vision_tower(tree_or_selection, names=names).at_instances_of(
      attention_layers.Attention
  )


def vision_tower_attention_heads(
    tree_or_selection: selectors.Selection | Any,
    *,
    names: Iterable[str] = DEFAULT_VISION_TOWER_NAMES,
    head_axis_names: Iterable[str] = DEFAULT_HEAD_AXIS_NAMES,
) -> selectors.Selection[named_axes.NamedArrayBase]:
  """Selects named arrays with attention-head axes inside the vision tower."""
  return vision_tower(tree_or_selection, names=names).at_subtrees_where(
      _head_axis_predicate(head_axis_names)
  )


def multimodal_projector(
    tree_or_selection: selectors.Selection | Any,
    *,
    names: Iterable[str] = DEFAULT_PROJECTOR_NAMES,
) -> selectors.Selection[grouping.NamedGroup]:
  """Selects the multimodal projector module(s)."""
  selection = _as_selection(tree_or_selection)
  name_set = _normalize_names(names)
  return selection.at_instances_of(grouping.NamedGroup).where(
      lambda group: _name_matches(group.name, name_set)
  )


def multimodal_projector_output(
    tree_or_selection: selectors.Selection | Any,
    *,
    names: Iterable[str] = DEFAULT_PROJECTOR_OUTPUT_NAMES,
) -> selectors.Selection[grouping.NamedGroup]:
  """Selects the tagged output of the multimodal projector.

  This helper expects the output activation to be wrapped in a ``NamedGroup``
  (e.g. ``NamedGroup(name="multimodal_projector_output", sublayers=[])``).
  """
  selection = _as_selection(tree_or_selection)
  name_set = _normalize_names(names)
  return selection.at_instances_of(grouping.NamedGroup).where(
      lambda group: _name_matches(group.name, name_set)
  )


def fusion_token_slices(
    *,
    num_visual_tokens: int,
    seq_axis: str = "seq",
) -> tuple[dict[str, slice], dict[str, slice]]:
  """Returns indexers for visual and text tokens split by a fusion boundary."""
  if num_visual_tokens < 0:
    raise ValueError("num_visual_tokens must be non-negative.")
  visual_slice = {seq_axis: pz.slice[:num_visual_tokens]}
  text_slice = {seq_axis: pz.slice[num_visual_tokens:]}
  return visual_slice, text_slice


def language_model(
    tree_or_selection: selectors.Selection | Any,
    *,
    names: Iterable[str] = DEFAULT_LM_NAMES,
) -> selectors.Selection[grouping.NamedGroup]:
  """Selects the language model container(s)."""
  selection = _as_selection(tree_or_selection)
  name_set = _normalize_names(names)
  return selection.at_instances_of(grouping.NamedGroup).where(
      lambda group: _name_matches(group.name, name_set)
  )


def lm_blocks(
    tree_or_selection: selectors.Selection | Any,
    *,
    names: Iterable[str] = DEFAULT_LM_NAMES,
    block_types: tuple[type[Any], ...] | None = (model_parts.TransformerBlock,),
    block_name_prefixes: Iterable[str] = DEFAULT_LM_BLOCK_PREFIXES,
) -> selectors.Selection:
  """Selects LM blocks by type and/or name prefix."""
  return language_model(tree_or_selection, names=names).at_subtrees_where(
      _block_predicate(block_types, block_name_prefixes)
  )


def lm_attention_layers(
    tree_or_selection: selectors.Selection | Any,
    *,
    names: Iterable[str] = DEFAULT_LM_NAMES,
) -> selectors.Selection[attention_layers.Attention]:
  """Selects attention layers inside the language model."""
  return language_model(tree_or_selection, names=names).at_instances_of(
      attention_layers.Attention
  )


def lm_attention_heads(
    tree_or_selection: selectors.Selection | Any,
    *,
    names: Iterable[str] = DEFAULT_LM_NAMES,
    head_axis_names: Iterable[str] = DEFAULT_HEAD_AXIS_NAMES,
) -> selectors.Selection[named_axes.NamedArrayBase]:
  """Selects named arrays with attention-head axes inside the language model."""
  return language_model(tree_or_selection, names=names).at_subtrees_where(
      _head_axis_predicate(head_axis_names)
  )
