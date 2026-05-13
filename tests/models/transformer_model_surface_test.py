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

"""Tests for the shared transformer intervention-site surface."""

from absl.testing import absltest
import jax.numpy as jnp
from penzai.models.transformer import model_surface
from penzai.models.transformer.variants import llamalike_common


def _build_model(
    *,
    num_decoder_blocks=2,
    tie_embedder_and_logits=True,
    use_post_attn_norm=False,
    use_post_ffw_norm=False,
    final_logit_softcap=None,
    use_layer_stack=False,
):
  return llamalike_common.build_llamalike_transformer(
      llamalike_common.LlamalikeTransformerConfig(
          num_kv_heads=2,
          query_head_multiplier=1,
          embedding_dim=16,
          projection_dim=4,
          mlp_hidden_dim=32,
          num_decoder_blocks=num_decoder_blocks,
          vocab_size=11,
          mlp_variant="geglu_approx",
          tie_embedder_and_logits=tie_embedder_and_logits,
          use_post_attn_norm=use_post_attn_norm,
          use_post_ffw_norm=use_post_ffw_norm,
          final_logit_softcap=final_logit_softcap,
          use_layer_stack=use_layer_stack,
          parameter_dtype=jnp.float32,
          activation_dtype=jnp.float32,
      ),
      init_base_rng=None,
  )


class TransformerModelSurfaceTest(absltest.TestCase):

  def test_unstacked_surface_exposes_required_sites(self):
    model = _build_model()

    self.assertEqual(
        model_surface.available_intervention_site_names(model),
        tuple(
            site_name
            for site_name in model_surface.COMMON_DECODER_SITE_NAMES
            if site_name not in ("post_attention_norm", "post_ffw_norm")
        ),
    )
    self.assertEqual(
        model_surface.select_intervention_site(model, "embedder").count(), 1
    )
    self.assertEqual(
        model_surface.select_intervention_site(model, "final_norm").count(), 1
    )
    self.assertEqual(
        model_surface.select_intervention_site(model, "lm_head").count(), 1
    )
    self.assertEqual(
        model_surface.select_intervention_site(
            model, "final_logits"
        ).count(),
        1,
    )
    self.assertEqual(model_surface.select_decoder_blocks(model).count(), 2)
    self.assertEqual(
        model_surface.select_block_intervention_sites(
            model, "pre_attention_norm"
        ).count(),
        2,
    )
    self.assertEqual(
        model_surface.select_block_intervention_sites(
            model, "attention"
        ).count(),
        2,
    )
    self.assertEqual(
        model_surface.select_block_intervention_sites(model, "mlp").count(),
        2,
    )
    self.assertEqual(
        model_surface.select_intervention_site(
            model, "post_attention_norm"
        ).count(),
        0,
    )
    self.assertEqual(
        model_surface.select_intervention_site(model, "post_ffw_norm").count(),
        0,
    )

  def test_optional_sites_and_final_logits_with_linear_head(self):
    model = _build_model(
        tie_embedder_and_logits=False,
        use_post_attn_norm=True,
        use_post_ffw_norm=True,
        final_logit_softcap=30.0,
    )

    self.assertEqual(
        model_surface.available_intervention_site_names(model),
        model_surface.COMMON_DECODER_SITE_NAMES,
    )
    self.assertEqual(
        model_surface.select_intervention_site(
            model, "post_attention_norm"
        ).count(),
        2,
    )
    self.assertEqual(
        model_surface.select_intervention_site(model, "post_ffw_norm").count(),
        2,
    )
    self.assertEqual(
        model_surface.select_intervention_site(model, "lm_head").count(), 1
    )
    self.assertEqual(
        model_surface.select_intervention_site(
            model, "final_logits"
        ).count(),
        1,
    )

  def test_layer_stack_selects_prototype_block_sites(self):
    model = _build_model(
        use_post_attn_norm=True,
        use_post_ffw_norm=True,
        use_layer_stack=True,
    )

    self.assertEqual(
        model_surface.available_intervention_site_names(model),
        model_surface.COMMON_DECODER_SITE_NAMES,
    )
    self.assertEqual(model_surface.select_decoder_blocks(model).count(), 1)
    self.assertEqual(
        model_surface.select_block_intervention_sites(
            model, "attention"
        ).count(),
        1,
    )
    self.assertEqual(
        model_surface.select_block_intervention_sites(model, "mlp").count(),
        1,
    )
    self.assertEqual(
        model_surface.select_intervention_site(
            model, "post_attention_norm"
        ).count(),
        1,
    )
    self.assertEqual(
        model_surface.select_intervention_site(model, "post_ffw_norm").count(),
        1,
    )

  def test_unknown_site_name_raises(self):
    model = _build_model()

    with self.assertRaisesRegex(ValueError, "Unknown transformer site name"):
      model_surface.select_intervention_site(model, "not_a_site")

    with self.assertRaisesRegex(ValueError, "Unknown transformer site name"):
      model_surface.select_block_intervention_sites(model, "embedder")


if __name__ == "__main__":
  absltest.main()
