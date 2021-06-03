# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Dict

from fairseq import checkpoint_utils
from fairseq.data.legacy.masked_lm_dictionary import MaskedLMDictionary
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import (
    TransformerDecoder,
    TransformerEncoder,
    TransformerModel,
    base_architecture as transformer_base_architecture,
)


@register_model("transformer_xlm_iwslt_decoder")
class TransformerFromPretrainedXLMModel(TransformerModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        TransformerModel.add_args(parser)
        parser.add_argument(
            "--pretrained-xlm-checkpoint",
            type=str,
            metavar="STR",
            help="XLM model to use for initializing transformer encoder "
                 "and/or decoder",
        )
        parser.add_argument(
            "--init-encoder-only",
            action="store_true",
            help="if set, don't load the XLM weights and embeddings into "
                 "decoder",
        )
        parser.add_argument(
            "--init-decoder-only",
            action="store_true",
            help="if set, don't load the XLM weights and embeddings into "
                 "encoder",
        )

    @classmethod
    def build_model(self, args, task, cls_dictionary=MaskedLMDictionary):
        assert hasattr(args, "pretrained_xlm_checkpoint"), (
            "You must specify a path for --pretrained-xlm-checkpoint to use "
            "--arch transformer_from_pretrained_xlm"
        )
        assert isinstance(task.source_dictionary,
                          cls_dictionary) and isinstance(
            task.target_dictionary, cls_dictionary
        ), (
            "You should use a MaskedLMDictionary when using --arch "
            "transformer_from_pretrained_xlm because the pretrained XLM model "
            "was trained using data binarized with MaskedLMDictionary. "
            "For translation, you may want to use --task "
            "translation_from_pretrained_xlm"
        )
        assert not (
                getattr(args, "init_encoder_only", False)
                and getattr(args, "init_decoder_only", False)
        ), "Only one of --init-encoder-only and --init-decoder-only can be set."
        return super().build_model(args, task)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerEncoderFromPretrainedXLM(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )


def upgrade_state_dict_with_xlm_weights(
        state_dict: Dict[str, Any], pretrained_xlm_checkpoint: str
) -> Dict[str, Any]:
    """
    Load XLM weights into a Transformer encoder or decoder model.

    Args:
        state_dict: state dict for either TransformerEncoder or
            TransformerDecoder
        pretrained_xlm_checkpoint: checkpoint to load XLM weights from

    Raises:
        AssertionError: If architecture (num layers, attention heads, etc.)
            does not match between the current Transformer encoder or
            decoder and the pretrained_xlm_checkpoint
    """
    if not os.path.exists(pretrained_xlm_checkpoint):
        raise IOError(
            "Model file not found: {}".format(pretrained_xlm_checkpoint))

    state = checkpoint_utils.load_checkpoint_to_cpu(pretrained_xlm_checkpoint)
    xlm_state_dict = state["model"]
    for key in xlm_state_dict.keys():

        for search_key in ["embed_tokens", "embed_positions", "layers"]:
            if search_key in key:
                subkey = key[key.find(search_key):]
                if "in_proj_weight" in subkey or \
                        "in_proj_bias" in subkey:
                    continue
                else:
                    assert subkey in state_dict, (
                        "{} \nTransformer encoder / decoder "
                        "state_dict does not contain {}. \nCannot "
                        "load {} from pretrained XLM checkpoint "
                        "{} into Transformer.".format(
                            str(state_dict.keys()), subkey, key,
                            pretrained_xlm_checkpoint
                        )
                    )

                state_dict[subkey] = xlm_state_dict[key]
    return state_dict


class TransformerEncoderFromPretrainedXLM(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        if getattr(args, "init_decoder_only", False):
            # Don't load XLM weights for encoder if --init-decoder-only
            return

        assert hasattr(args, "pretrained_xlm_checkpoint"), (
            "--pretrained-xlm-checkpoint must be specified to load Transformer "
            "encoder from pretrained XLM"
        )
        xlm_loaded_state_dict = upgrade_state_dict_with_xlm_weights(
            state_dict=self.state_dict(),
            pretrained_xlm_checkpoint=args.pretrained_xlm_checkpoint,
        )
        self.load_state_dict(xlm_loaded_state_dict, strict=True)


# class TransformerDecoderFromPretrainedXLM(TransformerDecoder):
#     def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
#         super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
#         if getattr(args, "init_encoder_only", False):
#             # Don't load XLM weights for decoder if --init-encoder-only
#             return
#         assert hasattr(args, "pretrained_xlm_checkpoint"), (
#             "--pretrained-xlm-checkpoint must be specified to load Transformer "
#             "decoder from pretrained XLM"
#         )
#
#         xlm_loaded_state_dict = upgrade_state_dict_with_xlm_weights(
#             state_dict=self.state_dict(),
#             pretrained_xlm_checkpoint=args.pretrained_xlm_checkpoint,
#         )
#         self.load_state_dict(xlm_loaded_state_dict, strict=True)


@register_model_architecture(
    "transformer_xlm_iwslt_decoder", "transformer_xlm_iwslt_decoder")
def transformer_xlm_iwslt_decoder(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 3072)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    transformer_base_architecture(args)
