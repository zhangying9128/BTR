# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
from typing import Dict, List, Optional

import torch
from fairseq.models import (
    FairseqDecoder,
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)


logger = logging.getLogger(__name__)


@register_model("hf_T5")
class HuggingFaceT5SEQ2SEQModel(FairseqLanguageModel):
    def __init__(self, decoder):
        super().__init__(decoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(HuggingFaceT5SEQ2SEQ(args, task))


class HuggingFaceT5SEQ2SEQ(FairseqDecoder):
    def __init__(self, args, task):
        try:
            from transformers import T5Config, T5ForConditionalGeneration
        except ImportError:
            raise ImportError(
                "\n\nPlease install huggingface/transformers with:"
                "\n\n  pip install transformers"
            )
        super().__init__(task.target_dictionary)
        config = T5Config(
            vocab_size=32128, 
            d_ff=2048,
            d_kv=64,
            d_model=768,
            num_heads=12,
            num_layers=12, 
            num_decoder_layers=12,
            feed_forward_proj="gated-gelu",
            tie_word_embeddings=False,
            decoder_start_token_id=0,
        )
        config.bidirectional_decoder = args.bidirectional_decoder
        self.model = T5ForConditionalGeneration(config)
        # set zero embedding for padding symbol
        self.pad_idx = task.target_dictionary.pad()


    def forward(
        self, input_ids, decoder_input_ids, decoder_attention_mask, **kwargs
    ):
        attention_mask = input_ids.ne(self.pad_idx).int()
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
        )

        return outputs.logits

    def max_positions(self):
        return (512, 512)



@register_model_architecture("hf_T5", "hf_T5")
def default_architecture(args):
    pass