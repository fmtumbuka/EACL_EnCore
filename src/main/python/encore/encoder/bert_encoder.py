# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#        Copyright (c) -2023 - Mtumbuka F.                                                       #
#        All rights reserved.                                                                       #
#                                                                                                   #
#        Redistribution and use in source and binary forms, with or without modification, are       #
#        permitted provided that the following conditions are met:                                  #    
#        1. Redistributions of source code must retain the above copyright notice, this list of     #
#           conditions and the following disclaimer.                                                #
#        2. Redistributions in binary form must reproduce the above copyright notice, this list of  #
#           conditions and the following disclaimer in the documentation and/or other materials     #
#           provided with the distribution.                                                         #
#                                                                                                   #
#        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS \"AS IS\" AND ANY      #
#        EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF    #
#        MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE #
#        COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,   #
#        EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF         #
#        SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)     #
#        HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR   #
#        TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS         #
#        SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                               #
#                                                                                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


__license__ = "BSD-2-Clause"
__version__ = "2023.1"
__date__ = "08 Mar 2023"
__author__ = ""
__maintainer__ = ""
__email__ = ""
__status__ = "Development"

import encore.encoder.base_encoder as base_encoder
import insanity
import torch
import typing
from transformers import BertForMaskedLM


class BertEncoder(base_encoder.BaseEncoder):
    """This class implements a BERT-based encoder."""
    def __init__(self, bert_version: str, num_tokens: int = 0):
        """
        This creates a new instance of `BertEncoder`.
        Args:
            bert_version (str): The version of BERT as specified in the user configurations.
            num_tokens (int): The number of additional tokens being used. This is specified only when the user has
                added special tokens to the vocabulary.
        """

        # Add super class call
        super().__init__()

        # Sanitize args.
        insanity.sanitize_type("bert_version", bert_version, str)
        insanity.sanitize_type("num_tokens", num_tokens, int)
        insanity.sanitize_range("num_tokens", num_tokens, minimum=0)

        # Create encoder.
        self._encoder = BertForMaskedLM.from_pretrained(bert_version)

        # If additional special tokens are used, resize the embeddings.
        if num_tokens > 0:
            self._encoder.resize_token_embeddings(num_tokens)

    def encoder(self) -> BertForMaskedLM:
        return self._encoder

    def hidden_size(self) -> int:
        return self._encoder.config.to_dict()["hidden_size"]

    def encode(
            self,
            input_seq: torch.Tensor,
            labels: torch.Tensor = None
    ) -> typing.Tuple[torch.FloatTensor, typing.Any]:
        """See `base_encoder.BaseEncoder.encode`"""
        encoder_outputs = self._encoder(input_ids=input_seq, labels=labels, output_hidden_states=True)
        return encoder_outputs.hidden_states[12], encoder_outputs.loss

