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
__date__ = "20 Mar 2023"
__author__ = ""
__maintainer__ = ""
__email__ = ""
__status__ = "Development"

import datapipeline.base_data_loader as base_data_loader
import datapipeline.cluster as cluster
import datapipeline.story as story_c
import torch.utils.data as data
import typing
from transformers import AutoTokenizer


class GigaWordDataset(data.Dataset):

    def __init__(
            self,
            data_loader: base_data_loader.BaseDataLoader,
            tokenizer: AutoTokenizer,
            loaded_data: typing.List[story_c.Story] = None
    ):
        """

        Args:
            data_loader:
        """
        super().__init__()
        # Sanitize args.

        # Store args.
        self._tokenizer = tokenizer
        self._data = []

        # Load data
        data = None
        if loaded_data is None:
            data = data_loader.load()
        else:
            data = loaded_data
        print("Pre-processing data...")
        # loop through clusters
        for story in data:
            for current_cluster in story.clusters:
                new_sentences = []
                for sentence in current_cluster.sentences:
                    new_entity_mask = []
                    new_tokens = []
                    new_token_idx = []
                    for token, mask in zip(sentence.text_arr, sentence.entity_mask):
                        tokenized_token = self._tokenizer.tokenize(token)
                        tokenized_token_idx = self._tokenizer.convert_tokens_to_ids(tokenized_token)
                        num_pieces = len(tokenized_token_idx)

                        if num_pieces > 1:
                            new_entity_mask.extend([mask] * num_pieces)
                            new_tokens.extend(tokenized_token)
                            new_token_idx.extend(tokenized_token_idx)
                        else:
                            new_entity_mask.append(mask)
                            new_tokens.append(token)
                            new_token_idx.extend(tokenized_token_idx)

                    # Add special tokens
                    new_entity_mask = [0] + new_entity_mask + [0]
                    new_tokens = [self._tokenizer.cls_token] + new_tokens + [self._tokenizer.sep_token]
                    new_token_idx = [self._tokenizer.cls_token_id] + new_token_idx + [self._tokenizer.sep_token_id]

                    # The number of ones in an entity represents the vectors that must be close during contrastive loss
                    # ....
                    label_placeholder = [0] * sum(new_entity_mask)

                    # Update sentence.
                    sentence.entity_mask = new_entity_mask
                    sentence.text_arr = new_tokens
                    sentence.token_ids = new_token_idx
                    sentence.label_placeholders = label_placeholder

                    new_sentences.append(sentence)
                    break
                new_cluster = cluster.Cluster(current_cluster.cluster_id)
                for sentence in new_sentences:
                    new_cluster.add_sentence(sentence)
                self._data.append(new_cluster)
        print("OK")
        print()

    def __getitem__(self, index) -> typing.List[cluster.Cluster]:
        """
        This return the input sequence with a corresponding entity mask.
        Args:
            index (int): The index for the sequence and entity mask to be retrieved.

        Returns:
            input_seq (list): A list of token ids for the tokens in the input sentence.
            input_mask (list): An entity mask with `1` indicating the position of an entity, and `0` for non-entities.
        """
        return [self._data[index]]

    def __len__(self) -> int:
        """int: Specifies the size of the dataset."""
        return len(self._data)

    @property
    def tokenizer(self) -> AutoTokenizer:
        """Specifies the tokenizer used."""
        return self._tokenizer
