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
__date__ = "09 Apr 2023"
__author__ = ""
__maintainer__ = ""
__email__ = ""
__status__ = "Development"

import insanity
import datapipeline.ace_entity_data_loader
import datapipeline.base_data_loader as base_data_loader
import datapipeline.input_example as input_example
import experiment
import torch.utils.data as data
import typing
import utilpackage.index_map as index_map
from transformers import AutoTokenizer


class ACEEntityDataset(data.Dataset):
    """This creates the ACE05 entity classification dataset."""
    def __init__(
            self,
            data_loader: base_data_loader.BaseDataLoader,
            entity_types: index_map.IndexMap,
            tokenizer: AutoTokenizer
    ):
        """
        This creates an instance of the `ACEEntityDataset`.
        Args:
            data_loader (:class::`base_data_loader.BaseDataLoader`): The specified data loader.
            entity_types (:class::`index_map.IndexMap`): An abstraction that maps entity types to indices.
            tokenizer (:class::`AutoTokenizer`): An instance of AutoTokenizer from the Transformers library that
                depends on the encoder specified in the experiment configuration.
        """
        super().__init__()

        # Sanitize args.
        insanity.sanitize_type("data_loader", data_loader, base_data_loader.BaseDataLoader)
        insanity.sanitize_type("entity_types", entity_types, index_map.IndexMap)

        # Store args.
        self._entity_types = entity_types
        self._tokenizer = tokenizer
        self._data = data_loader.load()
        print("Pre-processing data...")
        for sample_idx, item in enumerate(self._data):
            # Make sure that only labels that are in the specified entity types are retained.
            labels = [label if label in self._entity_types.all_values() else "O" for label in item.ent_labels]

            # In case of the tokenization splits words into pieces.
            new_labels = []
            new_tokens = []
            new_token_ids = []
            new_entity_mask = []
            for token, mask, label in zip(item.input_tokens, item.ent_mask, labels):
                # compute the number of pieces that the word was split into
                word_pieces = self._tokenizer.tokenize(token)
                token_idx = self._tokenizer.convert_tokens_to_ids(word_pieces)
                # check if the word was split into pieces
                if len(word_pieces) > 1:  # -> the word was split
                    new_entity_mask.extend([mask] * len(word_pieces))
                    new_labels.extend([label] * len(word_pieces))
                    new_tokens.extend(word_pieces)
                    new_token_ids.extend(token_idx)
                else:
                    new_entity_mask.append(mask)
                    new_labels.append(label)
                    new_tokens.extend(word_pieces)
                    new_token_ids.extend(token_idx)

            # Pre- and post-append tokens corresponding to the [CLS] and [SEP] tokens.
            new_entity_mask = [0] + new_entity_mask + [0]
            new_labels = ["O"] + new_labels + ["O"]
            new_tokens = [self._tokenizer.cls_token] + new_tokens + [self._tokenizer.sep_token]
            new_token_ids = [self._tokenizer.cls_token_id] + new_token_ids + [self._tokenizer.sep_token_id]
            new_label_idx = [self._entity_types.index(label) for label in new_labels]

            # Update current sample
            new_sample = input_example.InputExample(new_entity_mask, new_labels, new_tokens)
            new_sample.ent_labels_idx = new_label_idx
            new_sample.input_token_idx = new_token_ids
            self._data[sample_idx] = new_sample
        print("OK")

    def __getitem__(self, index) -> typing.Tuple[typing.List[int], typing.List[int], typing.List[int]]:
        """
        This returns the input example at a specific index.
        Args:
            index (int): The index of the example to be retrieved from the dataset.

        Returns:
            input_token_idx (list[int]): A sequence of indices of tokens in the tokenizer's vocabulary.
            ent_label_idx (list[int]): A sequence of indices of the entity labels in the entity label index map.
            ent_mask (list[int]): A mask with `1`s indicating the positions of entities and `0`s indicating non-entities
        """
        return self._data[index].input_token_idx, self._data[index].ent_labels_idx, self._data[index].ent_mask

    def __len__(self) -> int:
        """int: Specifies the size of the dataset."""
        return len(self._data)

    @property
    def entity_types(self) -> index_map.IndexMap:
        """Specifies the map of entity types to their indices."""
        return self._entity_types

    @property
    def tokenizer(self) -> AutoTokenizer:
        """Specifies the tokenizer used."""
        return self._tokenizer

