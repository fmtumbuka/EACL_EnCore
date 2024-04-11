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

import datapipeline.base_data_loader as base_data_loader
import random
import torch.utils.data as data
import typing
import utilpackage.index_map as index_map
from transformers import AutoTokenizer


class ACEDataset(data.Dataset):

    def __init__(
            self,
            data_loader: base_data_loader.BaseDataLoader,
            tokenizer: AutoTokenizer,
            rel_types: index_map.IndexMap,
            rel_mask: bool = False
    ):
        """

        Args:
            data_loader:
            tokenizer:
            rel_types:
            rel_mask:
        """
        super().__init__()
        # Sanitize args.

        # Store args.
        self._tokenizer = tokenizer
        self._rel_types = rel_types
        self._rel_mask = rel_mask
        self._data = []
        loaded_data = data_loader.load()
        print("Pre-processing data...")
        for sample_idx, item in enumerate(loaded_data):
            input_tokens = item.input_text_tokens
            arg_1 = item.relation_arg_1
            arg_2 = item.relation_arg_2
            # check if length of args
            arg_1_len = sum(1 for i in arg_1 if i != "O")
            arg_2_len = sum(1 for i in arg_2 if i != "O")

            if arg_1_len == 0 or arg_2_len == 0:
                continue
            else:
                input_head, head_mask = self.generate_mask(input_tokens, arg_1)
                _, tail_mask = self.generate_mask(input_tokens, arg_2)

                loaded_data[sample_idx].input_token_idx = input_head
                loaded_data[sample_idx].relation_arg_1_mask = head_mask
                loaded_data[sample_idx].relation_arg_2_mask = tail_mask
                loaded_data[sample_idx].relation_type_id = self._rel_types.index(item.relation_type)

                if self._rel_mask:
                    arg_1_index = [
                        idx for idx, label in enumerate(loaded_data[sample_idx].relation_arg_1) if label != "O"
                    ]
                    arg_2_index = [
                        idx for idx, label in enumerate(loaded_data[sample_idx].relation_arg_2) if label != "O"
                    ]
                    arg_1 = [loaded_data[sample_idx].input_text_tokens[i] for i in arg_1_index]
                    arg_2 = [loaded_data[sample_idx].input_text_tokens[i] for i in arg_2_index]

                    rel_prompt = loaded_data[sample_idx].input_text_tokens + \
                             [self._tokenizer.sep_token] + \
                             ["The", "relation", "between"] + \
                             arg_1 + \
                             ["and"] + \
                             arg_2 + \
                             ["is", self._tokenizer.mask_token, "?"]

                    rel_prompt_idx, rel_prompt_mask = self.generate_mask(rel_prompt)
                    loaded_data[sample_idx].rel_prompt = rel_prompt
                    loaded_data[sample_idx].rel_prompt_idx = rel_prompt_idx
                    loaded_data[sample_idx].rel_prompt_mask = rel_prompt_mask
                self._data.append(loaded_data[sample_idx])

        print("OK")

    @property
    def rel_types(self) -> index_map.IndexMap:
        return self._rel_types

    def __getitem__(self, index) -> typing.Tuple[
        typing.List, typing.List, typing.List, typing.List, typing.List, typing.List
    ]:
        """
        This return the input sequence with a corresponding entity mask.
        Args:
            index (int): The index for the sequence and entity mask to be retrieved.

        Returns:
            input_seq (list): A list of token ids for the tokens in the input sentence.
            input_mask (list): An entity mask with `1` indicating the position of an entity, and `0` for non-entities.
        """
        input_idx = self._data[index].input_token_idx
        head_entity_mask = self._data[index].relation_arg_1_mask
        tail_entity_mask = self._data[index].relation_arg_2_mask
        relation_type_id = [self._data[index].relation_type_id]
        rel_prompt_idx = self._data[index].rel_prompt_idx if self._data[index].rel_prompt_idx is not None else [self._data[index].rel_prompt_idx]
        rel_prompt_mask = self._data[index].rel_prompt_mask if self._data[index].rel_prompt_mask is not None else [self._data[index].rel_prompt_mask]
        if self._data[index].rel_prompt_idx is not None:
            head_entity_mask += [0] * (len(rel_prompt_idx) - len(head_entity_mask))
            tail_entity_mask += [0] * (len(rel_prompt_idx) - len(tail_entity_mask))

        return input_idx, head_entity_mask, tail_entity_mask, relation_type_id, rel_prompt_idx, rel_prompt_mask

    def __len__(self) -> int:
        """int: Specifies the size of the dataset."""
        return len(self._data)

    def generate_mask(
            self,
            input_tokens,
            input_mask=None
    ) -> typing.Tuple[typing.List, typing.List]:
        """

        Args:
            input_mask
            input_tokens:

        Returns:

        """
        new_tokens = []
        mask_token = self._tokenizer.mask_token
        mask_token_id = self._tokenizer.mask_token_id
        pad_token = self._tokenizer.pad_token
        pad_token_id = self._tokenizer.pad_token_id

        processed_input_mask = []
        processed_input_tokens = []

        if input_mask is None:
            processed_input_tokens = self._tokenizer(" ".join(input_tokens))["input_ids"]

            # Find positions of entity mentions
            rel_mask_position = processed_input_tokens.index(mask_token_id)
            processed_input_mask = [0] * len(processed_input_tokens)
            processed_input_mask[rel_mask_position] = 1
        else:
            # Find the length of the argument in terms of the number of tokens.
            length_arg_span = sum(1 for i in input_mask if i != 'O')
            span_count = 0
            for token, mask in zip(input_tokens, input_mask):
                if mask != "O":
                    if span_count == 0:
                        new_tokens.append(mask_token)
                        new_tokens.append(token)
                    elif 0 < span_count < length_arg_span:
                        new_tokens.append(token)

                    if length_arg_span == 1 or span_count == (length_arg_span - 1):
                        new_tokens.append(pad_token)
                    span_count += 1
                else:
                    new_tokens.append(token)

            # Tokenize tokens
            new_text_tokens = " ".join(new_tokens)
            tokenized_text = self._tokenizer(new_text_tokens)["input_ids"]

            # Find positions of entity mentions
            entity_start = tokenized_text.index(mask_token_id)
            entity_end = tokenized_text.index(pad_token_id)
            entity_mask = [0] * len(tokenized_text)
            if length_arg_span > 1:
                random_index = random.randint((entity_start + 1), (entity_end - 1))
                entity_mask[random_index] = 1
            else:
                for i in range(entity_start, entity_end):
                    entity_mask[i] = 1

            for token, mask in zip(tokenized_text, entity_mask):
                if token == mask_token_id or token == pad_token_id:
                    pass
                else:
                    processed_input_mask.append(mask)
                    processed_input_tokens.append(token)

            # Check the number of ones
            num_ones = sum(1 for i in processed_input_mask if i == 1)
            counter = 0
            if num_ones > 0:
                for idx, mask in enumerate(processed_input_mask):
                    if mask == 1 and counter == 0:
                        counter += 1
                    elif mask == 1 and counter > 0:
                        processed_input_mask[idx] = 0
                        counter += 1
                    else:
                        pass

        return processed_input_tokens, processed_input_mask

    @property
    def tokenizer(self) -> AutoTokenizer:
        """Specifies the tokenizer used."""
        return self._tokenizer
