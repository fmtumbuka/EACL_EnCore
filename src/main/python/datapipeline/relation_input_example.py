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
__date__ = "02 Apr 2023"
__author__ = ""
__maintainer__ = ""
__email__ = ""
__status__ = "Development"

import insanity
import typing


class RelationInputExample(object):
    """This encapsulates a single input example for single sentence relation extraction."""

    def __init__(
            self,
            input_text_tokens: list,
            relation_arg_1: list,
            relation_arg_2: list,
            relation_type: str,
            input_token_idx: list = None,
            relation_arg_1_mask: list = None,
            relation_arg_2_mask: list = None,
            relation_type_id: int = None
    ):
        """
        This creates an instance on `RelationInputExample`.
        Args:
            input_text_tokens (list): The list of tokens in the input text.
            relation_arg_1 (list): The BIO tags indicating the positions of head entity for arg-1 in the
                input text.
            relation_arg_2 (list): The BIO tags indicating the positions of head entity for arg-2 in the
                input text.
            relation_type (str): The relation type being expressed in the input text.
            input_token_idx (list): The list of ids for tokens in the input text.
            relation_arg_1_mask (list): The mask representing the positions of arg-1 entities in question
                with `1` indicating the position of entities, and `0` for none entities. This is derived
                from the list of BIO-tags.
            relation_arg_2_mask (list): The mask representing the positions of arg-2 entities in question
                with `1` indicating the position of entities, and `0` for none entities. This is derived
                from the list of BIO-tags.
            relation_type_id (int): The id of the relation being expressed.

        """

        # Sanitize args.
        insanity.sanitize_type("input_text_tokens", input_text_tokens, list)
        insanity.sanitize_type("relation_arg_1", relation_arg_1, list)
        insanity.sanitize_type("relation_arg_2", relation_arg_2, list)
        insanity.sanitize_type("relation_type", relation_type, str)

        if input_token_idx is not None:
            insanity.sanitize_type("input_token_idx", input_token_idx, list)

        if relation_arg_1_mask is not None:
            insanity.sanitize_type("relation_arg_1_mask", relation_arg_1_mask, list)

        if relation_arg_2_mask is not None:
            insanity.sanitize_type("relation_arg_2_mask", relation_arg_2_mask, list)

        if relation_type_id is not None:
            insanity.sanitize_type("relation_type_id", relation_type_id, int)

        # Store args.
        self._input_text_tokens = input_text_tokens
        self._relation_arg_1 = relation_arg_1
        self._relation_arg_2 = relation_arg_2
        self._relation_type = relation_type
        self._input_token_idx = input_token_idx
        self._relation_arg_1_mask = relation_arg_1_mask
        self._relation_arg_2_mask = relation_arg_2_mask
        self._relation_type_id = relation_type_id
        self._rel_prompt = None
        self._rel_prompt_idx = None
        self._rel_prompt_mask = None

    @property
    def input_text_tokens(self) -> typing.List:
        """list: Specifies the list of tokens in the input text."""
        return self._input_text_tokens

    @input_text_tokens.setter
    def input_text_tokens(self, input_text_tokens: list) -> None:
        self._input_text_tokens = input_text_tokens

    @property
    def relation_arg_1(self) -> typing.List:
        """list: Specifies BIO tags indicating the positions of arg_1 entity mentions in the input text. """
        return self._relation_arg_1

    @relation_arg_1.setter
    def relation_arg_1(self, relation_arg_1: list) -> None:
        self._relation_arg_1 = relation_arg_1

    @property
    def relation_arg_2(self) -> typing.List:
        """list: Specifies BIO tags indicating the positions of arg_2 entity mentions in the input text. """
        return self._relation_arg_2

    @relation_arg_2.setter
    def relation_arg_2(self, relation_arg_2: list) -> None:
        self._relation_arg_2 = relation_arg_2

    @property
    def rel_prompt(self) -> typing.Union[typing.List, None]:
        """(list or None): Specifies the relation prompt. Returns None if the relation prompt is not set."""
        return self._rel_prompt

    @rel_prompt.setter
    def rel_prompt(self, rel_prompt: typing.List) -> None:
        self._rel_prompt = rel_prompt

    @property
    def rel_prompt_idx(self) -> typing.Union[typing.List, None]:
        """(list or None): Specifies the list of ids for the tokens in the relation prompt."""
        return self._rel_prompt_idx

    @rel_prompt_idx.setter
    def rel_prompt_idx(self, rel_prompt_idx: typing.List) -> None:
        self._rel_prompt_idx = rel_prompt_idx

    @property
    def rel_prompt_mask(self) -> typing.Union[typing.List, None]:
        """
        (list, None): Specifies the mask for the relation mask with `1` indicating the position of the mask for the
        relation and `0` otherwise.
        """
        return self._rel_prompt_mask

    @rel_prompt_mask.setter
    def rel_prompt_mask(self, rel_prompt_mask: list) -> None:
        self._rel_prompt_mask = rel_prompt_mask

    @property
    def input_token_idx(self) -> typing.List:
        """list: Specifies the list of ids for tokens in the input text."""
        return self._input_token_idx

    @input_token_idx.setter
    def input_token_idx(self, input_token_idx: list) -> None:
        self._input_token_idx = input_token_idx

    @property
    def relation_arg_1_mask(self) -> typing.List:
        """list: Specifies the mask representing the positions of arg_1 entities."""
        return self._relation_arg_1_mask

    @relation_arg_1_mask.setter
    def relation_arg_1_mask(self, relation_arg_1_mask: list) -> None:
        self._relation_arg_1_mask = relation_arg_1_mask

    @property
    def relation_arg_2_mask(self) -> typing.List:
        """list: Specifies the mask representing the positions of arg_2 entities."""
        return self._relation_arg_2_mask

    @relation_arg_2_mask.setter
    def relation_arg_2_mask(self, relation_arg_2_mask: list) -> None:
        self._relation_arg_2_mask = relation_arg_2_mask

    @property
    def relation_type(self) -> str:
        """str: Specifies the relation type being expressed in the input text."""
        return self._relation_type

    @relation_type.setter
    def relation_type(self, relation_type: str) -> None:
        self._relation_type = relation_type

    @property
    def relation_type_id(self) -> int:
        """int: Specifies the id of the relation type."""
        return self._relation_type_id

    @relation_type_id.setter
    def relation_type_id(self, relation_type_id: int) -> None:
        self._relation_type_id = relation_type_id
