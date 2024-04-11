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
__date__ = "02 Mar 2023"
__author__ = ""
__maintainer__ = ""
__email__ = ""
__status__ = "Development"

import insanity
import typing


class Sentence(object):
    """This class represents a single sentence that serves as input."""

    def __init__(
            self,
            end: int,
            start: int,
            text: str,
            entity_mask = None,
            text_arr: list = None,
    ):
        """
        This creates an instance of `Sentence`.
        Args:
            end (int): The end index of the sentence in a given document.
            start (int): The starting index of the sentence in a given document.
            text (str): The actual text of the sentence in a given document.
            mention_position (int): The position of the current mention in the sentence.
            mention_positions (list): The positions of co-referring entities in a sentence.Default value is None.
            text_arr (list): A list of tokens in the input sentence.
            text_augmented (str): The text with special tokens around co-referring entities in a sentence.
            tokenized_text_mask (list): A mask indicating the positions of entity mentions in tokenized text. `1`
                indicates the position of the entity mention, and `0` indicates otherwise.
            tokenizer_text_idx (list): A list of token ids for the input text as given by the specified tokenizer.
        """

        # Sanitize args.
        insanity.sanitize_type("end", end, int)
        insanity.sanitize_type("start", start, int)
        insanity.sanitize_type("text", text, str)
        if text_arr is not None:
            insanity.sanitize_type("text_arr", text_arr, list)

        # Store args
        self._end = end
        self._start = start
        self._text = text
        self._entity_mask = entity_mask
        self._text_arr = text_arr
        self._label_placeholders = []
        self._token_ids = None

    def __eq__(self, other):
        return (
            isinstance(other, Sentence) and
            other.text == self.text and
            other.start == self.start and
            other.end == self.end
        )

    @property
    def end(self) -> int:
        """int: Specifies the end index of a given sentence in the given document."""
        return self._end

    @property
    def entity_mask(self) -> typing.List[int]:
        """entity_mask (list[int]): Specifies the positions of corefering entities using `1` and `0` otherwise."""
        return self._entity_mask

    @entity_mask.setter
    def entity_mask(self, entity_mask: typing.List[int]) -> None:
        self._entity_mask = entity_mask

    @property
    def label_placeholders(self) -> typing.List[int]:
        """label_placeholders (list[int]): Specifies default placeholders for co-referring entities."""
        return self._label_placeholders

    @label_placeholders.setter
    def label_placeholders(self, label_placeholders: typing.List[int]) -> None:
        self._label_placeholders = label_placeholders

    @property
    def start(self) -> int:
        """int: Specifies the start index of a given sentence in the given document."""
        return self._start

    @property
    def text(self) -> str:
        """str: Specifies the text of the given sentence."""
        return self._text

    @property
    def text_arr(self) -> list:
        """list: Specifies the list of tokens in the sentence as given by spacy."""
        return self._text_arr

    @text_arr.setter
    def text_arr(self, text_arr: list) -> None:
        self._text_arr = text_arr

    @property
    def token_ids(self) -> typing.List[int]:
        """token_ids (list[int]): Specifies the vocabulary ids of the tokens in the input sentence."""
        return self._token_ids

    @token_ids.setter
    def token_ids(self, token_ids: typing.List[int]) -> None:
        self._token_ids = token_ids

