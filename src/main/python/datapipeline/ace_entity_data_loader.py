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

import datapipeline.base_data_loader as base_data_loader
import datapipeline.input_example as input_example
import datapipeline.relation_input_example as relation_input_example
import datapipeline.story as story
import json
import os
import typing


class ACEEntityDataLoader(base_data_loader.BaseDataLoader):
    """This implements a loader that loads the ACE2005 dataset for entity classification."""

    BIO_FILE = "entity_BIO.json"
    """str: The file holding the BIO tagged entity data."""

    DEV_PARTITION = "dev"
    """str: The sub-dir holding the dev partition."""

    DEV_REL_FILE = "en-dev.json"
    """str: The file holding the dev partition data."""

    TEST_PARTITION = "test"
    """str: The sub-dir holding the test partition."""

    TEST_REL_FILE = "en-test.json"
    """str: The file holding the test partition data."""

    TOKENS_FILE = "token.json"
    """str: The file holding the tokenized data in each partition."""

    TRAIN_PARTITION = "train"
    """str: The sub-dir holding the train partition."""

    TRAIN_REL_FILE = "en-train.json"
    """str: The file holding the train partition data."""

    def __init__(self, *args, dev: bool = False, test: bool = False, fine_grained: bool = False):
        """

        Args:
            *args: See `base_data_loader.BaseDataLoader.__init__`
            fine_grained ():
            dev (bool): Specifies whether to load data from the dev partition or not.
            test (bool): Specifies whether to load data from the test partition or not.
            train (bool): Specifies whether to load data from the train partition or not.
        """
        super().__init__(*args)

        bio_data_file = None
        rel_data_file = None
        token_data_file = None
        if dev:
            bio_data_file = os.path.join(self.DEV_PARTITION, self.BIO_FILE)
            rel_data_file = os.path.join(self.DEV_PARTITION, self.DEV_REL_FILE)
            token_data_file = os.path.join(self.DEV_PARTITION, self.TOKENS_FILE)

        elif test:
            bio_data_file = os.path.join(self.TEST_PARTITION, self.BIO_FILE)
            rel_data_file = os.path.join(self.TEST_PARTITION, self.TEST_REL_FILE)
            token_data_file = os.path.join(self.TEST_PARTITION, self.TOKENS_FILE)

        else:
            bio_data_file = os.path.join(self.TRAIN_PARTITION, self.BIO_FILE)
            rel_data_file = os.path.join(self.TRAIN_PARTITION, self.TRAIN_REL_FILE)
            token_data_file = os.path.join(self.TRAIN_PARTITION, self.TOKENS_FILE)

        self._bio_data_file = bio_data_file
        self._rel_data_file = rel_data_file
        self._token_data_file = token_data_file
        self._fine_grained = fine_grained

    def load(self) -> typing.Union[
        typing.List[story.Story],
        typing.List[relation_input_example.RelationInputExample],
        typing.List[input_example.InputExample]
    ]:
        samples = []
        bio_data = json.load(open(os.path.join(self._data_path, self._bio_data_file), "r"))
        data = json.load(open(os.path.join(self._data_path, self._rel_data_file), "r"))
        tokens_data = json.load(open(os.path.join(self._data_path, self._token_data_file), "r"))
        for idx, sample in enumerate(data):
            input_tokens = tokens_data[idx]
            labels = ["O"] * len(input_tokens)
            for mention in sample["golden-entity-mentions"]:
                start, end = mention["head"]["position"]
                entity_type = mention["entity-type"].split(":")[0]
                labels[start] = entity_type
            entity_mask = [1 if tag != "O" else 0 for tag in labels]
            samples.append(input_example.InputExample(entity_mask, labels, input_tokens))
            # if idx == 3:
            #
            #     break
        return samples

