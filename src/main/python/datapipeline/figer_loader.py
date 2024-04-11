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
__date__ = "16 Jun 2023"
__author__ = ""
__maintainer__ = ""
__email__ = ""
__status__ = "Development"

import datapipeline.base_data_loader as base_data_loader
import datapipeline.input_example as input_example
import datapipeline.ontonotes_example as ontonotes_example
import datapipeline.relation_input_example as relation_input_example
import datapipeline.story as story
import json
import os
import typing


class FigerLoader(base_data_loader.BaseDataLoader):
    """"""

    DEV_FILE = "dev/dev.json"
    """"""

    TEST_FILE = "test/test.json"
    """"""

    TRAIN_FILE = "train/train.json"
    """"""

    def __init__(
            self,
            *args,
            dev: bool = False,
            test: bool = False,
            order_one: bool = False,
            order_two: bool = False,
            order_three: bool = False
    ):
        """

        Args:
            *args:
            dev:
            test:
            order_one:
            order_two:
            order_three:
        """
        super().__init__(*args)

        self._data_file = None
        if dev:
            self._data_file = os.path.join(self._data_path, self.DEV_FILE)
        elif test:
            self._data_file = os.path.join(self._data_path, self.TEST_FILE)
        else:
            self._data_file = os.path.join(self._data_path, self.TRAIN_FILE)

        self._dev = dev
        self._test = test
        self._order_one = order_one
        self._order_two = order_two
        self._order_three = order_three

    def load(self) -> typing.Union[
        typing.List[story.Story],
        typing.List[relation_input_example.RelationInputExample],
        typing.List[input_example.InputExample],
        typing.List[ontonotes_example.OntoNotesExample]
    ]:
        samples = []
        data = open(self._data_file).readlines()
        print("Reading data from: {}".format(self._data_file))
        for idx, line in enumerate(data):
            line = json.loads(line)
            for mention in line["mentions"]:
                mention_start = mention["start"]
                mention_end = mention["end"]
                labels = mention["labels"]
                if self._test and self._order_one:
                    if len(labels) == 1:
                        samples.append(
                            ontonotes_example.OntoNotesExample(
                                input_tokens=line["tokens"],
                                entity_start=mention_start,
                                entity_end=mention_end,
                                entity_head_pos=mention_start,
                                entity_labels=labels
                            )
                        )
                    else:
                        continue
                elif self._test and self._order_two:
                    if len(labels) == 2:
                        samples.append(
                            ontonotes_example.OntoNotesExample(
                                input_tokens=line["tokens"],
                                entity_start=mention_start,
                                entity_end=mention_end,
                                entity_head_pos=mention_start,
                                entity_labels=labels
                            )
                        )
                    else:
                        continue
                elif self._test and self._order_three:
                    if len(labels) == 3:
                        samples.append(
                            ontonotes_example.OntoNotesExample(
                                input_tokens=line["tokens"],
                                entity_start=mention_start,
                                entity_end=mention_end,
                                entity_head_pos=mention_start,
                                entity_labels=labels
                            )
                        )
                    else:
                        continue
                else:
                    samples.append(
                        ontonotes_example.OntoNotesExample(
                            input_tokens=line["tokens"],
                            entity_start=mention_start,
                            entity_end=mention_end,
                            entity_head_pos=mention_start,
                            entity_labels=labels
                        )
                    )
            # if idx == 6:
            #     break
            #
            # exit()
            # (start, end, words, labels, features) = line.strip().split("\t")
            #
            # input_tokens = words.rstrip().split()
            # entity_start = int(start)
            # entity_end = int(end)
            # entity_labels = labels.split()
            # span_head = [feat.split("|")[-1] for feat in features.split() if feat.split("|")[0] == "HEAD"]
            # remaining = input_tokens[entity_start:entity_end]
            #
            # span_head_str = " ".join(span_head)
            # index_remaining = None
            # try:
            #     index_remaining = remaining.index(" ".join(span_head))
            # except Exception as e:
            #     entity_start = entity_start - 1
            #     remaining = input_tokens[entity_start:entity_end]
            #     try:
            #         index_remaining = remaining.index(" ".join(span_head))
            #     except Exception as e:
            #         continue
            #
            # head_position = entity_start + index_remaining
            #
            # if self._test and self._order_one:
            #     if len(entity_labels) == 1:
            #         samples.append(
            #             ontonotes_example.OntoNotesExample(
            #                 input_tokens=input_tokens,
            #                 entity_start=entity_start,
            #                 entity_end=entity_end,
            #                 entity_head_pos=head_position,
            #                 entity_labels=entity_labels
            #             )
            #         )
            #     else:
            #         continue
            # elif self._test and self._order_two:
            #     if len(entity_labels) == 2:
            #         samples.append(
            #             ontonotes_example.OntoNotesExample(
            #                 input_tokens=input_tokens,
            #                 entity_start=entity_start,
            #                 entity_end=entity_end,
            #                 entity_head_pos=head_position,
            #                 entity_labels=entity_labels
            #             )
            #         )
            #     else:
            #         continue
            # elif self._test and self._order_three:
            #     if len(entity_labels) == 3:
            #         samples.append(
            #             ontonotes_example.OntoNotesExample(
            #                 input_tokens=input_tokens,
            #                 entity_start=entity_start,
            #                 entity_end=entity_end,
            #                 entity_head_pos=head_position,
            #                 entity_labels=entity_labels
            #             )
            #         )
            #     else:
            #         continue
            # else:
            #     samples.append(
            #         ontonotes_example.OntoNotesExample(
            #             input_tokens=input_tokens,
            #             entity_start=entity_start,
            #             entity_end=entity_end,
            #             entity_head_pos=head_position,
            #             entity_labels=entity_labels
            #         )
            #     )

        print("OK")
        print()

        return samples
