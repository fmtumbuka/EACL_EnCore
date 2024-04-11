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

import datapipeline.base_data_loader as base_data_loader
import datapipeline.cluster as cluster
import datapipeline.giga_word_dataset as gigaword_dataset
import datapipeline.relation_input_example as relation_input_example
import datapipeline.sentence as sentence
import datapipeline.story as story
import expbase.util as util
import os
import json
import typing
from transformers import AutoTokenizer


class GigaWordLoader(base_data_loader.BaseDataLoader):
    """This loads the Giga word corpus."""
    def __init__(self, *args, tokenizer_name: str = None):
        """

        Args:
            *args: See method: base_data_loader.BaseDataLoader.__init__
            tokenizer_name (str): The name of the tokenizer being used.
        """
        super().__init__(*args)
        self._tokenizer = tokenizer_name
        self._files = self._get_files()

    def _get_files(self) -> typing.List[str]:
        """files (list[str]): This retrieves all files in the specified directory."""
        files = []
        for r, d, f in os.walk(self._data_path):
            for file in f:
                if file.endswith(".json"):
                    files.append(os.path.join(r, file))
        return files

    def _pickle_part(self, data: typing.List[story.Story]) -> None:
        """
        This method pickles part of the loaded data.
        Args:
            data (list[:class::`story.Story`]): The list of stories.
        """
        data_loader = GigaWordLoader(self._data_path)
        tokenizer = AutoTokenizer.from_pretrained(self._tokenizer)
        dataset = gigaword_dataset.GigaWordDataset(data_loader, tokenizer, data)
        gigaword_data_path = os.path.join(self._data_path, "gigaword")
        pickle_file = f"{gigaword_data_path}.{self._tokenizer}.data-{len(data)}.pickle"
        print("Pickling dataset to {} for faster loading next time...".format(pickle_file))
        util.better_pickle.pickle_dump(dataset, pickle_file)
        print("OK")

    def load(self) -> typing.Union[typing.List[story.Story], typing.List[relation_input_example.RelationInputExample]]:

        stories = []  # To hold all stories.
        story_idx = 0
        for file_idx, file in enumerate(self._get_files()):
            data = open(file)
            data = json.load(data)
            data = data["stories"]
            for s_idx, single_story in enumerate(data):
                story_clusters = []
                # Process text.
                single_story = [line.rstrip() for line in single_story]
                single_story = " ".join(single_story)
                story_doc = self._nlp(single_story)

                """
                A list of sentences
                """
                sentence_list = []
                for sent in story_doc.sents:
                    sentence_list.append(
                        sentence.Sentence(sent.end, sent.start, sent.text)
                    )

                # Process sentences
                clusters = []
                for chain in story_doc._.coref_chains:
                    mentions = []
                    cluster_sents = []
                    for mention in chain:
                        mentions += mention

                    for m in mentions:
                        for sent in sentence_list:
                            if sent.start <= m <= sent.end:
                                position = m - sent.start
                                current_doc = self._nlp(sent.text)
                                current_doc_tokens = [token.text for token in current_doc]
                                entity_mask = [0 if idx != position else 1 for idx, _ in enumerate(current_doc_tokens)]
                                try:
                                    current_doc_tokens[position]
                                except IndexError:
                                    continue
                                sent.text_arr = current_doc_tokens

                                # Check if current sentence is already in the cluster.
                                equal_sents = [int(sent == sent_i) for sent_i in cluster_sents]
                                if sum(equal_sents) > 0:
                                    sent_index = equal_sents.index(1)
                                    existing_mask = cluster_sents[sent_index].entity_mask
                                    merged_mask = [
                                        1 if (old_mask == 1 or new_mask == 1) else 0 for old_mask, new_mask in zip(existing_mask, entity_mask)
                                    ]
                                    cluster_sents[sent_index].entity_mask = merged_mask
                                else:
                                    sent.entity_mask = entity_mask
                                    cluster_sents.append(sent)

                    clusters.append(cluster_sents)

                for idx, sent_cluster in enumerate(clusters):
                    if len(sent_cluster) > 0:
                        new_cluster = cluster.Cluster(idx)
                        for sent in sent_cluster:
                            new_cluster.add_sentence(sent)

                        story_clusters.append(new_cluster)
                stories.append(
                    story.Story(story_clusters, story_idx)
                )
                if s_idx % 100 == 0:
                    print("Processed sample {}/{} in file {}/{}".format(s_idx+1, len(data), file_idx, len(self._files)))
                # if story_idx == 4:
                #     break
                story_idx += 1
            # break
            self._pickle_part(stories)
        return stories

