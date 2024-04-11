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


import datapipeline.sentence as sentence
import insanity
import typing


class Cluster(object):
    """This is an abstraction of a co-reference chain."""

    def __init__(self, cluster_id: int):
        """
        This creates an instance of `Cluster`.
        Args:
            cluster_id (int): The id of the co-reference chain in the story.
        """
        # Sanitize args.
        insanity.sanitize_type("cluster_id", cluster_id, int)

        # Store args.
        self._cluster_id = cluster_id
        self._sentences = None

    def __len__(self) -> int:
        """int: This specifies the size of the cluster in terms of the number of sentences."""
        num_sentences = 0
        if self._sentences is not None:
            num_sentences = len(self._sentences)

        return num_sentences

    # Properties.
    @property
    def cluster_id(self) -> int:
        """int: Specifies the cluster id."""
        return self._cluster_id

    @property
    def sentences(self) -> typing.Union[typing.List[sentence.Sentence], None]:
        """list[::class:`sentence.Sentence`]: Returns a list of sentences in a given cluster."""
        return self._sentences

    # Methods.
    def add_sentence(self, sent: sentence.Sentence) -> None:
        """
        This method adds sentences to a cluster.
        Args:
            sent (::class:`sentence.Sentence`): A sentence to be added to a Cluster.
        """
        # Sanitize arg.
        insanity.sanitize_type("sent", sent, sentence.Sentence)

        # For the first addition, create the list and add the first element.
        if self._sentences is None:
            self._sentences = []

        self._sentences.append(sent)

