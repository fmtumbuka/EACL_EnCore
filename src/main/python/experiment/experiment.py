# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#        Copyright (c) -2022 - Mtumbuka F.                                                    #
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
__version__ = "2022.1"
__date__ = "28 Jul 2022"
__author__ = ""
__maintainer__ = ""
__email__ = ""
__status__ = "Development"

import expbase as xb
import experiment.config as config
import experiment.evaluation_executor as evaluation_executor
import experiment.training_executor as training_executor


APP_DESCRIPTION = "The description of the project."
"""str: The description of the app the will be printed at the beginning of the experiment."""

APP_NAME = "The project name."
"""str: The name of the application."""


class Experiment(xb.Experiment):
    """This class defines the experimental setup."""

    def __init__(self):
        """Creates a new instance of `Experiment`."""
        super().__init__(
            training_executor.TrainingExecutor,
            evaluation_executor.EvaluationExecutor,
            config.Config,
            APP_NAME,
            APP_DESCRIPTION
        )

    def _sanitize_config(self, conf: config.Config) -> None:

        # Make sure just either pretrain or finetune are specified.
        if (int(conf.pretrain) + int(conf.finetune)) > 1:
            raise ValueError(
                "You can only specify one of the two options. Either --pretrain or --finetune and not BOTH."
            )

        # Make sure only one encoder is specified.
        if (int(conf.albert_enc_ent) + int(conf.bert_enc_ent_uncased) + int(conf.roberta_enc_ent)) > 1:
            raise ValueError(
                "You can only use one entity encoder at a time. --albert-enc-ent or --bert-enc-ent or "
                "--roberta-enc-ent."
            )

        if (int(conf.albert_enc_rel) + int(conf.bert_enc_rel) + int(conf.roberta_enc_rel)) > 1:
            raise ValueError(
                "You can only use one relation encoder at a time. --albert-enc-rel or --bert-enc-rel or "
                "--roberta-enc-rel."
            )

        # Make sure that only one of --ht-only and --rel-only is specified
        if (int(conf.rel_only) + int(conf.ht_only)) > 1:
            raise ValueError("You can only specify either --ht-only or --rel-only, not both.")



