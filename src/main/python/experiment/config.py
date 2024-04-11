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

import argmagic.decorators as decorators
import expbase as xb
import insanity
import os
import typing
import torch


class Config(xb.BaseConfig):
    """This class has all the configurations for an experiment."""

    DEFAULT_ACE_DATASET = False
    """bool: The default setting for loading the ACE2005 dataset."""

    DEFAULT_ACE_FINE_GRAINED = False
    """bool: The default setting for using fine-grained relation types or  not."""

    DEFAULT_ALBERT_ENC_ENT = False
    """bool: The default setting for using an ALBERT based encoder for entities."""

    DEFAULT_ALBERT_ENC_REL = False
    """bool: The default setting for using an ALBERT based encoder for relations."""

    DEFAULT_BATCH_SIZE = 256
    """int: The default value for batch size."""

    DEFAULT_BERT_ENC_ENT_CASED = False
    """bool: The default setting for using the BERT based encoder for entities."""

    DEFAULT_BERT_ENC_ENT_UNCASED = False
    """bool: The default setting for using the BERT based encoder for entities."""

    DEFAULT_BERT_ENC_REL = False
    """bool: The default setting for using the BERT based encoder for relations."""

    DEFAULT_CLASSIFIER_LAYERS = 1
    """int: The default number of layers in the classifiers."""

    DEFAULT_CP_LAYER = False
    """bool: The default setting for using the contrastive loss intermediate layer."""

    DEFAULT_CP_LOSS_TAU = 0.07
    """float: The default value for the contrastive loss temperature."""

    DEFAULT_DATA_DIR = ""
    """str: The default value for the data directory."""

    DEFAULT_DEV_PART = False
    """bool: The default setting for loading data from the dev partition."""

    DEFAULT_DROPOUT_RATE = 0.0
    """float: The dropout rate to use (across all components that use dropout)."""

    DEFAULT_ENTITY_ONLY = False
    """bool: The default setting for considering entity positions only when computing loss."""

    DEFAULT_EVAL_BATCH_SIZE = 1
    """int: The default batch size during evaluation."""

    DEFAULT_EVAL_GPU = False
    """bool: The default setting for running model evaluation on gpu."""

    DEFAULT_FIGER = False

    DEFAULT_FINETUNE = False
    """bool: The default setting for fine-tuning the model."""

    DEFAULT_FULL_SPAN_MASK = False
    """bool: The default setting for masking the entire span."""

    DEFAULT_FREEZE_ENT_ENC = False
    """bool: The default setting for freezing the entity encoder during the experiment."""

    DEFAULT_FREEZE_REL_ENC = False
    """bool: The default setting for freezing the relation encoder during the experiment."""

    DEFAULT_GPU = False
    """bool: The default setting for using GPU when training."""

    DEFAULT_GRAD_ACC_ITERS = 1
    """int: The default number of iterations to accumulate gradients over."""

    DEFAULT_HT_ONLY = False
    """bool: The default setting for considering only head and tail embeddings to predict relation type."""

    DEFAULT_LEARNING_RATE = 0.00005
    """float: The default value for the model learning rate."""

    DEFAULT_MAX_GRAD_NORM = 2.0
    """float: The default maximum norm that any gradients are clipped to."""

    DEFAULT_MERGE_DATASETS = False
    """bool: The default setting for merging pickled datasets."""

    DEFAULT_MLM_PERCENTAGE = 0.15
    """float: The default value for masking percentage when performing MLM."""

    DEFAULT_MLP_CLASSIFIER = False
    """bool: The default setting for using the mlp classifier."""

    DEFAULT_NUM_EPOCHS = 1
    """int: The default value for the number of training epochs."""

    DEFAULT_ONTONOTES = False
    """bool: The default setting for using OntoNotes dataset."""

    DEFAULT_ORDER_ONE = False
    """bool: The default setting for only considering order one labels in fine-grained typing during evaluation."""

    DEFAULT_ORDER_THREE = False
    """bool: The default setting for only considering order three labels in fine-grained typing during evaluation."""

    DEFAULT_ORDER_TWO = False
    """bool: The default setting for only considering order two labels in fine-grained typing during evaluation."""

    DEFAULT_PREDICTION_THRESHOLD = 0.5
    """float: The default probability threshold for setting the prediction as true."""

    DEFAULT_PRETRAIN = False
    """bool: The default setting for pre-training the encoder."""

    DEFAULT_PRINT_INT = 1
    """int: The default value for logging details when running experiments."""

    DEFAULT_PRINT_PREDICTIONS = False
    """bool: The default setting for printing predictions to a file during evaluation."""

    DEFAULT_REL_MASK = False
    """bool: The default setting for using the mask token to encode relations or not."""

    DEFAULT_REL_ONLY = False
    """bool: The default setting for only using the relation encoder only."""

    DEFAULT_ROBERTA_ENC_ENT = False
    """bool: The default setting for using a RoBERTa based encoder for entities."""

    DEFAULT_ROBERTA_ENC_REL = False
    """bool: The default setting for using a RoBERTa based encoder for relations."""

    DEFAULT_SENTENCES = False
    """bool: The default setting for loading individual sentences rather than stories."""

    def __init__(self):
        super().__init__()
        self._ace_dataset = self.DEFAULT_ACE_DATASET
        self._ace_fine_grained = self.DEFAULT_ACE_FINE_GRAINED
        self._albert_enc_ent = self.DEFAULT_ALBERT_ENC_ENT
        self._albert_enc_rel = self.DEFAULT_ALBERT_ENC_REL
        self._batch_size = self.DEFAULT_BATCH_SIZE
        self._bert_enc_ent_cased = self.DEFAULT_BERT_ENC_ENT_CASED
        self._bert_enc_ent_uncased = self.DEFAULT_BERT_ENC_ENT_UNCASED
        self._bert_enc_rel = self.DEFAULT_BERT_ENC_REL
        self._classifier_layers = self.DEFAULT_CLASSIFIER_LAYERS
        self._cp_layer = self.DEFAULT_CP_LAYER
        self._cp_loss_tau = self.DEFAULT_CP_LOSS_TAU
        self._data_dir = self.DEFAULT_DATA_DIR
        self._dev_part = self.DEFAULT_DEV_PART
        self._dropout_rate = self.DEFAULT_DROPOUT_RATE
        self._entity_only = self.DEFAULT_ENTITY_ONLY
        self._eval_batch_size = self.DEFAULT_EVAL_BATCH_SIZE
        self._eval_gpu = self.DEFAULT_EVAL_GPU
        self._figer = self.DEFAULT_FIGER
        self._finetune = self.DEFAULT_FINETUNE
        self._full_span_mask = self.DEFAULT_FULL_SPAN_MASK
        self._freeze_ent_enc = self.DEFAULT_FREEZE_ENT_ENC
        self._freeze_rel_enc = self.DEFAULT_FREEZE_REL_ENC
        self._gpu = self.DEFAULT_GPU
        self._grad_acc_iters = self.DEFAULT_GRAD_ACC_ITERS
        self._ht_only = self.DEFAULT_HT_ONLY
        self._learning_rate = self.DEFAULT_LEARNING_RATE
        self._num_epochs = self.DEFAULT_NUM_EPOCHS
        self._merge_datasets = self.DEFAULT_MERGE_DATASETS
        self._max_grad_norm = self.DEFAULT_MAX_GRAD_NORM
        self._mlm_percentage = self.DEFAULT_MLM_PERCENTAGE
        self._mlp_classifier = self.DEFAULT_MLP_CLASSIFIER
        self._ontonotes = self.DEFAULT_ONTONOTES
        self._order_one = self.DEFAULT_ORDER_ONE
        self._order_three = self.DEFAULT_ORDER_THREE
        self._order_two = self.DEFAULT_ORDER_TWO
        self._prediction_threshold = self.DEFAULT_PREDICTION_THRESHOLD
        self._pretrain = self.DEFAULT_PRETRAIN
        self._print_int = self.DEFAULT_PRINT_INT
        self._print_predictions = self.DEFAULT_PRINT_PREDICTIONS
        self._rel_mask = self.DEFAULT_REL_MASK
        self._rel_only = self.DEFAULT_REL_ONLY
        self._roberta_enc_ent = self.DEFAULT_ROBERTA_ENC_ENT
        self._roberta_enc_rel = self.DEFAULT_ROBERTA_ENC_REL
        self._sentences = self.DEFAULT_SENTENCES
        self._checkpoint = None
        self._checkpoints_dir = None
        self._num_classes = 1

    # Properties
    @decorators.optional
    @property
    def ace_dataset(self) -> bool:
        """bool: Specifies whether to load the ACE2005 dataset or not."""
        return self._ace_dataset

    @ace_dataset.setter
    def ace_dataset(self, ace_dataset: bool) -> None:
        self._ace_dataset = bool(ace_dataset)

    @property
    def ace_fine_grained(self) -> bool:
        """bool: Specifies whether to load fine-grained ACE or not."""
        return self._ace_fine_grained

    @ace_fine_grained.setter
    def ace_fine_grained(self, ace_fine_grained: bool) -> None:
        self._ace_fine_grained = bool(ace_fine_grained)

    @property
    def albert_enc_ent(self) -> bool:
        """bool: Specifies whether to use an ALBERT based encoder or not for entities."""
        return self._albert_enc_ent

    @albert_enc_ent.setter
    def albert_enc_ent(self, albert_enc_ent: bool) -> None:
        self._albert_enc_ent = bool(albert_enc_ent)

    @property
    def albert_enc_rel(self) -> bool:
        """bool: Specifies whether to use an ALBERT based encoder or not for relations."""
        return self._albert_enc_rel

    @albert_enc_rel.setter
    def albert_enc_rel(self, albert_enc_rel: bool) -> None:
        self._albert_enc_rel = bool(albert_enc_rel)

    @property
    def batch_size(self) -> int:
        """int: Specifies the batch size used during training."""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int) -> None:
        insanity.sanitize_type("batch_size", batch_size, int)
        insanity.sanitize_range("batch_size", batch_size, minimum=1)
        self._batch_size = batch_size

    @property
    def bert_enc_ent_cased(self) -> bool:
        """bool: Specifies whether to use BERT based encoder or not for entities."""
        return self._bert_enc_ent_cased

    @bert_enc_ent_cased.setter
    def bert_enc_ent_cased(self, bert_enc_ent_cased: bool) -> None:
        self._bert_enc_ent_cased = bool(bert_enc_ent_cased)

    @property
    def bert_enc_ent_uncased(self) -> bool:
        """bool: Specifies whether to use BERT based encoder or not for entities."""
        return self._bert_enc_ent_uncased

    @bert_enc_ent_uncased.setter
    def bert_enc_ent_uncased(self, bert_enc_ent_uncased: bool) -> None:
        self._bert_enc_ent_uncased = bool(bert_enc_ent_uncased)

    @property
    def bert_enc_rel(self) -> bool:
        """bool: Specifies whether to use BERT based encoder or not for relations."""
        return self._bert_enc_rel

    @bert_enc_rel.setter
    def bert_enc_rel(self, bert_enc_rel: bool) -> None:
        self._bert_enc_rel = bool(bert_enc_rel)

    @decorators.optional
    @property
    def checkpoint(self) -> typing.Optional[str]:
        """str: The path of a checkpoint to load the model state from."""
        return self._checkpoint

    @checkpoint.setter
    def checkpoint(self, checkpoint: str) -> None:
        checkpoint = str(checkpoint)
        if not os.path.isfile(checkpoint):
            raise ValueError("The provided <checkpoint> does not refer to an existing file: '{}'".format(checkpoint))
        self._checkpoint = checkpoint

    @decorators.optional
    @property
    def checkpoints_dir(self) -> typing.Optional[str]:
        """str: The path where checkpoints to evaluate are."""
        return self._checkpoints_dir

    @checkpoints_dir.setter
    def checkpoints_dir(self, checkpoints_dir: str) -> None:
        checkpoints_dir = str(checkpoints_dir)
        if not os.path.isdir(checkpoints_dir):
            raise ValueError(
                "The provided <checkpoint> does not refer to an existing file: '{}'".format(checkpoints_dir))
        self._checkpoints_dir = checkpoints_dir

    @property
    def classifier_layers(self) -> int:
        """int: Specifies the number of layers in the classifier."""
        return self._classifier_layers

    @classifier_layers.setter
    def classifier_layers(self, classifier_layers: int) -> None:
        insanity.sanitize_type("classifier_layers", classifier_layers, int)
        insanity.sanitize_range("classifier_layers", classifier_layers, minimum=1)
        self._classifier_layers = int(classifier_layers)

    @property
    def cp_layer(self) -> bool:
        """bool: Specifies whether to use the contrastive loss intermediate layer or not."""
        return self._cp_layer

    @cp_layer.setter
    def cp_layer(self, cp_layer: bool) -> None:
        self._cp_layer = bool(cp_layer)

    @property
    def cp_loss_tau(self) -> float:
        """float: Specifies the tau in the contrastive loss equation."""
        return self._cp_loss_tau

    @cp_loss_tau.setter
    def cp_loss_tau(self, cp_loss_tau: float) -> None:
        insanity.sanitize_type("cp_loss_tau", cp_loss_tau, float)
        insanity.sanitize_range("cp_loss_tau", cp_loss_tau, minimum=0.0)
        self._cp_loss_tau = cp_loss_tau

    @property
    def data_dir(self) -> str:
        """str: Specifies the path to the data directory."""
        return self._data_dir

    @data_dir.setter
    def data_dir(self, data_dir: str) -> None:
        # TODO sanitize
        self._data_dir = str(data_dir)

    @property
    def dev_part(self) -> bool:
        """bool: Specifies whether to load data from the dev partition or not."""
        return self._dev_part

    @dev_part.setter
    def dev_part(self, dev_part: bool) -> None:
        self._dev_part = bool(dev_part)

    @property
    def dropout_rate(self) -> float:
        """float: The dropout rate to use (across all components that use dropout)."""
        return self._dropout_rate

    @dropout_rate.setter
    def dropout_rate(self, dropout_rate: float) -> None:
        insanity.sanitize_type("dropout_rate", dropout_rate, float)
        dropout_rate = float(dropout_rate)
        insanity.sanitize_range("dropout_rate", dropout_rate, minimum=0, maximum=1, max_inclusive=False)
        self._dropout_rate = dropout_rate

    @property
    def entity_only(self) -> bool:
        """bool: Specifies whether to use the entity positions only when computing the training loss or not"""
        return self._entity_only

    @entity_only.setter
    def entity_only(self, entity_only: bool) -> None:
        self._entity_only = bool(entity_only)

    @property
    def eval_batch_size(self) -> int:
        """int: Specifies the batch size used during evaluation."""
        return self._eval_batch_size

    @eval_batch_size.setter
    def eval_batch_size(self, eval_batch_size: int) -> None:
        insanity.sanitize_type("eval_batch_size", eval_batch_size, int)
        insanity.sanitize_range("eval_batch_size", eval_batch_size, minimum=1)
        self._eval_batch_size = eval_batch_size

    @property
    def eval_gpu(self) -> bool:
        """bool: Specifies whether to use GPU during model evaluation or not."""
        return self._eval_gpu

    @eval_gpu.setter
    def eval_gpu(self, eval_gpu: bool) -> None:
        if eval_gpu and not torch.cuda.is_available():
            raise ValueError("There is no GPU on the local machine.")
        self._eval_gpu = bool(eval_gpu)

    @property
    def figer(self) -> bool:
        return self._figer

    @figer.setter
    def figer(self, figer: bool) -> None:
        self._figer = bool(figer)

    @property
    def finetune(self) -> bool:
        """bool: Specifies whether to finetune the model or not."""
        return self._finetune

    @finetune.setter
    def finetune(self, finetune: bool) -> None:
        self._finetune = bool(finetune)

    @property
    def full_span_mask(self) -> bool:
        """bool: Specifies whether to mask the full entity span or not."""
        return self._full_span_mask

    @full_span_mask.setter
    def full_span_mask(self, full_span_mask: bool) -> None:
        self._full_span_mask = bool(full_span_mask)

    @property
    def freeze_ent_enc(self) -> bool:
        """bool: Specifies whether to freeze the entity encoder during the experiment or not."""
        return self._freeze_ent_enc

    @freeze_ent_enc.setter
    def freeze_ent_enc(self, freeze_ent_enc: bool) -> None:
        self._freeze_ent_enc = bool(freeze_ent_enc)

    @property
    def freeze_rel_enc(self) -> bool:
        """bool: Specifies whether to freeze the relation encoder during the experiment or not."""
        return self._freeze_rel_enc

    @freeze_rel_enc.setter
    def freeze_rel_enc(self, freeze_rel_enc: bool) -> None:
        self._freeze_rel_enc = bool(freeze_rel_enc)

    @property
    def gpu(self) -> bool:
        """bool: Specifies whether to use GPU during model training or not."""
        return self._gpu

    @gpu.setter
    def gpu(self, gpu: bool) -> None:
        if gpu and not torch.cuda.is_available():
            raise ValueError("There is no GPU on the local machine.")
        self._gpu = bool(gpu)

    @property
    def grad_acc_iters(self) -> int:
        """int: The number of iterations to accumulate gradients over."""
        return self._grad_acc_iters

    @grad_acc_iters.setter
    def grad_acc_iters(self, grad_acc_iters: int) -> None:
        insanity.sanitize_type("grad_acc_iters", grad_acc_iters, int)
        insanity.sanitize_range("grad_acc_iters", grad_acc_iters, minimum=1)
        self._grad_acc_iters = int(grad_acc_iters)

    @property
    def ht_only(self) -> bool:
        """bool: Specifies whether to only consider head and tail entity embeddings of relation classification or not"""
        return self._ht_only

    @ht_only.setter
    def ht_only(self, ht_only: bool) -> None:
        self._ht_only = bool(ht_only)

    @property
    def learning_rate(self) -> float:
        """float: Specifies the learning rate for the model during training."""
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate: float) -> None:
        self._learning_rate = float(learning_rate)

    @decorators.optional
    @property
    def max_grad_norm(self) -> float:
        """float: The maximum norm that any gradients are clipped to."""
        return self._max_grad_norm

    @max_grad_norm.setter
    def max_grad_norm(self, max_grad_norm: float) -> None:
        self._max_grad_norm = float(max_grad_norm)

    @property
    def merge_datasets(self) -> bool:
        """bool: Specifies whether to merge datasets or not."""
        return self._merge_datasets

    @merge_datasets.setter
    def merge_datasets(self, merge_datasets: bool) -> None:
        self._merge_datasets = bool(merge_datasets)

    @property
    def mlm_percentage(self) -> float:
        """float: Specifies the percentage of tokens to be masked out in a given batch."""
        return self._mlm_percentage

    @mlm_percentage.setter
    def mlm_percentage(self, mlm_percentage: float) -> None:
        insanity.sanitize_type("mlm_percentage", mlm_percentage, float)
        insanity.sanitize_range("mlm_percentage", mlm_percentage, minimum=0.00)
        insanity.sanitize_range("mlm_percentage", mlm_percentage, maximum=1.0)
        self._mlm_percentage = float(mlm_percentage)

    @property
    def mlp_classifier(self) -> bool:
        """bool: Specifies whether to use an MLP classifier or not"""
        return self._mlp_classifier

    @mlp_classifier.setter
    def mlp_classifier(self, mlp_classifier: bool) -> None:
        self._mlp_classifier = bool(mlp_classifier)

    @decorators.optional
    @property
    def num_classes(self) -> int:
        """int: Specifies the number of classification classes."""
        return self._num_classes

    @num_classes.setter
    def num_classes(self, num_classes: int) -> None:
        insanity.sanitize_type("num_classes", num_classes, int)
        insanity.sanitize_range("num_classes", num_classes, minimum=1)
        self._num_classes = int(num_classes)

    @property
    def num_epochs(self) -> int:
        """int: Specifies the number of training epochs."""
        return self._num_epochs

    @num_epochs.setter
    def num_epochs(self, num_epochs: int) -> None:
        insanity.sanitize_type("num_epochs", num_epochs, int)
        insanity.sanitize_range("num_epochs", num_epochs, minimum=1)
        self._num_epochs = int(num_epochs)

    @property
    def ontonotes(self) -> bool:
        """bool: Specifies whether to load OntoNotes or not."""
        return self._ontonotes

    @ontonotes.setter
    def ontonotes(self, ontonotes: bool) -> None:
        self._ontonotes = bool(ontonotes)

    @property
    def order_one(self) -> bool:
        """bool: Specifies whether to only consider first order labels or not."""
        return self._order_one

    @order_one.setter
    def order_one(self, order_one: bool) -> None:
        self._order_one = bool(order_one)

    @property
    def order_three(self) -> bool:
        """bool: Specifies whether to only consider third order labels or not."""
        return self._order_three

    @order_three.setter
    def order_three(self, order_three: bool) -> None:
        self._order_three = bool(order_three)

    @property
    def order_two(self) -> bool:
        """bool: Specifies whether to only consider second order labels or not."""
        return self._order_two

    @order_two.setter
    def order_two(self, order_two: bool) -> None:
        self._order_two = bool(order_two)


    @property
    def prediction_threshold(self) -> float:
        """float: Specifies the probability threshold for setting the prediction as true."""
        return self._prediction_threshold

    @prediction_threshold.setter
    def prediction_threshold(self, prediction_threshold: float) -> None:
        self._prediction_threshold = float(prediction_threshold)

    @property
    def pretrain(self) -> bool:
        """bool: Specifies whether to pretrain the encoder or not."""
        return self._pretrain

    @pretrain.setter
    def pretrain(self, pretrain: bool) -> None:
        self._pretrain = bool(pretrain)

    @property
    def print_int(self) -> int:
        """int: Specifies the logging interval for experiment details during the experiment."""
        return self._print_int

    @print_int.setter
    def print_int(self, print_int) -> None:
        self._print_int = print_int

    @decorators.optional
    @property
    def print_predictions(self) -> bool:
        """bool: Specifies whether to print predictions during evaluation or not."""
        return self._print_predictions

    @print_predictions.setter
    def print_predictions(self, print_predictions: bool) -> None:
        self._print_predictions = bool(print_predictions)

    @property
    def rel_mask(self) -> bool:
        """bool: Specifies whether to use the mask token to encode the relation or not."""
        return self._rel_mask

    @rel_mask.setter
    def rel_mask(self, rel_mask: bool) -> None:
        self._rel_mask = bool(rel_mask)

    @property
    def rel_only(self) -> bool:
        """bool: Specifies whether to use the relation encoder only."""
        return self._rel_only

    @rel_only.setter
    def rel_only(self, rel_only: bool) -> None:
        self._rel_only = bool(rel_only)

    @property
    def roberta_enc_ent(self) -> bool:
        """bool: Specifies whether to use a RoBERTa based encoder or not for entities."""
        return self._roberta_enc_ent

    @roberta_enc_ent.setter
    def roberta_enc_ent(self, roberta_enc_ent: bool) -> None:
        self._roberta_enc_ent = bool(roberta_enc_ent)

    @property
    def roberta_enc_rel(self) -> bool:
        """bool: Specifies whether to use a RoBERTa based encoder or not for relations."""
        return self._roberta_enc_rel

    @roberta_enc_rel.setter
    def roberta_enc_rel(self, roberta_enc_rel: bool) -> None:
        self._roberta_enc_rel = bool(roberta_enc_rel)

    @property
    def sentences(self) -> bool:
        """bool: Specifies whether to load data as individual sentences or not."""
        return self._sentences

    @sentences.setter
    def sentences(self, sentences: bool) -> None:
        self._sentences = bool(sentences)
