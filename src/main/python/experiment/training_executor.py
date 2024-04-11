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

import itertools

import accelerate
import collections
import expbase as xb
import expbase.util as util
import experiment
import experiment.component_factory as component_factory
import experiment.encore_model_checkpoint as encore_model_checkpoint
import experiment.entity_encoder_checkpoint as entity_encoder_checkpoint
import functools
import json
import numpy as np
import operator
import shutil
import torch
import torch.nn as nn
import torch.nn.utils as utils
import torch.utils.data as data
import os


class TrainingExecutor(xb.TrainingExecutor):
    """This implements the training routing for an experiment based on the configurations given."""

    PRETRAIN_PROTO = "pretrain.proto"

    def __init__(self, *args, **kwargs):
        """This creates an instance of the `TrainingExecutor`."""
        super().__init__(*args, **kwargs)

        # Attributes
        self._cross_entropy_loss = None
        self._dataset = None
        self._model = None
        self._optimizer = None
        self._start_epoch = 0

    # Methods

    def _init(self) -> None:
        # Call routines in the Component factory to create the attributes above
        self._cross_entropy_loss = nn.CrossEntropyLoss()
        self._dataset = component_factory.ComponentFactory.create_dataset(self._conf)
        self._model = component_factory.ComponentFactory.create_model(self._conf)
        self._optimizer = component_factory.ComponentFactory.create_optimizer(self._conf, self._model)

        # create a helper for creating and maintaining checkpoints
        self._ckpt_saver = xb.CheckpointSaver(
            target_dir=self._conf.results_dir,
            filename_pattern="after-{steps}.ckpt"
        )

    def _run_training(self) -> None:

        data_loader = data.DataLoader(
            self._dataset,
            batch_size=self._conf.batch_size,
            collate_fn=lambda x: functools.reduce(operator.add, x) if (
                        self._conf.pretrain or self._conf.ontonotes or self._conf.figer) else lambda x: zip(*x),
            shuffle=True,
            drop_last=True
        )
        print("Num of trainable params: {}".format(sum(p.numel() for p in self._model.parameters() if p.requires_grad)))
        print("Size of dataset: {}".format(len(self._dataset)))
        print("Num of iterations per epoch: {}".format(len(data_loader)))
        print()

        print("Starting model training...")
        epoch_durations = []
        num_steps = 0
        total_loss = 0

        loss_function = None
        if self._conf.ontonotes or self._conf.figer:
            loss_function = nn.BCEWithLogitsLoss(reduction="none")

        for epoch in range(self._conf.num_epochs):
            with util.Timer("finished epoch", terminal_break=True) as epoch_timer:
                # Print epoch header.
                util.printing_tools.print_header("Epoch {}".format(epoch), level=0)
                # Move model to gpu if training on a gpu is specified
                if self._conf.gpu:
                    self._model.cuda()

                self._optimizer.zero_grad()
                iteration_durations = []
                iteration_losses = []
                for iteration_idx, batch in enumerate(data_loader):
                    # Check whether to print details or not.
                    print_details = (iteration_idx + 1) % self._conf.print_int == 0
                    with util.Timer(
                            "finished iteration",
                            skip_output=not print_details,
                            terminal_break=True
                    ) as iteration_timer:
                        if print_details:
                            util.printing_tools.print_header("Iteration {}".format(iteration_idx), level=1)

                        if self._conf.ontonotes or self._conf.figer:
                            # Prepare batch for computation.
                            input_seq, mlm_labels, entity_mask, entity_labels = component_factory.ComponentFactory.prepare_ontonotes_batch(
                                batch,
                                self._dataset.entity_labels,
                                self._model.tokenizer,
                                self._conf.full_span_mask
                            )

                            # Move tensors to GPU if training is on GPU
                            if self._conf.gpu:
                                input_seq = input_seq.cuda()
                                mlm_labels = mlm_labels.cuda()
                                entity_mask = entity_mask.cuda()
                                entity_labels = entity_labels.cuda()

                            # Forward pass
                            model_pred, mlm_loss = self._model(input_seq, entity_mask, mlm_labels)

                            # Compute loss
                            logit_loss = loss_function(model_pred, entity_labels.float()).mean()
                            if self._conf.freeze_ent_enc:
                                total_loss = logit_loss
                            else:
                                total_loss = mlm_loss + logit_loss

                            if print_details:
                                print("BCE loss: {:.4f}".format(logit_loss))
                                print("Entity MLM loss: {:.4f}".format(mlm_loss))
                                print()
                                print("Avg. loss: {:.4f}".format(total_loss))
                                print("Ok")

                            total_loss.backward()
                            iteration_losses.append(total_loss.item())
                            if (iteration_idx + 1) % self._conf.grad_acc_iters == 0:  # -> all needed grads accumulated

                                if print_details:
                                    print("updating model parameters...")

                                utils.clip_grad_norm_(self._model.parameters(), self._conf.max_grad_norm)
                                self._optimizer.step()
                                self._optimizer.zero_grad()

                                if print_details:
                                    print("OK")

                        elif self._conf.pretrain:

                            # Generate contrastive learning labels.
                            input_seq, input_masks, labels = component_factory.ComponentFactory.generate_cl_labels(
                                batch,
                                self._model.tokenizer.pad_token_id
                            )
                            input_seq = torch.LongTensor(input_seq)
                            input_masks = torch.LongTensor(input_masks)
                            cl_labels = torch.LongTensor(list(itertools.chain(*labels)))

                            # Generate MLM labels
                            input_seq, input_labels = component_factory.ComponentFactory.mask_sequences(
                                input_seq,
                                self._model.tokenizer.mask_token_id,
                                input_masks,
                                self._conf.mlm_percentage
                            )
                            # Reduce tensor dimensions if they exceed 512
                            _, dim_size = input_seq.shape
                            if dim_size > 512:
                                input_seq = input_seq[:, :512]
                                input_masks = input_masks[:, :512]
                                input_labels = input_labels[:, :512]

                            try:
                                # Move tensors to GPU if GPU is specified
                                if self._conf.gpu:
                                    cl_labels = cl_labels.cuda()
                                    input_labels = input_labels.cuda()
                                    input_masks = input_masks.cuda()
                                    input_seq = input_seq.cuda()

                                # Compute losses
                                mlm_loss, cls_loss = self._model(input_seq, cl_labels, input_masks, input_labels)
                            except Exception as e:
                                continue

                            if cls_loss != 0:
                                total_loss = cls_loss + mlm_loss
                                iteration_losses.append(total_loss.item())
                                if print_details:
                                    print("Contrastive loss: {:.4f}".format(cls_loss))
                                    print("MLM loss: {:.4f}".format(mlm_loss))
                                    print()
                                    print("Total loss: {:.4f}".format(total_loss))
                                    print("Ok")

                        else:
                            input_seq, target_ent_labels, entity_mask = batch

                            # Pad sequences.
                            input_seq = component_factory.ComponentFactory.pad_sequences(
                                input_seq,
                                self._dataset.tokenizer.pad_token_id
                            )
                            entity_mask = component_factory.ComponentFactory.pad_sequences(
                                entity_mask,
                                0
                            )
                            target_ent_labels = component_factory.ComponentFactory.pad_sequences(
                                target_ent_labels,
                                self._dataset.entity_types.index("O")
                            )
                            # TODO!!! target_ent_labels
                            entity_mask = torch.LongTensor(entity_mask)
                            target_ent_labels = torch.LongTensor(target_ent_labels)
                            # Generate MLM labels.
                            input_seq = torch.LongTensor(input_seq)

                            input_seq, input_labels = component_factory.ComponentFactory.mask_sequences(
                                input_seq,
                                self._dataset.tokenizer.mask_token_id,
                                entity_mask,
                                self._conf.mlm_percentage
                            )

                            # Call model
                            if self._conf.gpu:
                                input_seq = input_seq.cuda()
                                input_labels = input_labels.cuda()
                                entity_mask = entity_mask.cuda()
                                target_ent_labels = target_ent_labels.cuda()

                            predictions, mlm_loss = self._model(input_seq, entity_mask, input_labels)
                            cls_loss = 0
                            entity_mlm_loss = 0
                            cls_loss = self._cross_entropy_loss(
                                predictions.view(-1, predictions.shape[2]),
                                target_ent_labels.view(-1)
                            )

                            if mlm_loss != 0 and not self._conf.freeze_ent_enc:
                                entity_mlm_loss = mlm_loss

                            total_loss = cls_loss + entity_mlm_loss
                            iteration_losses.append(total_loss.item())

                            if print_details:
                                print("Classification loss: {:.4f}".format(cls_loss))
                                print("Entity MLM loss: {:.4f}".format(entity_mlm_loss))
                                print()
                                print("Avg. loss: {:.4f}".format(total_loss))
                                print("Ok")
                        try:
                            total_loss.backward(retain_graph=True)
                            if (iteration_idx + 1) % self._conf.grad_acc_iters == 0:  # -> all needed grads accumulated

                                if print_details:
                                    print("updating model parameters...")

                                utils.clip_grad_norm_(self._model.parameters(), self._conf.max_grad_norm)
                                self._optimizer.step()
                                self._optimizer.zero_grad()

                                if print_details:
                                    print("OK")
                        except Exception as e:
                            pass
                    if print_details:
                        print()
                        util.printing_tools.print_end("Iteration {}".format(iteration_idx), level=1)
                    # End of iteration.
                # Store iteration duration and update step counter
                # iteration_durations.append(iteration_timer.total)
                num_steps += 1
            # Print additional epoch details
            # print("# of iterations: {}".format(len(iteration_durations)))
            # print("Avg. duration per iteration: {:.3f}s".format(np.mean(iteration_durations)))
            print("Avg. epoch loss: {:.4f}".format(np.mean(iteration_losses)))
            print()
            util.printing_tools.print_end("Epoch {}".format(epoch), level=0)
            # End of epoch

            # Store epoch duration
            epoch_durations.append(epoch_timer.total)
            # Create checkpoint
            print("Creating checkpoint...")
            ckpt = None
            self._model.to("cpu")
            if self._conf.pretrain:
                ckpt = entity_encoder_checkpoint.EntityEncoderCheckpoint(
                    epoch,
                    round(float(np.mean(iteration_losses)), 4),
                    self._model.state_dict(),
                    self._optimizer.state_dict(),
                    model_encoder_state=self._model.encoder.state_dict()
                )
            else:
                ckpt = encore_model_checkpoint.EnCoreModelCheckpoint(
                    epoch,
                    self._model.state_dict(),
                    self._optimizer.state_dict()
                )
            ckpt_path = self._ckpt_saver.save(ckpt, steps=epoch)
            self._deliver_ckpt(ckpt_path)
            print("OK")
            print()

        print("Avg. duration per epoch: {:3f}s".format(np.mean(epoch_durations)))
