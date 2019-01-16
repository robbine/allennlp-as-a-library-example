#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F



class IntentSlotModelDecoder(torch.nn.Module):
    """
    `IntentSlotModelDecoder` implements the decoder layer for intent-slot models.
    Intent-slot models jointly predict intent and slots from an utterance.
    At the core these models learn to jointly perform document classification
    and word tagging tasks.

    `IntentSlotModelDecoder` accepts arguments for decoding both document
     classification and word tagging tasks, namely, `doc_input_size` and `word_input_size`.

    Args:
        config (type): Configuration object of type IntentSlotModelDecoder.Config.
        doc_input_size (type): Dimension of input Tensor for projecting document
        representation.
        word_input_size (type): Dimension of input Tensor for projecting word
        representation.
        doc_output_size (type): Dimension of projected output Tensor for document
        classification.
        word_output_size (type): Dimension of projected output Tensor for word tagging.

    Attributes:
        use_doc_probs_in_word (bool): Whether to use intent probabilities for
        predicting slots.
        doc_decoder (type): Document/intent decoder module.
        word_decoder (type): Word/slot decoder module.

    """

    def __init__(
        self,
        doc_input_size: int,
        word_input_size: int,
        doc_output_size: int,
        word_output_size: int,
        use_doc_probs_in_word: bool = False,
    ) -> None:
        super().__init__(config)

        self.use_doc_probs_in_word = use_doc_probs_in_word
        self.doc_decoder = nn.Linear(doc_input_size, doc_output_size)

        if self.use_doc_probs_in_word:
            word_input_size += doc_output_size

        self.word_decoder = nn.Linear(word_input_size, word_output_size)

    def forward(self, x_d: torch.Tensor, x_w: torch.Tensor) -> List[torch.Tensor]:
        logit_d = self.doc_decoder(x_d)
        if self.use_doc_probs_in_word:
            # Get doc probability distribution
            doc_prob = F.softmax(logit_d, 1)
            word_input_shape = x_w.size()
            doc_prob = doc_prob.unsqueeze(1).repeat(1, word_input_shape[1], 1)
            x_w = torch.cat((x_w, doc_prob), 2)

        return [logit_d, self.word_decoder(x_w)]
