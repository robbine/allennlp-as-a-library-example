from typing import Dict, List, Tuple
from typing import Optional
import numpy
from overrides import overrides
import torch
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTMCell

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.attention import LegacyAttention
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.modules.similarity_functions import SimilarityFunction
from allennlp.models.model import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import util
from allennlp.nn.beam_search import BeamSearch
from allennlp.training.metrics import BLEU
from allennlp.nn import InitializerApplicator, RegularizerApplicator

from my_library.modules.decoders.transformer_decoder import TransformerDecoder


@Model.register("mass")
class MASS(Model):
    """
    This ``MASS`` class is a :class:`Model` which takes a sequence, encodes it, and then
    uses the encoded representations to decode another sequence.  You can use this as the basis for
    a neural machine translation system, an abstractive summarization system, or any other common
    seq2seq problem.  The model here is simple, but should be a decent starting place for
    implementing recent models for these tasks.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        Vocabulary containing source and target vocabularies. They may be under the same namespace
        (`tokens`) or the target tokens can have a different namespace, in which case it needs to
        be specified as `target_namespace`.
    source_embedder : ``TextFieldEmbedder``, required
        Embedder for source side sequences
    encoder : ``Seq2SeqEncoder``, required
        The encoder of the "encoder/decoder" model
    max_decoding_steps : ``int``
        Maximum length of decoded sequences.
    target_namespace : ``str``, optional (default = 'target_tokens')
        If the target side vocabulary is different from the source side's, you need to specify the
        target's namespace here. If not, we'll assume it is "tokens", which is also the default
        choice for the source side, and this might cause them to share vocabularies.
    target_embedding_dim : ``int``, optional (default = source_embedding_dim)
        You can specify an embedding dimensionality for the target side. If not, we'll use the same
        value as the source embedder's.
    attention : ``Attention``, optional (default = None)
        If you want to use attention to get a dynamic summary of the encoder outputs at each step
        of decoding, this is the function used to compute similarity between the decoder hidden
        state and encoder outputs.
    attention_function: ``SimilarityFunction``, optional (default = None)
        This is if you want to use the legacy implementation of attention. This will be deprecated
        since it consumes more memory than the specialized attention modules.
    beam_size : ``int``, optional (default = None)
        Width of the beam for beam search. If not specified, greedy decoding is used.
    scheduled_sampling_ratio : ``float``, optional (default = 0.)
        At each timestep during training, we sample a random number between 0 and 1, and if it is
        not less than this value, we use the ground truth labels for the whole batch. Else, we use
        the predictions from the previous time step for the whole batch. If this value is 0.0
        (default), this corresponds to teacher forcing, and if it is 1.0, it corresponds to not
        using target side ground truth labels.  See the following paper for more information:
        `Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks. Bengio et al.,
        2015 <https://arxiv.org/abs/1506.03099>`_.
    use_bleu : ``bool``, optional (default = True)
        If True, the BLEU metric will be calculated during validation.
    """

    def __init__(self,
                 vocab: Vocabulary,
                 source_embedder: TextFieldEmbedder,
                 transformer_encoder: Seq2SeqEncoder,
                 max_decoding_steps: int,
                 beam_size: int = None,
                 target_namespace: str = "tokens",
                 target_embedding_dim: int = None,
                 use_bleu: bool = True,
                 use_fp16: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(MASS, self).__init__(vocab)
        self._target_namespace = target_namespace
        # We need the start symbol to provide as the input at the first timestep of decoding, and
        # end symbol as a way to indicate the end of the decoded sequence.
        self._start_index = self.vocab.get_token_index(START_SYMBOL,
                                                       self._target_namespace)
        self._end_index = self.vocab.get_token_index(END_SYMBOL,
                                                     self._target_namespace)
        self._mask_index = self.vocab.get_token_index('[MASK]',
                                                      self._target_namespace)

        if use_bleu:
            pad_index = self.vocab.get_token_index(self.vocab._padding_token,
                                                   self._target_namespace)  # pylint: disable=protected-access
            self._bleu = BLEU(exclude_indices={
                pad_index, self._end_index, self._start_index, self._mask_index
            })
        else:
            self._bleu = None

        # At prediction time, we use a beam search to find the most likely sequence of target tokens.
        beam_size = beam_size or 1
        self._max_decoding_steps = max_decoding_steps
        self._beam_search = BeamSearch(
            self._end_index, max_steps=max_decoding_steps, beam_size=beam_size)

        # Dense embedding of source vocab tokens.
        self._source_embedder = source_embedder

        # Encodes the sequence of source embeddings into a sequence of hidden states.
        self._encoder = transformer_encoder

        # Dense embedding of vocab words in the target space.
        target_embedding_dim = target_embedding_dim or source_embedder.get_output_dim(
        )
        self._target_embedder = source_embedder

        # Decoder output dim needs to be the same as the encoder output dim since we initialize the
        # hidden state of the decoder with the final hidden state of the encoder.
        self._encoder_output_dim = self._encoder.get_output_dim()
        self._decoder_output_dim = self._encoder_output_dim

        self._decoder_input_dim = target_embedding_dim

        self._decoder = TransformerDecoder(
            use_fp16,
            self._target_embedder,
            decoder_layers=6,
            dropout=0.1,
            decoder_embed_dim=self._encoder_output_dim,
            decoder_ffn_embed_dim=target_embedding_dim,
            decoder_attention_heads=4,
            decoder_output_dim=self._decoder_output_dim,
            max_target_positions=512,
            attention_dropout=0.1,
        )
        initializer(self)

    def take_step(self, last_predictions: torch.Tensor,
                  state: Dict[str, torch.Tensor]
                  ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Take a decoding step. This is called by the beam search class.

        Parameters
        ----------
        last_predictions : ``torch.Tensor``
            A tensor of shape ``(group_size,)``, which gives the indices of the predictions
            during the last time step.
        state : ``Dict[str, torch.Tensor]``
            A dictionary of tensors that contain the current state information
            needed to predict the next step, which includes the encoder outputs,
            the source mask, and the decoder hidden state and context. Each of these
            tensors has shape ``(group_size, *)``, where ``*`` can be any other number
            of dimensions.

        Returns
        -------
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]
            A tuple of ``(log_probabilities, updated_state)``, where ``log_probabilities``
            is a tensor of shape ``(group_size, num_classes)`` containing the predicted
            log probability of each class for the next step, for each item in the group,
            while ``updated_state`` is a dictionary of tensors containing the encoder outputs,
            source mask, and updated decoder hidden state and context.

        Notes
        -----
            We treat the inputs as a batch, even though ``group_size`` is not necessarily
            equal to ``batch_size``, since the group may contain multiple states
            for each source sentence in the batch.
        """
        start_predictions = {'tokens': last_predictions.unsqueeze(1)}
        # shape: (group_size, num_classes)
        output_projections, state = self._prepare_output_projections(
            start_predictions, state)

        # shape: (group_size, num_classes)
        class_log_probabilities = F.log_softmax(output_projections, dim=-1)
        class_log_probabilities = class_log_probabilities.squeeze(1)

        return class_log_probabilities, state

    @overrides
    def forward(
            self,  # type: ignore
            encoder_tokens: Dict[str, torch.LongTensor],
            decoder_tokens: Dict[str, torch.LongTensor] = None,
            target_tokens: Dict[str, torch.LongTensor] = None,
            positions: torch.LongTensor = None,
    ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Make foward pass with decoder logic for producing the entire target sequence.

        Parameters
        ----------
        encoder_tokens : ``Dict[str, torch.LongTensor]``
           The output of `TextField.as_array()` applied on the source `TextField`. This will be
           passed through a `TextFieldEmbedder` and then through an encoder.
        target_tokens : ``Dict[str, torch.LongTensor]``, optional (default = None)
           Output of `Textfield.as_array()` applied on target `TextField`. We assume that the
           target tokens are also represented as a `TextField`.

        Returns
        -------
        Dict[str, torch.Tensor]
        """
        state = self._encode(encoder_tokens)

        if decoder_tokens:
            # The `_forward_loop` decodes the input sequence and computes the loss during training
            # and validation.
            output_dict = self._forward_loop(
                state,
                decoder_tokens=decoder_tokens,
                target_tokens=target_tokens,
                positions=positions)
        else:
            output_dict = {}

        if not self.training:
            if not decoder_tokens:
                predictions = self._forward_beam_search(state)
                output_dict.update(predictions)
                if target_tokens and self._bleu:
                    # shape: (batch_size, beam_size, max_sequence_length)
                    top_k_predictions = output_dict["predictions"]
                    # shape: (batch_size, max_predicted_sequence_length)
                    best_predictions = top_k_predictions[:, 0, :]
                    self._bleu(best_predictions, target_tokens["tokens"])
            else:
                if target_tokens and self._bleu:
                    best_predictions = output_dict["predictions"]
                    self._bleu(best_predictions, target_tokens["tokens"])

        return output_dict

    def _encode(self, source_tokens: Dict[str, torch.Tensor]
                ) -> Dict[str, torch.Tensor]:
        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
        embedded_input = self._source_embedder(source_tokens)
        # shape: (batch_size, max_input_sequence_length)
        source_mask = util.get_text_field_mask(source_tokens)
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = self._encoder(embedded_input, source_mask)
        return {
            "source_mask": source_mask,
            "encoder_outputs": encoder_outputs,
        }

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]
               ) -> Dict[str, torch.Tensor]:
        """
        Finalize predictions.

        This method overrides ``Model.decode``, which gets called after ``Model.forward``, at test
        time, to finalize predictions. The logic for the decoder part of the encoder-decoder lives
        within the ``forward`` method.

        This method trims the output predictions to the first end symbol, replaces indices with
        corresponding tokens, and adds a field called ``predicted_tokens`` to the ``output_dict``.
        """
        predicted_indices = output_dict["predictions"]
        if not isinstance(predicted_indices, numpy.ndarray):
            predicted_indices = predicted_indices.detach().cpu().numpy()
        all_predicted_tokens = []
        for indices in predicted_indices:
            # Beam search gives us the top k results for each source sentence in the batch
            # but we just want the single best.
            if len(indices.shape) > 1:
                indices = indices[0]
            indices = list(indices)
            # Collect indices till the first end_symbol
            if self._end_index in indices:
                indices = indices[:indices.index(self._end_index)]
            predicted_tokens = [
                self.vocab.get_token_from_index(
                    x, namespace=self._target_namespace) for x in indices
            ]
            all_predicted_tokens.append(predicted_tokens)
        output_dict["predicted_tokens"] = all_predicted_tokens
        return output_dict

    def _forward_loop(
            self,
            state: Dict[str, torch.Tensor],
            decoder_tokens: Dict[str, torch.LongTensor] = None,
            target_tokens: Dict[str, torch.LongTensor] = None,
            positions: torch.LongTensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Make forward pass during training or do greedy search during prediction.

        Notes
        -----
        We really only use the predictions from the method to test that beam search
        with a beam size of 1 gives the same results.
        """
        decoder_token_mask = util.get_text_field_mask(decoder_tokens)
        decoder_padding_mask = (decoder_token_mask == 0)
        # shape: (batch_size, num_classes)
        logits, state = self._prepare_output_projections(
            decoder_tokens,
            state,
            decoder_padding_mask=decoder_padding_mask,
            positions=positions)
        # shape: (batch_size, num_classes)
        class_probabilities = F.softmax(logits, dim=-1)

        # shape (predicted_classes): (batch_size,)
        _, predictions = torch.max(class_probabilities, -1)

        output_dict = {"predictions": predictions}

        if target_tokens is not None:
            # Compute loss.
            target_mask = decoder_token_mask
            targets = target_tokens["tokens"]
            loss = self._get_loss(logits, targets, target_mask)
            output_dict["loss"] = loss

        return output_dict

    def _forward_beam_search(
            self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Make forward pass during prediction using a beam search."""
        batch_size = state["source_mask"].size(0)
        start_tokens = state["source_mask"].new_full(
            (batch_size, ), fill_value=self._mask_index)
        # shape (all_top_k_predictions): (batch_size, beam_size, num_decoding_steps)
        # shape (log_probabilities): (batch_size, beam_size)
        all_top_k_predictions, log_probabilities = self._beam_search.search(
            start_tokens, state, self.take_step)

        output_dict = {
            "class_log_probabilities": log_probabilities,
            "predictions": all_top_k_predictions,
        }
        return output_dict

    def _prepare_output_projections(self,
                                    last_predictions: Dict[str, torch.LongTensor],
                                    state: Dict[str, torch.Tensor],
                                    decoder_padding_mask: torch.LongTensor = None,
                                    positions: torch.LongTensor = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:  # pylint: disable=line-too-long
        """
        Decode current state and last prediction to produce produce projections
        into the target space, which can then be used to get probabilities of
        each target token for the next step.

        Inputs are the same as for `take_step()`.
        """
        # shape: (group_size, max_input_sequence_length, encoder_output_dim)
        encoder_out = state["encoder_outputs"]

        # shape: (group_size, max_input_sequence_length)
        encoder_padding_mask = (state["source_mask"] == 0)

        # shape: (group_size, target_embedding_dim)
        prev_output_tokens = last_predictions
        decoder_output = self._decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            encoder_padding_mask=encoder_padding_mask,
            decoder_padding_mask=decoder_padding_mask,
            positions=positions)

        return decoder_output, state

    @staticmethod
    def _get_loss(logits: torch.LongTensor, targets: torch.LongTensor,
                  target_mask: torch.LongTensor) -> torch.Tensor:
        """
        Compute loss.

        Takes logits (unnormalized outputs from the decoder) of size (batch_size,
        num_decoding_steps, num_classes), target indices of size (batch_size, num_decoding_steps+1)
        and corresponding masks of size (batch_size, num_decoding_steps+1) steps and computes cross
        entropy loss while taking the mask into account.

        The length of ``targets`` is expected to be greater than that of ``logits`` because the
        decoder does not need to compute the output corresponding to the last timestep of
        ``targets``. This method aligns the inputs appropriately to compute the loss.

        During training, we want the logit corresponding to timestep i to be similar to the target
        token from timestep i + 1. That is, the targets should be shifted by one timestep for
        appropriate comparison.  Consider a single example where the target has 3 words, and
        padding is to 7 tokens.
           The complete sequence would correspond to <S> w1  w2  w3  <E> <P> <P>
           and the mask would be                      1   1   1   1   1   0   0
           and let the logits be                     l1  l2  l3  l4  l5  l6
        We actually need to compare:
           the sequence           w1  w2  w3  <E> <P> <P>
           with masks             1   1   1   1   0   0
           against                l1  l2  l3  l4  l5  l6
           (where the input was)  <S> w1  w2  w3  <E> <P>
        """
        return util.sequence_cross_entropy_with_logits(logits, targets,
                                                       target_mask)

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if self._bleu and not self.training:
            all_metrics.update(self._bleu.get_metric(reset=reset))
        return all_metrics
