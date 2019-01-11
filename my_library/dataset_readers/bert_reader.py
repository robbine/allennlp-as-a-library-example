# -*- coding: utf-8 -*-
from typing import Dict, Iterable, List
import logging
import random
import time
import collections
from typing import Iterator

from allennlp.common import Params, Tqdm
from allennlp.common.file_utils import cached_path
from allennlp.data import Field, Token
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, ArrayField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer
from numpy import array
import six

from my_library.token_indexers import BertSingleIdTokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
										  ["index", "label"])


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
	"""Truncates a pair of sequences to a maximum sequence length."""
	while True:
		total_length = len(tokens_a) + len(tokens_b)
		if total_length <= max_num_tokens:
			break

		trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
		assert len(trunc_tokens) >= 1

		# We want to sometimes truncate from the front and sometimes from the
		# back to add more randomness and avoid biases.
		if rng.random() < 0.5:
			del trunc_tokens[0]
		else:
			trunc_tokens.pop()


def convert_to_unicode(text):
	"""Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
	if six.PY3:
		if isinstance(text, str):
			return text
		elif isinstance(text, bytes):
			return text.decode("utf-8", "ignore")
		else:
			raise ValueError("Unsupported string type: %s" % (type(text)))
	elif six.PY2:
		if isinstance(text, str):
			return text.decode("utf-8", "ignore")
		elif isinstance(text, unicode):
			return text
		else:
			raise ValueError("Unsupported string type: %s" % (type(text)))
	else:
		raise ValueError("Not running on Python2 or Python 3?")


@DatasetReader.register("bert_reader")
class BertDatasetReader(DatasetReader):
	def __init__(self, max_seq_length: int, dupe_factor: int, short_seq_prob: float, masked_lm_prob: float,
				 max_predictions_per_seq: int, lazy: bool = False, tokenizer: Tokenizer = None,
				 token_indexers: Dict[str, TokenIndexer] = None) -> None:
		super().__init__(lazy)
		self.max_seq_length = max_seq_length
		self.dupe_factor = dupe_factor
		self.short_seq_prob = short_seq_prob
		self.masked_lm_prob = masked_lm_prob
		self.max_predictions_per_seq = max_predictions_per_seq
		self.lazy = lazy
		self._tokenizer = tokenizer or WordTokenizer()
		self._token_indexers = token_indexers or {"tokens": BertSingleIdTokenIndexer()}
		self.rng = random.Random(time.time())

	def _read(self, input_file: str):
		"""Create `TrainingInstance`s from raw text."""
		all_documents = [[]]
		dictionary = {}
		# Input file format:
		# (1) One sentence per line. These should ideally be actual sentences, not
		# entire paragraphs or arbitrary spans of text. (Because we use the
		# sentence boundaries for the "next sentence prediction" task).
		# (2) Blank lines between documents. Document boundaries are needed so
		# that the "next sentence prediction" task doesn't span between documents.
		with open(cached_path(input_file), "r") as reader:
			for line in reader:
				line = convert_to_unicode(line)
				if not line:
					break
				line = line.strip()
				# Empty lines are used as document delimiters
				if not line:
					all_documents.append([])
				tokens = self._tokenizer.tokenize(line)
				if tokens:
					all_documents[-1].append(tokens)
					for token in tokens:
						dictionary[token.text] = dictionary.get(token.text, 0) + 1

		all_documents = [x for x in all_documents if x]
		self.rng.shuffle(all_documents)
		instances = []
		if self.lazy:
			for _ in range(self.dupe_factor):
				for document_index in range(len(all_documents)):
					for instance in self.create_instances_from_document_dummy(all_documents, document_index, dictionary):
						yield instance
		else:
			for _ in range(self.dupe_factor):
				for document_index in range(len(all_documents)):
					instances.extend(
						self.create_instances_from_document(
							all_documents, document_index, dictionary))

			self.rng.shuffle(instances)
			for instance in instances:
				yield instance

	def text_to_instance(self, tokens_a, tokens_b, is_random_next=False, dictionary=None) -> Instance:
		assert len(tokens_a) >= 1
		assert len(tokens_b) >= 1

		tokens = []
		segment_ids = []
		tokens.append(Token("[CLS]"))
		segment_ids.append(0)
		for token in tokens_a:
			tokens.append(token)
			segment_ids.append(0)

		tokens.append(Token("[SEP]"))
		segment_ids.append(0)

		for token in tokens_b:
			tokens.append(token)
			segment_ids.append(1)
		tokens.append(Token("[SEP]"))
		segment_ids.append(1)
		input_mask = [1] * len(tokens)
		(tokens, masked_lm_positions,
		 masked_lm_labels) = self.create_masked_lm_predictions(
			tokens, dictionary)
		masked_lm_weights = [1.0] * len(masked_lm_positions)
		fields: Dict[str, Field] = {}
		fields["input_mask"] = ArrayField(array(input_mask))
		fields['tokens'] = TextField(tokens, self._token_indexers)
		fields['segment_ids'] = ArrayField(array(segment_ids))
		fields['next_sentence_labels'] = LabelField(0 if is_random_next else 1, skip_indexing=True)
		fields['masked_lm_positions'] = ArrayField(array(masked_lm_positions))
		fields['masked_lm_weights'] = ArrayField(array(masked_lm_weights))
		fields['masked_lm_labels'] = TextField(masked_lm_labels, self._token_indexers)
		return Instance(fields)

	def create_instances_from_document_dummy(self, all_documents, document_index,
									   dictionary=None) -> Iterator[Instance]:  # type: ignore
		"""Creates `TrainingInstance`s for a single document."""
		document = all_documents[document_index]

		# Account for [CLS], [SEP], [SEP]
		max_num_tokens = self.max_seq_length - 3

		i = 0
		while i < len(document):
			if i + 1 < len(document):
				tokens_a = document[i]
				# Random next
				is_random_next = False
				if self.rng.random() < 0.5:
					is_random_next = True

					# This should rarely go for more than one iteration for large
					# corpora. However, just to be careful, we try to make sure that
					# the random document is not the same as the document
					# we're processing.
					random_document_index = 0
					for _ in range(10):
						random_document_index = self.rng.randint(0, len(all_documents) - 1)
						if random_document_index != document_index:
							break

					random_document = all_documents[random_document_index]
					tokens_b = random.choice(random_document)
				# Actual next
				else:
					is_random_next = False
					tokens_b = document[i+1]
				truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, self.rng)
				yield self.text_to_instance(tokens_a, tokens_b, is_random_next, dictionary)
			i += 2
		return

	def create_instances_from_document(self, all_documents, document_index,
									   dictionary=None) -> List[Instance]:  # type: ignore
		"""Creates `TrainingInstance`s for a single document."""
		document = all_documents[document_index]

		# Account for [CLS], [SEP], [SEP]
		max_num_tokens = self.max_seq_length - 3

		# We *usually* want to fill up the entire sequence since we are padding
		# to `max_seq_length` anyways, so short sequences are generally wasted
		# computation. However, we *sometimes*
		# (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
		# sequences to minimize the mismatch between pre-training and fine-tuning.
		# The `target_seq_length` is just a rough target however, whereas
		# `max_seq_length` is a hard limit.
		target_seq_length = max_num_tokens
		if self.rng.random() < self.short_seq_prob:
			target_seq_length = self.rng.randint(2, max_num_tokens)

		# We DON'T just concatenate all of the tokens from a document into a long
		# sequence and choose an arbitrary split point because this would make the
		# next sentence prediction task too easy. Instead, we split the input into
		# segments "A" and "B" based on the actual "sentences" provided by the user
		# input.
		instances = []
		current_chunk = []
		current_length = 0
		i = 0
		while i < len(document):
			segment = document[i]
			current_chunk.append(segment)
			current_length += len(segment)
			if i == len(document) - 1 or current_length >= target_seq_length:
				if current_chunk:
					# `a_end` is how many segments from `current_chunk` go into the `A`
					# (first) sentence.
					a_end = 1
					if len(current_chunk) >= 2:
						a_end = self.rng.randint(1, len(current_chunk) - 1)

					tokens_a = []
					for j in range(a_end):
						tokens_a.extend(current_chunk[j])

					tokens_b = []
					# Random next
					is_random_next = False
					if len(current_chunk) == 1 or self.rng.random() < 0.5:
						is_random_next = True
						target_b_length = target_seq_length - len(tokens_a)

						# This should rarely go for more than one iteration for large
						# corpora. However, just to be careful, we try to make sure that
						# the random document is not the same as the document
						# we're processing.
						random_document_index = 0
						for _ in range(10):
							random_document_index = self.rng.randint(0, len(all_documents) - 1)
							if random_document_index != document_index:
								break

						random_document = all_documents[random_document_index]
						random_start = self.rng.randint(0, len(random_document) - 1)
						for j in range(random_start, len(random_document)):
							tokens_b.extend(random_document[j])
							if len(tokens_b) >= target_b_length:
								break
						# We didn't actually use these segments so we "put them back" so
						# they don't go to waste.
						num_unused_segments = len(current_chunk) - a_end
						i -= num_unused_segments
					# Actual next
					else:
						is_random_next = False
						for j in range(a_end, len(current_chunk)):
							tokens_b.extend(current_chunk[j])
						# logger.info('tokens a {}'.format(tokens_a))
						# logger.info('tokens b {}'.format(tokens_b))
					truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, self.rng)
					instances.append(self.text_to_instance(tokens_a, tokens_b, is_random_next, dictionary))
				current_chunk = []
				current_length = 0
			i += 1

		return instances

	def create_masked_lm_predictions(self, tokens, dictionary=None):
		"""Creates the predictions for the masked LM objective.
		tokens: List[str]
		"""

		cand_indexes = []
		for (i, token) in enumerate(tokens):
			if token.text == "[CLS]" or token.text == "[SEP]":
				continue
			cand_indexes.append(i)

		self.rng.shuffle(cand_indexes)

		output_tokens = list(tokens)

		num_to_predict = min(self.max_predictions_per_seq,
							 max(1, int(round(len(tokens) * self.masked_lm_prob))))

		masked_lms = []
		covered_indexes = set()
		changed = False
		for index in cand_indexes:
			if len(masked_lms) >= num_to_predict:
				break
			if index in covered_indexes:
				continue
			covered_indexes.add(index)

			masked_token = None
			# 80% of the time, replace with [MASK]
			if self.rng.random() < 0.8:
				masked_token = Token("[MASK]")
			else:
				# 10% of the time, keep original
				if self.rng.random() < 0.5:
					masked_token = tokens[index]
				# 10% of the time, replace with random word
				else:
					if dictionary is None:
						masked_token = tokens[index]
					else:
						changed = True
						masked_token = Token(random.choice(list(dictionary.items()))[0])

			output_tokens[index] = masked_token

			masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

		masked_lms = sorted(masked_lms, key=lambda x: x.index)
		# if changed:
		# 	logger.info('original tokens {}'.format(tokens))
		# 	logger.info('output tokens {}'.format(output_tokens))
		masked_lm_positions = []
		masked_lm_labels = []
		for p in masked_lms:
			masked_lm_positions.append(p.index)
			masked_lm_labels.append(p.label)

		return (output_tokens, masked_lm_positions, masked_lm_labels)
