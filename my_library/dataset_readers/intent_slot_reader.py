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
from allennlp.data.token_indexers.single_id_token_indexer import SingleIdTokenIndexer
from numpy import array
import six

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

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

@DatasetReader.register("intent_slot_reader")
class IntentSlotDatasetReader(DatasetReader):
	def __init__(self, max_seq_length: int,
				 lazy: bool = False,
				 coding_scheme: str = "IOB1",
				 label_namespace: str = "labels",
				 tokenizer: Tokenizer = None,
				 token_indexers: Dict[str, TokenIndexer] = None) -> None:
		super().__init__(lazy)
		self.max_seq_length = max_seq_length
		self.lazy = lazy
		self.coding_scheme = coding_scheme
		self.label_namespace = label_namespace
		self._original_coding_scheme = "IOB1"
		self._tokenizer = tokenizer or WordTokenizer()
		self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
		self.rng = random.Random(time.time())

	def _read(self, input_file: str):
		with open(cached_path(input_file), "r") as reader:
			for line in reader:
				line = convert_to_unicode(line)
				if not line:
					break
				line = line.strip()
				parts = line.split('\t')
				if len(parts) != 3:
					logger.info('wrong text format '.format(line))
					continue
				label = parts[0]
				raw_slots = parts[1]
				raw_query = parts[2]
				tokens = self._tokenizer.tokenize(raw_query)
				yield self.text_to_instance(tokens, tags, label)


	def text_to_instance(self, tokens, tags, label) -> Instance:
		tokens = []
		segment_ids = []
		tokens.append(Token("[BEGIN]"))
		segment_ids.append(0)
		for token in tokens_a:
			tokens.append(token)
			segment_ids.append(0)

		tokens.append(Token("[END]"))
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
		fields['labels'] = LabelField(label, label_namespace='labels')
		fields['masked_lm_labels'] = TextField(masked_lm_labels, self._token_indexers)
		return Instance(fields)
