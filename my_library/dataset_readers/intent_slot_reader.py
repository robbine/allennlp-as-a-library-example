# -*- coding: utf-8 -*-
from typing import Dict, Iterable, List
import logging
import random
import time
import csv
import collections
from typing import Iterator

from allennlp.common import Params, Tqdm
from allennlp.common.file_utils import cached_path
from allennlp.data import Field, Token
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, ArrayField, LabelField, SequenceLabelField, Field, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.token_indexers.single_id_token_indexer import SingleIdTokenIndexer
from numpy import array
import six

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

SlotInstance = collections.namedtuple("SlotInstance", ["start_index", "end_index", "slot_id"])

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

def parse_slots_string(raw_slots):
	slots = raw_slots.split(',')
	res = []
	for slot in slots:
		parts = slot.split(':')
		# always add offset 1 since due to BOS symbol
		start = int(parts[0])
		end = int(parts[1])
		slot_id = parts[2]
		res.append(SlotInstance(start_index=start, end_index=end, slot_id=slot_id))
	return res

@DatasetReader.register("intent_slot_reader")
class IntentSlotDatasetReader(DatasetReader):
	def __init__(self, max_seq_length: int,
				 lazy: bool = False,
				 coding_scheme: str = "IOB1",
				 label_namespace: str = "labels",
				 tag_namespace: str = "tags",
				 tokenizer: Tokenizer = None,
				 token_indexers: Dict[str, TokenIndexer] = None) -> None:
		super().__init__(lazy)
		self.max_seq_length = max_seq_length
		self.lazy = lazy
		self.coding_scheme = coding_scheme
		self.label_namespace = label_namespace
		self.tag_namespace = tag_namespace
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
				tags = parse_slots_string(raw_slots)
				yield self.text_to_instance(tokens, tags, label)


	def text_to_instance(self, tokens, tags, label) -> Instance:
		sequence = TextField(tokens, self._token_indexers)
		tag_ids = ['O', 'O']
		index = 0
		cur_index = 0
		tags.append(SlotInstance(start_index=1000, end_index=1001, slot_id='fake_slot_id'))
		slot = tags[index]
		# skip leading [CLS] BOS token
		i = 2
		while i < len(tokens):
			token = tokens[i]
			if cur_index + len(token.text) <= slot.start_index:
				tag_ids.append('O')
				cur_index = cur_index + len(token.text)
			elif cur_index >= slot.end_index:
				index = index + 1
				slot = tags[index]
				i = i - 1
			else:
				if len(tag_ids) == 0 or tag_ids[-1] == 'O':
					tag_ids.append('B-' + slot.slot_id)
				else:
					tag_ids.append('I-' + slot.slot_id)
				cur_index = cur_index + len(token.text)
			i = i + 1
		input_mask = [1] * len(tokens)
		fields: Dict[str, Field] = {}
		fields['slot_tags'] = SequenceLabelField(tag_ids, sequence, "slot_tags")
		fields["metadata"] = MetadataField({"words": [x.text for x in tokens]})
		fields["input_mask"] = ArrayField(array(input_mask))
		fields['tokens'] = sequence
		fields['labels'] = LabelField(label, label_namespace=self.label_namespace)
		fields['tags'] = SequenceLabelField(tag_ids, sequence, self.tag_namespace)
		return Instance(fields)
