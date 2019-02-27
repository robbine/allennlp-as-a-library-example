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

SlotInstance = collections.namedtuple("SlotInstance",
                                      ["start_index", "end_index", "slot_id"])


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
    if raw_slots == '':
        return []
    slots = raw_slots.split(',')
    res = []
    for slot in slots:
        parts = slot.split(':')
        start = int(parts[0])
        end = int(parts[1])
        slot_id = parts[2]
        res.append(
            SlotInstance(start_index=start, end_index=end, slot_id=slot_id))
    return res


@DatasetReader.register("intent_slot_reader")
class IntentSlotDatasetReader(DatasetReader):
    def __init__(self,
                 max_seq_length: int,
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
        self._token_indexers = token_indexers or {
            "tokens": SingleIdTokenIndexer()
        }
        self.rng = random.Random(time.time())

    def _read(self, input_file: str):
        all_records = []
        with open(cached_path(input_file), "r") as reader:
            for line in reader:
                line = convert_to_unicode(line)
                if not line:
                    break
                line = line.strip('\n')
                parts = line.split('\t')
                if len(parts) != 3:
                    print(parts)
                    logger.info('wrong text format '.format(line))
                    continue
                all_records.append(parts)
        self.rng.shuffle(all_records)
        instances = []
        if self.lazy:
            for record in all_records:
                label = record[0]
                raw_slots = record[1]
                raw_query = record[2]
                tags = parse_slots_string(raw_slots)
                yield self.text_to_instance(raw_query, tags, label)
        else:
            for record in all_records:
                label = record[0]
                raw_slots = record[1]
                raw_query = record[2]
                tags = parse_slots_string(raw_slots)
                instances.append(self.text_to_instance(raw_query, tags, label))
            self.rng.shuffle(instances)
            for instance in instances:
                yield instance

    def text_to_instance(self, raw_query, tags, label) -> Instance:
        tokens = self._tokenizer.tokenize(raw_query)
        sequence = TextField(tokens, self._token_indexers)
        if tags is not None:
            tag_ids = ['O', 'O']
            cur_index = 0
            index = 0
            tags.append(
                SlotInstance(
                    start_index=1000, end_index=1001, slot_id='fake_slot_id'))
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
        fields["metadata"] = MetadataField({"words": [x.text for x in tokens]})
        fields["input_mask"] = ArrayField(array(input_mask))
        fields['tokens'] = sequence
        if label:
            fields['labels'] = LabelField(
                label, label_namespace=self.label_namespace)
        if tags is not None:
            fields['tags'] = SequenceLabelField(tag_ids, sequence,
                                                self.tag_namespace)
            fields['slot_tags'] = SequenceLabelField(tag_ids, sequence,
                                                     "slot_tags")
        return Instance(fields)
