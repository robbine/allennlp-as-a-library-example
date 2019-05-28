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


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def mask_seq(tokens, rng):
    """Masks input sequence and output start and end positions"""
    length = len(tokens)
    assert length > 1
    k = length / 2
    start = rng.randint(0, length - k - 1)
    end = start + k - 1
    return start, end


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


@DatasetReader.register("mass_reader")
class MASSDatasetReader(DatasetReader):
    def __init__(self,
                 max_seq_length: int,
                 dupe_factor: int,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self.max_seq_length = max_seq_length
        self.dupe_factor = dupe_factor
        self.lazy = lazy
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers
        self.rng = random.Random(time.time())

    def _read(self, input_file: str):
        """Create `TrainingInstance`s from raw text."""
        document = []
        dictionary = {}
        with open(cached_path(input_file), "r") as reader:
            for line in reader:
                line = convert_to_unicode(line)
                line = line.strip()
                if not line:
                    break
                tokens = self._tokenizer.tokenize(line)
                if tokens:
                    document.append(tokens)
                    for token in tokens:
                        dictionary[token.text] = dictionary.get(token.text, 0) + 1

        self.rng.shuffle(document)
        instances = []
        if self.lazy:
            for _ in range(self.dupe_factor):
                for instance in self.create_instances_from_document(document, dictionary):
                    yield instance
        else:
            for _ in range(self.dupe_factor):
                instances.extend(self.create_instances_from_document(document), dictionary)
            self.rng.shuffle(instances)
            for instance in instances:
                yield instance

    def text_to_instance(self, tokens, start, end, dictionary=None) -> Instance:
        encoder_tokens = []
        decoder_tokens = ['[MASK]']
        target_tokens = []
        for idx, token in enumerate(tokens):
            if start <= idx <= end:
                masked_token = None
                # 80% of the time, replace with [MASK]
                if self.rng.random() < 0.8:
                    masked_token = Token("[MASK]")
                else:
                    # 10% of the time, keep original
                    if self.rng.random() < 0.5:
                        masked_token = token
                    # 10% of the time, replace with random word
                    else:
                        if dictionary is None:
                            masked_token = token
                        else:
                            masked_token = Token(random.choice(list(dictionary.items()))[0])
                encoder_tokens.append(masked_token)
                target_tokens.append(token)
            else:
                encoder_tokens.append(token)
        decoder_tokens.extend(target_tokens[:-1])
        positions = list(range(start, end + 1))
        fields: Dict[str, Field] = {}
        fields["positions"] = ArrayField(array(positions))
        fields['encoder_tokens'] = TextField(encoder_tokens,
                                             self._token_indexers)
        fields['decoder_tokens'] = TextField(decoder_tokens,
                                             self._token_indexers)
        fields['target_tokens'] = TextField(target_tokens,
                                            self._token_indexers)
        return Instance(fields)

    def create_instances_from_document(
            self, document, dictionary=None) -> Iterator[Instance]:  # type: ignore
        """Creates `TrainingInstance`s for a single document."""

        for tokens in document:
            start, end = mask_seq(tokens, self.rng)
            yield self.text_to_instance(tokens, start, end, dictionary)
