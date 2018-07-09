# -*- coding: utf-8 -*-
from typing import Dict
import json
import logging

from overrides import overrides

import tqdm

from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
import pandas as pd
import random
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("biaowen_mingxi")
class BiaoWenMingXiDatasetReader(DatasetReader):
    """
    Reads a JSON-lines file containing papers from the Semantic Scholar database, and creates a
    dataset suitable for document classification using these papers.

    Expected format for each input line: {"paperAbstract": "text", "title": "text", "venue": "text"}

    The JSON could have other fields, too, but they are ignored.

    The output of ``read`` is a list of ``Instance`` s with the fields:
        title: ``TextField``
        abstract: ``TextField``
        label: ``LabelField``

    where the ``label`` is derived from the venue of the paper.

    Parameters
    ----------
    lazy : ``bool`` (optional, default=False)
        Passed to ``DatasetReader``.  If this is ``True``, training will start sooner, but will
        take longer per batch.  This also allows training with datasets that are too large to fit
        in memory.
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the title and abstrct into words or other kinds of tokens.
        Defaults to ``WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input token representations. Defaults to ``{"tokens":
        SingleIdTokenIndexer()}``.
    """
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 sheet_name: str = 'Sheet1') -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._sheet_name = sheet_name

    @overrides
    def _read(self, file_path):
        df = pd.read_excel(cached_path(file_path), sheet_name=self._sheet_name)
        std_id_2_question = dict(zip(df['标准问题ID'], df['标准问题标题']))

        logger.info("Reading instances from lines in file at: %s", file_path)
        for index, row in df.iterrows():
            if row.empty:
                continue
            std_id = row['标准问题ID']
            for key in random.sample(std_id_2_question.keys(), 2):
                if key != std_id:
                    neg_doc = std_id_2_question[key]
                else:
                    continue
            doc = row['标准问题标题']
            query = row['标准问题问句标题']
            yield self.text_to_instance(doc, query, 'positive')
            yield self.text_to_instance(neg_doc, query, 'negative')

    @overrides
    def text_to_instance(self, doc: str, query: str, label: str) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokenized_doc = self._tokenizer.tokenize(doc)
        tokenized_query = self._tokenizer.tokenize(query)
        doc_field = TextField(tokenized_doc, self._token_indexers)
        query_field = TextField(tokenized_query, self._token_indexers)
        fields = {'premise': doc_field, 'hypothesis': query_field, 'label': LabelField(label)}
        return Instance(fields)

    @classmethod
    def from_params(cls, params: Params) -> 'BiaoWenMingXiDatasetReader':
        lazy = params.pop('lazy', False)
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        sheet_name = params.pop('sheet_name', 'Sheet1')
        params.assert_empty(cls.__name__)
        return cls(lazy=lazy, tokenizer=tokenizer, token_indexers=token_indexers, sheet_name=sheet_name)
