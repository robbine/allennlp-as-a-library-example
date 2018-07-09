import re
from typing import List

from overrides import overrides
import spacy

from allennlp.common import Params, Registrable
from allennlp.common.util import get_spacy_model
from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers.word_splitter import WordSplitter
import jieba
import jieba.posseg as pseg


@WordSplitter.register('jieba')
class JiebaWordSplitter(WordSplitter):
    """
    Does really simple tokenization.  NLTK was too slow, so we wrote our own simple tokenizer
    instead.  This just does an initial split(), followed by some heuristic filtering of each
    whitespace-delimited token, separating contractions and punctuation.  We assume lower-cased,
    reasonably well-formed English sentences as input.
    """
    def __init__(self, cut_all=False, hmm=False, cut_for_search=False, pos_tags=False) -> None:
        # These are certainly incomplete.  But at least it's a start.
        self.special_cases = set(['mr.', 'mrs.', 'etc.', 'e.g.', 'cf.', 'c.f.', 'eg.', 'al.'])
        self.contractions = set(["n't", "'s", "'ve", "'re", "'ll", "'d", "'m"])
        self.contractions |= set([x.replace("'", "’") for x in self.contractions])
        self.ending_punctuation = set(['"', "'", '.', ',', ';', ')', ']', '}', ':', '!', '?', '%', '”', "’"])
        self.beginning_punctuation = set(['"', "'", '(', '[', '{', '#', '$', '“', "‘"])
        self.cut_all = cut_all
        self.hmm = hmm
        self.cut_for_search=cut_for_search
        self.pos_tags = pos_tags

    @overrides
    def split_words(self, sentence: str) -> List[Token]:
        """
        Splits a sentence into word tokens.  We handle four kinds of things: words with punctuation
        that should be ignored as a special case (Mr. Mrs., etc.), contractions/genitives (isn't,
        don't, Matt's), and beginning and ending punctuation ("antennagate", (parentheticals), and
        such.).

        The basic outline is to split on whitespace, then check each of these cases.  First, we
        strip off beginning punctuation, then strip off ending punctuation, then strip off
        contractions.  When we strip something off the beginning of a word, we can add it to the
        list of tokens immediately.  When we strip it off the end, we have to save it to be added
        to after the word itself has been added.  Before stripping off any part of a token, we
        first check to be sure the token isn't in our list of special cases.
        """
        if self.pos_tags:
            cut_res = pseg.lcut(sentence=sentence, HMM=self.hmm)
            fields = [text for text, _ in cut_res]
            tags = [tag for _, tag in cut_res]
        else:
            if self.cut_for_search:
                fields = jieba.cut_for_search(sentence=sentence, HMM=self.hmm)
            else:
                fields = jieba.cut(sentence, cut_all=self.cut_all, HMM=self.hmm)
        tokens: List[Token] = []
        for idx, field in enumerate(fields):
            add_at_end: List[Token] = []
            while self._can_split(field) and field[0] in self.beginning_punctuation:
                tokens.append(Token(field[0]))
                field = field[1:]
            while self._can_split(field) and field[-1] in self.ending_punctuation:
                add_at_end.insert(0, Token(field[-1]))
                field = field[:-1]

            # There could (rarely) be several contractions in a word, but we check contractions
            # sequentially, in a random order.  If we've removed one, we need to check again to be
            # sure there aren't others.
            remove_contractions = True
            while remove_contractions:
                remove_contractions = False
                for contraction in self.contractions:
                    if self._can_split(field) and field.lower().endswith(contraction):
                        add_at_end.insert(0, Token(field[-len(contraction):]))
                        field = field[:-len(contraction)]
                        remove_contractions = True
            if field:
                if self.pos_tags:
                    tokens.append(Token(field, pos=tags[idx], tag=tags[idx]))
                else:
                    tokens.append(Token(field))
            tokens.extend(add_at_end)
        return tokens

    def _can_split(self, token: str):
        return token and token.lower() not in self.special_cases

    @classmethod
    def from_params(cls, params: Params) -> 'WordSplitter':
        cut_all = params.pop_bool('cut_all', False)
        hmm = params.pop_bool('hmm', False)
        cut_for_search = params.pop_bool('cut_for_search', False)
        pos_tags = params.pop_bool('pos_tags', False)
        params.assert_empty(cls.__name__)
        return cls(cut_all, hmm, cut_for_search, pos_tags)



