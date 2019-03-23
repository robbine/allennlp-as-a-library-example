from typing import Tuple
import json
from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.service.predictors.predictor import Predictor
from allennlp.data.tokenizers.word_splitter import BertBasicWordSplitter


@Predictor.register('joint_intent_slot_predictor')
class JointIntentSlotPredictor(Predictor):
    """"Predictor wrapper for the AcademicPaperClassifier"""

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        query = json_dict['query']
        instance = self._dataset_reader.text_to_instance(
            query, tags=None, label=None)
        return instance


@Predictor.register('joint_intent_slot_predictor_plain')
class JointIntentSlotPredictorPlain(Predictor):
    """"Predictor wrapper for the AcademicPaperClassifier"""

    @overrides
    def load_line(self, line: str) -> JsonDict:
        line = line.strip()
        if line == '':
            raise Exception('empty query to predict')
        return {'query': line}

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        query = json_dict['query']
        instance = self._dataset_reader.text_to_instance(
            query, tags=None, label=None)
        return instance

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:  # pylint: disable=no-self-use
        """
        If you don't want your outputs in JSON-lines format
        you can override this function to output them differently.
        """
        first_intent_score = outputs['top 3 intents'][0]
        first_intent = first_intent_score.split(':')[0]
        first_intent = first_intent.replace('anna_assistant/', '')
        return first_intent + "\n"
