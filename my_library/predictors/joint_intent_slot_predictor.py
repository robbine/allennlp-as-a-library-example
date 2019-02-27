from typing import Tuple

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
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        print('----------->')
        query = json_dict['query']
        instance = self._dataset_reader.text_to_instance(
            query, tags=None, label=None)
        return instance
