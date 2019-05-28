from typing import List, Iterator, Optional
import argparse
import sys
import json

from allennlp.common.checks import check_for_gpu, ConfigurationError
from allennlp.common.util import lazy_groups_of
from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor, JsonDict
from allennlp.data import Instance
from allennlp.common.util import import_submodules
import argparse


def parse_args():
    """Parse command line arguments.

    Args:
        None.

    Returns:
        A argparse.Namespace object which contains all parsed argument values.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-file', help='archive file path')
    parser.add_argument('--predictor', help='predictor name')
    parser.add_argument('--include-package', help='package name')
    args = parser.parse_args()
    return args


def _get_predictor(archive_file, predictor) -> Predictor:
    archive = load_archive(archive_file, cuda_device=-1)

    return Predictor.from_archive(archive, predictor)


class _PredictManager:
    def __init__(self, predictor: Predictor) -> None:
        self._predictor = predictor

    def _get_json_data(self, query) -> Iterator[JsonDict]:
        return self._predictor.load_line(query)

    def run(self, query) -> None:
        batch_json = self._get_json_data(query)
        result = self._predictor.predict_json(batch_json)
        return result


def main():
    args = parse_args()
    import_submodules(args.include_package)
    predictor = _get_predictor(args.archive_file, args.predictor)
    manager = _PredictManager(predictor)
    for line in sys.stdin:
        print(manager.run(line.strip()))


if __name__ == '__main__':
    sys.exit(main())
