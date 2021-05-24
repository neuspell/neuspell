import os

from . import off_the_shelf
from . import seq_modeling
from .corrector_bertsclstm import BertsclstmChecker
from .corrector_cnnlstm import CnnlstmChecker
from .corrector_lstmlstm import NestedlstmChecker
from .corrector_sclstm import SclstmChecker
from .corrector_sclstmbert import SclstmbertChecker
from .corrector_subwordbert import BertChecker
from ..util import is_module_available

__all__ = []
_all__checkers = []

if is_module_available("allennlp"):
    from .corrector_elmosclstm import ElmosclstmChecker
    from .corrector_sclstmelmo import SclstmelmoChecker

    _all__checkers = [
        "BertsclstmChecker",
        "CnnlstmChecker",
        "ElmosclstmChecker",
        "NestedlstmChecker",
        "SclstmChecker",
        "SclstmbertChecker",
        "SclstmelmoChecker",
        "BertChecker"
    ]

else:
    _all__checkers = [
        "BertsclstmChecker",
        "CnnlstmChecker",
        "NestedlstmChecker",
        "SclstmChecker",
        "SclstmbertChecker",
        "BertChecker"
    ]
_all__checkers.extend(off_the_shelf._all__checkers)


def available_checkers():
    return _all__checkers


class NeuralCheckersFactory:
    NAME_TO_CHECKER_MAPPINGS = {
        "bertscrnn-probwordnoise": BertsclstmChecker,
        "cnn-lstm-probwordnoise": CnnlstmChecker,
        "lstm-lstm-probwordnoise": NestedlstmChecker,
        "scrnn-probwordnoise": SclstmChecker,
        "scrnnbert-probwordnoise": SclstmbertChecker,
        "subwordbert-probwordnoise": BertChecker
    }

    if is_module_available("allennlp"):
        NAME_TO_CHECKER_MAPPINGS.update({
            "elmoscrnn-probwordnoise": ElmosclstmChecker,
            "scrnnelmo-probwordnoise": SclstmelmoChecker,
        })

    @staticmethod
    def from_pretrained(name_or_path, **kwargs):

        if os.path.exists(name_or_path):
            # name = os.path.split(name_or_path)[-1]
            msg = "To load a model from a path, directy use checker name as XxxChecker " \
                  "instead of using NeuralCheckersFactory"
            raise NotImplementedError(msg)

        # create appropriate corrector
        try:
            kwargs.update({"name": name_or_path})
            checker = NeuralCheckersFactory.NAME_TO_CHECKER_MAPPINGS[name_or_path](**kwargs)
            checker.from_pretrained()
            return checker
        except KeyError as e:
            msg = f"Found checker name: {name_or_path}. " \
                  f"Expected a checker name in " \
                  f"{[*NeuralCheckersFactory.NAME_TO_CHECKER_MAPPINGS.keys()]}"
            raise Exception(msg) from e


__all__.extend(_all__checkers)
__all__.extend(["available_checkers"])
__all__.extend(["seq_modeling"])
__all__.extend(["NeuralCheckersFactory"])
