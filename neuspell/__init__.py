__version__ = "0.9.0"
__author__ = 'Sai Muralidhar Jayanthi, Danish Pruthi, and Graham Neubig'
__email__ = "jsaimurali001@gmail.com"

# from .corrector_aspell import CorrectorAspell as AspellChecker
# from .corrector_jamspell import CorrectorJamspell as JamspellChecker
from .corrector_bertsclstm import CorrectorBertSCLstm as BertsclstmChecker
from .corrector_cnnlstm import CorrectorCnnLstm as CnnlstmChecker
from .corrector_lstmlstm import CorrectorLstmLstm as NestedlstmChecker
from .corrector_sclstm import CorrectorSCLstm as SclstmChecker
from .corrector_sclstmbert import CorrectorSCLstmBert as SclstmbertChecker
from .corrector_subwordbert import CorrectorSubwordBert as BertChecker
from .seq_modeling.util import is_module_available

if is_module_available("allennlp"):
    from .corrector_elmosclstm import CorrectorElmoSCLstm as ElmosclstmChecker
    from .corrector_sclstmelmo import CorrectorSCLstmElmo as SclstmelmoChecker

    __all__ = [
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
    __all__ = [
        "BertsclstmChecker",
        "CnnlstmChecker",
        "NestedlstmChecker",
        "SclstmChecker",
        "SclstmbertChecker",
        "BertChecker"
    ]
