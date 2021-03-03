__version__ = "0.7.0"
__author__ = 'Sai Muralidhar Jayanthi, Danish Pruthi, and Graham Neubig'
__email__ = "jsaimurali001@gmail.com"

"""10 spell checkers"""

# from .corrector_aspell import CorrectorAspell as AspellChecker
# from .corrector_jamspell import CorrectorJamspell as JamspellChecker

from .corrector_bertsclstm import CorrectorBertSCLstm as BertsclstmChecker
from .corrector_cnnlstm import CorrectorCnnLstm as CnnlstmChecker
from .corrector_elmosclstm import CorrectorElmoSCLstm as ElmosclstmChecker
from .corrector_lstmlstm import CorrectorLstmLstm as NestedlstmChecker
from .corrector_sclstm import CorrectorSCLstm as SclstmChecker
from .corrector_sclstmbert import CorrectorSCLstmBert as SclstmbertChecker
from .corrector_sclstmelmo import CorrectorSCLstmElmo as SclstmelmoChecker
from .corrector_subwordbert import CorrectorSubwordBert as BertChecker

__all__ = [
    "BertsclstmChecker",
    "CnnlstmChecker",
    "ElmosclstmChecker",
    "NestedlstmChecker",
    "SclstmChecker",
    "SclstmbertChecker",
    "SclstmelmoChecker",
    "BertChecker",
]
