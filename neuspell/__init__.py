
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/.")



# 10 spell checkers

from corrector_aspell import CorrectorAspell as AspellChecker
from corrector_jamspell import CorrectorJamspell as JamspellChecker

from corrector_cnnlstm import CorrectorCnnLstm as CnnlstmChecker
from corrector_sclstm import CorrectorSCLstm as SclstmChecker
from corrector_lstmlstm import CorrectorLstmLstm as NestedlstmChecker
from corrector_subwordbert import CorrectorSubwordBert as BertChecker

from corrector_elmosclstm import CorrectorElmoSCLstm as ElmosclstmChecker
from corrector_sclstmelmo import CorrectorSCLstmElmo as SclstmelmoChecker
from corrector_bertsclstm import CorrectorBertSCLstm as BertsclstmChecker
from corrector_sclstmbert import CorrectorSCLstmBert as SclstmbertChecker
