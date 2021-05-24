from .util import is_module_available

__all__ = []
_all__checkers = []

if is_module_available("aspell"):
    from .corrector_aspell import AspellChecker

    _all__checkers.extend(["AspellChecker"])

if is_module_available("jamspell"):
    from .corrector_jamspell import JamspellChecker

    _all__checkers.extend(["JamspellChecker"])

__all__.extend(_all__checkers)
