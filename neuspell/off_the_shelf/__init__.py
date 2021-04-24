from .util import is_module_available

__all__ = []

if is_module_available("aspell"):
    from .corrector_aspell import AspellChecker

    __all__.extend(["AspellChecker"])

if is_module_available("jamspell"):
    from .corrector_jamspell import JamspellChecker

    __all__.extend(["JamspellChecker"])
