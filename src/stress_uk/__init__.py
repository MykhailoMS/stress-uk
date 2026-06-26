"""stress-uk - розставляє наголоси в українському тексті символом U+0301.

    from stress_uk import stressify_text
    print(stressify_text("Привіт, як справи?"))
"""
from .infer import Stressifier, get_default_stressifier, stressify, stressify_text

__all__ = ["Stressifier", "get_default_stressifier", "stressify", "stressify_text"]
__version__ = "0.1.1"
