from .. import __version__

from pathlib import Path
from collections import Counter
from torch import dtype

import pyfiglet
import colorama
import warnings
import logging as logger
import time

from tktkt.util.printing import percent, intsep, roundHuman

__all__ = ["__version__", "log", "warn", "printLamotoWelcome", "formatTypeDistribution"]


def log(*args, **kwargs):
    print(f"[LaMoTO | {time.strftime('%Y-%m-%d %H:%M:%S')}]", *args, **kwargs)


def _monkeypatched_print_warning(message, category, filename, lineno, file=None, line=None):
    bold = colorama.Style.BRIGHT
    unbold = colorama.Style.NORMAL
    path = Path(filename)
    print(f"{colorama.Fore.YELLOW}{bold}{category.__name__} ({path.parent.name}/{path.name} at L{lineno}):{unbold} {message}{colorama.Fore.RESET}")

warnings.showwarning = _monkeypatched_print_warning

from warnings import warn as _warn
warn = _warn


def printLamotoWelcome():
    # Render text
    text = pyfiglet.figlet_format("LaMoTO", font="banner3-D")  # The decent fonts are: "banner3-D", "doh", "basic", "big", "colossal", "isometric1", "larry3d", "lean", "roman", "rowancap", "standard", "starwars", "univers"

    # Pad text (font-specific code)
    lines = text.split("\n")
    lines = [line for line in lines if line.strip()]

    line_length = len(lines[0])
    lines.insert(0, ":"*line_length)
    for i, line in enumerate(lines):
        lines[i] = "::"+ line + ":"

    # Make border
    for i, line in enumerate(lines):
        lines[i] = "| "+ line + " |"

    line_length = len(lines[0])
    lines.insert(0, "+" + "-"*(line_length-2) + "+")
    version_string = "--" + __version__
    lines.append("+" + version_string + "-"*(line_length-2 - len(version_string)) + "+")

    # Print
    print()
    print("\n".join(lines))
    print()


def formatTypeDistribution(type_counts: Counter[dtype], min_fraction_to_show: float=0.005) -> list[tuple[str,str,str]]:
    """
    Given a counter with the amount of weights of each PyTorch dtype in a model, format the type, amount and fraction
    to be human-readable.
    """
    total = type_counts.total()
    return [
        (f"{dt}", roundHuman(n, round_to=0), percent(n, total, decimals=1) + "%")
        for dt, n in sorted(type_counts.items(), key=lambda kv: kv[1], reverse=True)
        if n/total >= min_fraction_to_show
    ]
