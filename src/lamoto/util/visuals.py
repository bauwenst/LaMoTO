from .. import __version__

import pyfiglet
import time

import logging as logger


def log(*args, **kwargs):
    print(f"[LaMoTO | {time.strftime('%Y-%m-%d %H:%M:%S')}]", *args, **kwargs)


def warn(*args):
    logger.warning(" ".join(map(repr, args)))  # The join-map-repr is what print() does by default.


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
    version_string = "--" + "v" + __version__
    lines.append("+" + version_string + "-"*(line_length-2 - len(version_string)) + "+")

    # Print
    print()
    print("\n".join(lines))
    print()
