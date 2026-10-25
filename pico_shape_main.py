"""
Raspberry Pi Pico - Shape Name Display over USB Serial
=======================================================
Uses the official ssd1306.py driver (must already be saved on the Pico
as ssd1306.py) plus a non-blocking USB-serial reader that displays
whatever shape NAME the PC app sends (e.g. "circle", "pentagon",
"parallelogram").

Files needed on the Pico:
    main.py      <- this file (runs automatically on boot)
    ssd1306.py   <- the official MicroPython SSD1306 driver

Wiring (same as the digit project):
    SSD1306 SDA -> Pico GPIO 4
    SSD1306 SCL -> Pico GPIO 5
    SSD1306 VCC -> 3V3
    SSD1306 GND -> GND

Why text instead of drawing the actual shape outline: the PC side can
tell apart 19 similar classes (nonagon vs decagon etc.), but re-drawing
a recognizable outline on a 128x64 1-bit OLED for all 19 shapes adds a
lot of fragile per-shape drawing code for very little payoff -  a big,
centered text label is simpler and 100% reliable. If later you want
actual outline icons on the OLED instead of text, that can be added as
its own phase.
"""

import sys
import select
from machine import Pin, SoftI2C
from ssd1306 import SSD1306_I2C

# ======================================================================
# OLED setup
# ======================================================================
i2c = SoftI2C(scl=Pin(5), sda=Pin(4), freq=400000)
oled = SSD1306_I2C(128, 64, i2c)

VALID_SHAPES = (
    "circle", "oval", "triangle", "square", "rectangle",
    "parallelogram", "rhombus", "trapezoid", "kite",
    "pentagon", "hexagon", "heptagon", "octagon", "nonagon", "decagon",
    "star", "heart", "crescent", "semicircle",
)

CHAR_W = 8  # default SSD1306 font is 8x8 px per character


def show_message(line1, line2=""):
    oled.fill(0)
    oled.text(line1, 0, 0)
    if line2:
        oled.text(line2, 0, 16)
    oled.show()


def show_shape_name(name):
    """Big-ish centered label: name on one line (scale 2 if it fits,
    otherwise fall back to normal size so long names don't clip)."""
    oled.fill(0)
    label = name.upper()

    # Try to fake a 2x scale by drawing the text twice with a 1px offset
    # (cheap "bold" look) only if it comfortably fits at that width.
    normal_w = len(label) * CHAR_W
    if normal_w <= oled.width:
        x = (oled.width - normal_w) // 2
        y = (oled.height - CHAR_W) // 2
        oled.text(label, x, y)
        oled.text(label, x + 1, y)  # bold effect
    else:
        # Long name (e.g. "PARALLELOGRAM") - shrink by wrapping onto 2 lines
        mid = len(label) // 2
        # wrap on nearest split so we don't cut mid-letter awkwardly
        line1, line2 = label[:mid], label[mid:]
        x1 = max(0, (oled.width - len(line1) * CHAR_W) // 2)
        x2 = max(0, (oled.width - len(line2) * CHAR_W) // 2)
        oled.text(line1, x1, 20)
        oled.text(line2, x2, 34)

    oled.show()


# ======================================================================
# Main loop: read one shape name per line from USB serial and display it
# ======================================================================
show_message("Shape Receiver", "Waiting for PC...")

poll = select.poll()
poll.register(sys.stdin, select.POLLIN)

buffer = ""

while True:
    events = poll.poll(100)  # 100 ms so the loop stays responsive
    if not events:
        continue

    char = sys.stdin.read(1)
    if char in ("\n", "\r"):
        line = buffer.strip().lower()
        buffer = ""
        if not line:
            continue
        if line in VALID_SHAPES:
            show_shape_name(line)
        else:
            show_message("Invalid data:", line[:16])
    else:
        buffer += char
        if len(buffer) > 32:  # safety: drop runaway garbage input
            buffer = ""
