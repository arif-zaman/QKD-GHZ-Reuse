"""Render Figure 1 from the analytical readout/reset probabilities.

ReportLab is imported only when the figure is requested. The PDF uses embedded
TeX Times fonts when installed and standard Times-Roman otherwise. The open
endpoint at delta=1 marks an undefined conditional readout accuracy.
"""
from pathlib import Path
import shutil
import subprocess
import sys

from reportlab.graphics import renderPDF
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Circle, Drawing, Line, String
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics


READOUT_STRENGTH = 0.5
BLUE = colors.HexColor("#155B85")
ORANGE = colors.HexColor("#B4511D")
DARK = colors.HexColor("#252525")
FONT = "FigureTimes"


def figure_font():
    """Embed TeX Times when available, with a portable PDF base-font fallback."""
    kpsewhich = shutil.which("kpsewhich")
    if kpsewhich:
        try:
            afm, pfb = [
                subprocess.check_output(
                    [kpsewhich, name], text=True, stderr=subprocess.DEVNULL
                ).strip()
                for name in ("utmr8a.afm", "utmr8a.pfb")
            ]
            if Path(afm).is_file() and Path(pfb).is_file():
                face = pdfmetrics.EmbeddedType1Face(afm, pfb)
                pdfmetrics.registerTypeFace(face)
                pdfmetrics.registerFont(
                    pdfmetrics.Font(FONT, face.name, "WinAnsiEncoding")
                )
                return FONT
        except (OSError, subprocess.CalledProcessError):
            pass
    print(
        "TeX Times fonts unavailable; using Times-Roman (not embedded in the PDF).",
        file=sys.stderr,
    )
    return "Times-Roman"


def make_figure(output):
    font = figure_font()
    renderPDF.STATE_DEFAULTS["fontName"] = font
    drawing = Drawing(252, 155)
    chart = LinePlot()
    chart.x, chart.y = 34, 28
    chart.width, chart.height = 207, 95
    delta = [i / 200 for i in range(201)]
    chart.data = [
        [(d, (1 + READOUT_STRENGTH * d) / 2) for d in delta],
        [(d, 0.5) for d in delta],
        [(d, (1 - d * d) / 2) for d in delta],
    ]
    chart.joinedLines = 1
    chart.lines[0].strokeColor = BLUE
    chart.lines[0].strokeWidth = 1.4
    chart.lines[1].strokeColor = DARK
    chart.lines[1].strokeWidth = 1.1
    chart.lines[1].strokeDashArray = [1.5, 2]
    chart.lines[2].strokeColor = ORANGE
    chart.lines[2].strokeWidth = 1.4
    chart.lines[2].strokeDashArray = [5, 2.5]
    for axis in [chart.xValueAxis, chart.yValueAxis]:
        axis.valueMin, axis.valueMax = 0, 1
        axis.strokeColor = DARK
        axis.strokeWidth = 0.55
        axis.labels.fontName = font
        axis.labels.fontSize = 8.5
        axis.labelTextFormat = "%.1f"
    chart.xValueAxis.valueSteps = [0, .2, .4, .6, .8, 1]
    chart.yValueAxis.valueSteps = [0, .25, .5, .75, 1]
    chart.yValueAxis.labelTextFormat = "%.2g"
    chart.xValueAxis.labels.dy = -2
    chart.yValueAxis.labels.dx = -3
    chart.yValueAxis.visibleGrid = 1
    chart.yValueAxis.gridStrokeColor = colors.HexColor("#DDDDDD")
    chart.yValueAxis.gridStrokeWidth = .35
    drawing.add(chart)
    drawing.add(String(137.5, 5, "Population contrast", textAnchor="middle",
                       fontName=font, fontSize=9))
    drawing.add(String(34, 128, "Probability", fontName=font, fontSize=9))
    # Compact external legend; line styles remain distinct in grayscale.
    legend = [
        (4, 148, BLUE, None, "Before reset: P(y=x)"),
        (133, 148, DARK, [1.5, 2], "Accepted: P(y=x | S)"),
        (133, 135, ORANGE, [5, 2.5], "Reset acceptance: r"),
    ]
    for x, y, color, dash, label in legend:
        drawing.add(Line(x, y, x + 15, y, strokeColor=color,
                         strokeWidth=1.2, strokeDashArray=dash))
        drawing.add(String(x + 19, y - 2.8, label, fontName=font,
                           fontSize=8.1))
    drawing.add(Circle(chart.x + chart.width, chart.y + chart.height / 2,
                       2.3, fillColor=colors.white, strokeColor=DARK,
                       strokeWidth=1.0))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    renderPDF.drawToFile(drawing, str(output), initialFontName=font)


