from dataclasses import dataclass
from enum import Enum


class FileType(Enum):
    """File types supported by the plot extractor."""

    PNG = "png"
    JPEG = "jpeg"
    PDF = "pdf"
    SVG = "svg"
    CSV = "csv"
    JSON = "json"
    TXT = "txt"
    HTML = "html"


@dataclass
class PlotOutput:
    """Represents a plot/chart output."""

    format: FileType
    content_base64: str
    width: int | None = None
    height: int | None = None
    dpi: int | None = None
