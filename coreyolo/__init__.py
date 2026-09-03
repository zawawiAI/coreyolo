"""CoreYOLO: open-source YOLO-style detection with a Core ML deployment path."""

from coreyolo.nn.model import CoreYOLO, build_model
from coreyolo.results import Result
from coreyolo.sdk import YOLO
from coreyolo.utils import __version__

__all__ = ["YOLO", "Result", "CoreYOLO", "build_model", "__version__"]
