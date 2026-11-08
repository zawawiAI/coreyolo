"""CoreYOLO: independent detector with a Core ML deployment path."""

from coreyolo.nn.model import CoreYOLO, build_model
from coreyolo.results import Result
from coreyolo.sdk import Detector
from coreyolo.utils import __version__

__all__ = ["Detector", "Result", "CoreYOLO", "build_model", "__version__"]
