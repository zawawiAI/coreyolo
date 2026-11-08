"""YOLOSilicon: independent detector with a Core ML deployment path."""

from yolosilicon.nn.model import YOLOSilicon, build_model
from yolosilicon.results import Result
from yolosilicon.sdk import Detector
from yolosilicon.utils import __version__

__all__ = ["Detector", "Result", "YOLOSilicon", "build_model", "__version__"]
