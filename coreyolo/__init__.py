"""CoreYOLO: independent detector with a Core ML deployment path.

Not affiliated with Ultralytics. YOLO is a trademark of its owners.
Prefer ``from coreyolo import Detector``. ``YOLO`` is a compatibility alias.
"""

from coreyolo.nn.model import CoreYOLO, build_model
from coreyolo.results import Result
from coreyolo.sdk import Detector, YOLO
from coreyolo.utils import __version__

__all__ = ["Detector", "YOLO", "Result", "CoreYOLO", "build_model", "__version__"]
