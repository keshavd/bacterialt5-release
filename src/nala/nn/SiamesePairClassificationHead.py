from ..constants import model_lookup
from .bases.PairClassificationBase import PairClassificationBase


class SiamesePairClassificationHead(PairClassificationBase):
    def __init__(self, version="v1", options=None, providers=["CPUExecutionProvider"]):
        super().__init__(
            onnx_fh=model_lookup[version]["classification_head"],
            options=options,
            providers=providers,
        )
