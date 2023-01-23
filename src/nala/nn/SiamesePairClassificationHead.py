from ..constants import siamese_head
from .bases.PairClassificationBase import PairClassificationBase


class SiamesePairClassificationHead(PairClassificationBase):
    def __init__(self, options=None, providers=["CPUExecutionProvider"]):
        super().__init__(
            onnx_fh=siamese_head,
            options=options,
            providers=providers,
        )
