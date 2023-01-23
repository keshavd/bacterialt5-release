from .bases.ONNXBase import ONNXBase
from ..constants import siamese_head
from ..structs import SequenceClassifierOutput

class SiamesePairClassificationHead(ONNXBase):
    def __init__(self, options=None, providers=["CPUExecutionProvider"]):
        super().__init__(
            onnx_fh=siamese_head,
            options=options,
            providers=providers,
        )

    def get_output(self, input_a, input_b, token_type_ids=None):
        output = self.session.run(
            None,
            {
                "input_a": input_a,
                "input_b": input_b
            },
        )
        return SequenceClassifierOutput(
            logits=output[0]
        )
