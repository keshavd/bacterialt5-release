from ...structs import SequenceClassifierOutput
from .ONNXBase import ONNXBase


class PairClassificationBase(ONNXBase):
    def get_output(self, input_a, input_b, token_type_ids=None):
        output = self.session.run(
            None,
            {"input_a": input_a, "input_b": input_b},
        )
        return SequenceClassifierOutput(logits=output[0])
