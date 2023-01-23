from ...structs import SequenceClassifierOutput
from .BertONNXBase import BertONNXBase


class PairClassificationBase(BertONNXBase):
    def get_output(self, input_a, input_b, token_type_ids=None):
        output = self.session.run(
            None,
            {"input_a": input_a, "input_b": input_b},
        )
        return SequenceClassifierOutput(logits=output[0])
