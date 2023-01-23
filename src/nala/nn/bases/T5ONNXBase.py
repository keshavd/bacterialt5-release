from .ONNXBase import ONNXBase
from ...structs import BaseModelOutputWithPooling


class T5ONNXBase(ONNXBase):
    def get_output(self, input_ids, attention_mask, token_type_ids=None):
        if input_ids.shape[0] > self.max_batch_size:
            raise ValueError("Input batch size is greater than max batch size")
        output = self.session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
        )
        return BaseModelOutputWithPooling(last_hidden_state=output[0])
