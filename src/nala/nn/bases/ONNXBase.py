import onnxruntime as ort
from ...structs import BaseModelOutputWithPooling


class ONNXBase:
    def __init__(
        self,
        onnx_fh,
        options: ort.SessionOptions,
        providers: list = None,
        max_batch_size: int = 32,
    ):
        self.onnx_fh = onnx_fh
        self.options = options
        self.providers = providers
        self.session = None
        self.max_batch_size = max_batch_size
        self.setup()

    def setup(self):
        self.session = ort.InferenceSession(self.onnx_fh, self.options, self.providers)

    def __call__(self, *args, **kwargs):
        return self.get_output(*args, **kwargs)

    @staticmethod
    def split_dataframe(df, batch_size=32):
        batches = list()
        num_chunks = len(df) // batch_size + (1 if len(df) % batch_size else 0)
        for i in range(num_chunks):
            batches.append(df[i * batch_size : (i + 1) * batch_size])
        return batches

    def get_output(self, input_ids, attention_mask, token_type_ids):
        if input_ids.shape[0] > self.max_batch_size:
            raise ValueError("Input batch size is greater than max batch size")
        output = self.session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )
        return BaseModelOutputWithPooling(
            last_hidden_state=output[0],
            pooler_output=output[1],
        )
