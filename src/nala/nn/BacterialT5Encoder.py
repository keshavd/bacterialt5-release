import time
import pandas as pd
from tqdm import tqdm
from transformers.data.data_collator import DataCollatorWithPadding
from .bases.ONNXBase import ONNXBase
from ..constants import bacterial_t5_encoder_onnx_fh
from src.nala.encode.tokenizers.DNABertTokenizer import DNABertTokenizer
from ..structs import BaseModelOutputWithPooling


class BacterialT5Encoder(ONNXBase):
    def __init__(self, options=None, providers=["CPUExecutionProvider"]):
        super().__init__(
            onnx_fh=bacterial_t5_encoder_onnx_fh,
            options=options,
            providers=providers,
        )
        self.tokenizer = DNABertTokenizer()
        self.collate_fn = DataCollatorWithPadding(
            tokenizer=self.tokenizer, padding=True, return_tensors="np"
        )

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
        return BaseModelOutputWithPooling(
            last_hidden_state=output[0]
        )

    def get_base_df_from_tokenized_df(
        self, tokenized_df, batch_size=32, debug=True, time_arr=None
    ):
        split_df = self.split_dataframe(tokenized_df, batch_size=batch_size)
        base_df = []
        for df in tqdm(split_df, desc="batches"):
            batch = self.collate_fn(
                df[["input_ids", "attention_mask"]].to_dict(
                    orient="list"
                )
            )
            if debug:
                start = time.time()
                base = self.get_output(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                end = time.time()
                time_arr.append(end - start)
            else:
                base = self.get_output(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
            for i in range(len(df)):
                base_df.append(
                    {
                        "aa_id": df.iloc[i]["aa_id"],
                        "window": df.iloc[i]["window"],
                        "last_hidden_state": base.last_hidden_state[i],
                        "pooler_output": base.pooler_output[i],
                    }
                )
        base_df = pd.DataFrame(base_df)
        return base_df
