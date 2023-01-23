from transformers import BertTokenizerFast
from src.nala.constants import dna_bert_tokenizer


class DNABertTokenizer(BertTokenizerFast):
    def __new__(cls, *args, add_sentinel=False, **kwargs):
        fast_tokenizer = BertTokenizerFast.from_pretrained(dna_bert_tokenizer)
        if add_sentinel:
            fast_tokenizer.eos_token = "[SEP]"
            fast_tokenizer.add_special_tokens(
                {"additional_special_tokens": [f"<extra_id_{i}>" for i in range(100)]}
            )
        return fast_tokenizer
