# test_this1.py
import unittest
from parameterized import parameterized

"""
Testing Output Shape

* Tests if model loads
* Tests if tokenizes correctly
* Tests if onnx model can generate output
* Tests if output shape is correct
"""

def get_base_output(input_seq, seq_max_length=69, kmer_size=6):
    from nala.nn.BacterialT5Encoder import BacterialT5Encoder
    from nala.encode.preprocessing import split_orf_into_kmers

    model = BacterialT5Encoder()
    kmers = split_orf_into_kmers(input_seq, kmer=kmer_size)
    tokens = model.tokenizer(
        kmers, return_tensors="np", padding="max_length", max_length=seq_max_length
    )
    out = model.get_output(**tokens)["last_hidden_state"]
    return out.shape


class TestBaseOutputShape(unittest.TestCase):
    @parameterized.expand(
        [
            (
                "random",
                "ATGCTAGAATTAATTTGCATTTCTTATAACGACTATTTAATTAGGGGGGATTTCAAAGTG",
                (1, 69, 1024),
            ),
        ]
    )
    def test_output_shape(self, name, input, expected):
        assert get_base_output(input) == expected
