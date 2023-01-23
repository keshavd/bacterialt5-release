# test_this1.py
import unittest
from parameterized import parameterized

"""
Test Siamese Output Shape

* Tests if model loads
* Tests if tokenizes correctly
* Tests if onnx model can generate output
* Tests if output shape is correct

"""

def get_siamese_output(input_a, input_b, seq_max_length=69, kmer_size=6):
    from nala.nn.BacterialT5Encoder import BacterialT5Encoder
    from nala.nn.SiamesePairClassificationHead import SiamesePairClassificationHead
    from nala.encode.preprocessing import split_orf_into_kmers

    model = BacterialT5Encoder()
    kmers_a = split_orf_into_kmers(input_a, kmer=kmer_size)
    tokens_a = model.tokenizer(
        kmers_a, return_tensors="np", padding="max_length", max_length=seq_max_length
    )
    out_a = model.get_output(**tokens_a)["last_hidden_state"]
    kmers_b = split_orf_into_kmers(input_b, kmer=kmer_size)
    tokens_b = model.tokenizer(
        kmers_b, return_tensors="np", padding="max_length", max_length=seq_max_length
    )
    out_b = model.get_output(**tokens_b)["last_hidden_state"]
    siamese_model = SiamesePairClassificationHead()
    siamese_out = siamese_model.get_output(out_a, out_b)
    return siamese_out.shape


class TestBaseOutputShape(unittest.TestCase):
    @parameterized.expand(
        [
            (
                "random",
                "ATGCTAGAATTAATTTGCATTTCTTAT",
                "AACGACTATTTAATTAGGGGGGATTTCAAAGTG",
                (1, 2),
            ),
        ]
    )
    def test_siamese_shape(self, name, input_a, input_b, expected):
        assert get_siamese_output(input_a, input_b) == expected
