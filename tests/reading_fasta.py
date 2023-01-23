import unittest
from parameterized import parameterized

"""
Test FNA to DF function

* Reads in FNA File
* Generates DataFrame
* Returns Length of Nucleotide Sequence

"""
def get_nucleotide_seq_length(fna_fh):
    from nala.seq.genome_processing import fna_to_df
    import os

    path = os.path.dirname(__file__)
    df = fna_to_df(f"{path}/{fna_fh}")
    return df["nucleotide_seq"][0].__len__()


class TestReadingFasta(unittest.TestCase):
    @parameterized.expand(
        [
            ("vancomycin.fasta", 65044),
        ]
    )
    def test_reading_fasta(self, fna_fh, expected_length):
        self.assertEqual(get_nucleotide_seq_length(fna_fh), expected_length)
