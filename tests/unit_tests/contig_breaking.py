import unittest
from parameterized import parameterized

"""
Test breaking contigs into pieces
"""


def get_broken_seqs(fna_fh, num_breaks):
    from nala.seq.genome_processing import fna_to_df, sequence_break_dist
    import os

    path = os.path.dirname(__file__)
    df = fna_to_df(f"{path}/{fna_fh}")
    seq = df["nucleotide_seq"][0]
    seqs = sequence_break_dist(seq, num_breaks=num_breaks)
    return seqs


class TestContigBreaking(unittest.TestCase):
    @parameterized.expand(
        [
            ("../dat/vancomycin.fasta", 10),
        ]
    )
    def test_reading_fasta(self, fna_fh, piece_count):
        self.assertEqual(len(get_broken_seqs(fna_fh, piece_count - 1)), piece_count)
