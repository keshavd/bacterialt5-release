import unittest
from parameterized import parameterized

"""
Test breaking contigs into pieces
"""

def get_broken_seqs(fna_fh, piece_count):
    from nala.seq.genome_processing import fna_to_df, sequence_break_dist
    import os

    path = os.path.dirname(__file__)
    df = fna_to_df(f"{path}/{fna_fh}")
    seq = df["nucleotide_seq"][0]
    seqs = sequence_break_dist(seq, num_breaks=piece_count)
    return seqs


class TestContigBreaking(unittest.TestCase):
    @parameterized.expand(
        [
            ("vancomycin.fasta", 10),
        ]
    )
    def test_reading_fasta(self, fna_fh, piece_count):
        self.assertEqual(len(get_broken_seqs(fna_fh, piece_count)), piece_count)
