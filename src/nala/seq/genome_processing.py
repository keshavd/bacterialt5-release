from Bio import SeqIO
import xxhash as xxh
import pandas as pd


def fna_to_df(fna_fh) -> pd.DataFrame:
    """
    Generate a DataFrame from an FNA file
    :param fna_fh:
    :return:
    """
    contigs = list(SeqIO.parse(fna_fh, "fasta"))
    contig_rows = []
    for c in contigs:
        nucleotide_seq = str(c.seq)
        contig_id = str(c.id)
        contig_len = len(nucleotide_seq)
        row = {
            "contig_id": contig_id,
            "contig_len": contig_len,
            "nucleotide_seq": nucleotide_seq,
            "nuc_id": xxh.xxh32_intdigest(nucleotide_seq),
        }
        contig_rows.append(row)
    contig_df = pd.DataFrame(contig_rows)
    return contig_df


def sequence_break_n(seq, n):
    return [seq[i : i + n] for i in range(0, len(seq), n)]


def sequence_break_dist(seq, seed=69, num_breaks=10, mu=0, sigma=1):
    import numpy as np
    from scipy.special import softmax
    from itertools import accumulate

    seq_len = seq.__len__()
    np.random.seed(seed)
    dist = np.random.normal(mu, sigma, num_breaks)
    dist = softmax(dist)
    dist = dist * seq_len
    dist = dist.astype(int)
    dist = list(accumulate(dist, lambda x, y: x + y))
    pieces = list(
        map(lambda x: seq[x[0] : x[1]], zip([0] + list(dist), dist + [seq_len]))
    )
    print(dist)
    return pieces
