def get_nucleotide_seq_length(fna_fh):
    from nala.seq.genome_processing import fna_to_df, sequence_break_dist
    import os

    path = os.path.dirname(__file__)
    df = fna_to_df(f"{path}/{fna_fh}")
    seq = df["nucleotide_seq"][0]
    seqs = sequence_break_dist(seq)
    return seqs