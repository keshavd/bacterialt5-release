from Bio import SeqIO
import xxhash as xxh
import pandas as pd
import pyrodigal
from tqdm import tqdm


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
