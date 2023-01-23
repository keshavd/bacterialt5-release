from nala.seq.genome_processing import fna_to_df
from pathlib import Path
path = Path(".")
fna_to_df(f"{path.absolute()}/vancomycin.fasta")

