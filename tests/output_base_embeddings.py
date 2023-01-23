from nala.nn.BacterialT5Encoder import BacterialT5Encoder
from nala.encode.tokenizers.DNABertTokenizer import DNABertTokenizer
from nala.encode.preprocessing import split_orf_into_kmers
model = BacterialT5Encoder()

tokenizer = DNABertTokenizer(add_sentinel=True)
seq_example = "ATGCTAGAATTAATTTGCATTTCTTATAACGACTATTTAATTAGGGGGGATTTCAAAGTG"
kmers = split_orf_into_kmers(seq_example, 6)
tokens = tokenizer(kmers, return_tensors='np', padding="max_length", max_length=1000)