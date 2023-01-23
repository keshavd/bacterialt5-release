from nala.nn.BacterialT5Encoder import BacterialT5Encoder
from nala.nn.SiamesePairClassificationHead import SiamesePairClassificationHead
from nala.encode.preprocessing import split_orf_into_kmers

input_seq = "ATGCTAGAATTAATTTGCATTTCTTATAACGACTATTTAATTAGGGGGGATTTCAAAGTG"
kmer_size = 6
seq_max_length = 69

model = BacterialT5Encoder()
kmers = split_orf_into_kmers(input_seq, kmer=kmer_size)
tokens = model.tokenizer(kmers, return_tensors='np', padding="max_length", max_length=seq_max_length)
out = model.get_output(**tokens)['last_hidden_state']
out1, out2 = out, out
cls_head = DomainClassificationHead()
cls_head.get_output(input_a=out1, input_b=out2)
