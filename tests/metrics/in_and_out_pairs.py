from nala.seq.genome_processing import fna_to_df, sequence_break_dist
from nala.nn.BacterialT5Encoder import BacterialT5Encoder
from nala.encode.preprocessing import split_orf_into_kmers
from nala.nn.SiamesePairClassificationHead import SiamesePairClassificationHead
from scipy.special import softmax

model = BacterialT5Encoder()
siamese_model = SiamesePairClassificationHead()

num_breaks = 4
kmer_size = 6
seq_max_length = 1024

df = fna_to_df(
    "/mnt/storage/grid/home/keshav/projects/bacterialt5-release/tests/dat/vancomycin.fasta",
    break_fn=lambda x:sequence_break_dist(x, num_breaks=num_breaks),
)
df2 = fna_to_df(
    "/mnt/storage/grid/home/keshav/projects/bacterialt5-release/tests/dat/avermectin.fasta",
    break_fn=lambda x:sequence_break_dist(x, num_breaks=num_breaks),
)

all_bases = []
for row in df.itertuples():
    right = row.nucleotide_seq[-seq_max_length:]
    left = row.nucleotide_seq[:seq_max_length]
    for input_seq in [right, left]:
        kmers = split_orf_into_kmers(input_seq, kmer=kmer_size)
        tokens = model.tokenizer(
            kmers, return_tensors="np", padding="max_length", max_length=seq_max_length
        )
        out = model.get_output(**tokens)["last_hidden_state"]
        all_bases.append(out)

all_bases2 = []
for row in df2.itertuples():
    right = row.nucleotide_seq[-seq_max_length:]
    left = row.nucleotide_seq[:seq_max_length]
    for input_seq in [right, left]:
        kmers = split_orf_into_kmers(input_seq, kmer=kmer_size)
        tokens = model.tokenizer(
            kmers, return_tensors="np", padding="max_length", max_length=seq_max_length
        )
        out = model.get_output(**tokens)["last_hidden_state"]
        all_bases2.append(out)

for i in range(len(all_bases)):
    for j in range(i + 1, len(all_bases)):
        # In
        logits = siamese_model.get_output(all_bases[i], all_bases[j])["logits"]
        probs = softmax(logits)
        if probs[0][0] > 0.9:
            print(f"In - {i} vs {j}: {probs}")
        # Out
        logits = siamese_model.get_output(all_bases[i], all_bases2[j])["logits"]
        probs = softmax(logits)
        if probs[0][0] > 0.9:
            print(f"Out - {i} vs {j}: {probs}")
