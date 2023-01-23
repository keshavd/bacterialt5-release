from nltk import ngrams


def split_orf_into_kmers(orf, kmer):
    return " ".join(["".join(x) for x in ngrams(orf, kmer)])