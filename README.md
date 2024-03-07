# Bacterial T5

BacterialT5 is a T5 model trained end-to-end on a variety of different bacterial genomes.

This repository is the release package of the BacterialT5 module. All helper functions are included in the package. This includes:

- Preprocessing FASTA files into data structures more appropriate for embedding (includes DNA-BERT Tokenization, windowing, and recognizing genes etc.)
- Embedding the FASTA file via the Encoder side of the BacterialT5 model.
- Classifying pairs of FASTA sequences based on a custom Siamese Head architecture.

## Table of Contents

- Requirements
- Installation
- Configuration
- Usage
- Troubleshooting
- FAQ

## Requirements

- [transformers](https://pypi.org/project/transformers/)
- [onnxruntime](https://pypi.org/project/onnxruntime/)
- [NetworkX](https://networkx.org/)
- [Pandas](https://pandas.pydata.org/)
- [BioPython](https://biopython.org/)
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)


## Installation

The package is pip installable.

```
git pull https://github.com/keshavd/bacterialt5-release.git
cd bacterialt5-release
pip install .
```

## Configuration

Until the publication is released, models are currently distributed on a per-use case basis. Open an issue to request access for the models.
Modification of the model locations can be made in the [constants file](https://github.com/keshavd/bacterialt5-release/blob/master/src/nala/constants.py)

## Usage

### Embedding
To embed with the encoder side of BacterialT5

```
from nala.nn.BacterialT5Encoder import BacterialT5Encoder
from nala.encode.preprocessing import split_orf_into_kmers


# Instantiate Model
base_model = BacterialT5Encoder(
            version="v2", providers=["CPUExecutionProvider"]
        )

# Transform the input DNA sequence
kmers = split_orf_into_kmers(input_seq, kmer=6)
tokens = base_model.tokenizer(
        kmers, return_tensors="np", padding="max_length", max_length=max_length
    )

# Embed
embedding = base_model.get_output(**tokens)["last_hidden_state"]
```
### Siamese Classification

To Calculate the probability of two sequences being spatially related to each other after being embedded.

```
from nala.nn.SiamesePairClassificationHead import SiamesePairClassificationHead

# Instantiate the classification head

siamese_model = SiamesePairClassificationHead(
            version="v2", providers=["CPUExecutionProvider"]
        )

# Calculate Probabilities between two embeddings

logits = siamese_model.get_output(emb_i, emb_j)["logits"]
proba = softmax(logits)
```

## Troubleshooting

* Ensure you are using a kmer of 6 for the most effective perfomance
* This only works for Bacterial and Archea Genomes not Fungal and/or Eukaryotic.
* Use the (unit test suite)[https://github.com/keshavd/bacterialt5-release/tree/master/tests] to see where the package is failing.

## FAQ

### What is NALA?

NALA or Neural-network-guided Arrangement and Linkage Assembly, is a the in-house Magarvey Lab assembly pipeline.
It uses a variety of different techniques to create robust bacterial genome assemblies even when conventional techniques fail.
The publication is current in process. A pre-print will soon be available (May 2024).

BacterialT5 is used as one of the methods for assembly polishing.

### What is ONNX?

ONNX is a framework for quantizing deep learning models. Using ONNX for quanitization and the ONNX Runtime Framework (ORT),
it is possible to run models with large gains in speeds.

## Can I have access to the tooling used to create this model?

All tooling related to training T5 on consumer level hardware will be released with the publication :).
