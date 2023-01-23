# Tokenizers
dna_bert_tokenizer="/mnt/storage/grid/var/models/dnabert-6-base/6-new-12w-0"
cls_token_id = 2
pad_token_id = 0

# Transformer Models
bacterial_t5_encoder_onnx_fh = "/mnt/storage/grid/var/models/onnx/bacterial_t5/encoder.onnx"
siamese_head = "/mnt/storage/grid/var/models/onnx/bacterial_t5/siamese_head.onnx"
batch_size = 32
gpu_id = 0

# Windowing
max_length = 768
step = 256

# Pair Classification
domain_cls_head_confidence_threshold = 0.65
mask_low_confidence_residues = True
use_notation_ids = True
