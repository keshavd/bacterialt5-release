# Tokenizers
dna_bert_tokenizer = "/mnt/storage/grid/var/models/dnabert-6-base/6-new-12w-0"

# Transformer Models
batch_size = 32
gpu_id = 0

# ONNX Models
model_lookup = {
    "v1": {
        "encoder": "/mnt/storage/grid/var/models/onnx/bacterial_t5/v1/encoder.onnx",
        "classification_head": "/mnt/storage/grid/var/models/onnx/bacterial_t5/v1/siamese_head.onnx",
    },
    "v2": {
        "encoder": "/mnt/storage/grid/var/models/onnx/bacterial_t5/v2/encoder.onnx",
        "classification_head": "/mnt/storage/grid/var/models/onnx/bacterial_t5/v2/siamese_head.onnx",
    },
    "v3": {
        "encoder": "/mnt/storage/grid/var/models/onnx/bacterial_t5/v3/encoder.onnx",
        "classification_head": "/mnt/storage/grid/var/models/onnx/bacterial_t5/v3/siamese_head.onnx",
    },
}
