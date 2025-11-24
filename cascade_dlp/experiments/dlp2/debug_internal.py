"""
Test accessing internal GLiNER2 methods to get span information.
"""
from gliner2 import GLiNER2
import torch

model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
model.eval()

text = "Contact John Smith at john.smith@company.com"
pii_types = ["person name", "email address"]

print("=" * 70)
print("Testing internal GLiNER2 span extraction")
print("=" * 70)

# Create schema
schema = model.create_schema().entities(pii_types)

# Prepare record
record = {"text": text, "schema": schema.build()}

# Transform using processor
transformed = model.processor.transform_single_record(record)
print(f"\n1. Transformed keys: {transformed.keys()}")

# Format input
format_result = model.processor.format_input_with_mapping(
    transformed["schema_tokens_list"],
    transformed["text_tokens"]
)

print(f"\n2. Format result keys: {format_result.keys()}")
print(f"   Text tokens: {transformed['text_tokens']}")

# Prepare batch
prepared = {
    "text": text,
    "schema": schema.build(),
    "transformed": transformed,
    "inputs": format_result["inputs"],
    "mapped_indices": format_result["mapped_indices"],
    "subword_list": format_result["subword_list"],
    "schema_tokens_list": transformed["schema_tokens_list"],
    "batch_index": 0
}

# Encode
with torch.no_grad():
    device = next(model.encoder.parameters()).device
    inputs = {k: v.to(device) for k, v in prepared["inputs"].items()}
    encoder_outputs = model.encoder(**inputs)
    token_embeddings = encoder_outputs.last_hidden_state[0]
    
print(f"\n3. Token embeddings shape: {token_embeddings.shape}")

# Compute span representations
span_info = model.compute_span_rep(token_embeddings.unsqueeze(0))
print(f"\n4. Span info keys: {span_info.keys()}")
print(f"   Span rep shape: {span_info['span_rep'].shape}")

# Get schema embeddings
schema_embs = []
for i, schema_tokens in enumerate(transformed["schema_tokens_list"]):
    emb_list = []
    for token in schema_tokens:
        token_id = model.processor.tokenizer.convert_tokens_to_ids(token)
        if token_id != model.processor.tokenizer.unk_token_id:
            emb = model.encoder.embeddings.word_embeddings(
                torch.tensor([token_id], device=device)
            )
            emb_list.append(emb[0])
    if emb_list:
        schema_embs.append(torch.stack(emb_list))

print(f"\n5. Schema embeddings: {len(schema_embs)} schemas")

# Try to get field names
field_names = []
for schema_tokens in transformed["schema_tokens_list"]:
    for i in range(len(schema_tokens) - 1):
        if schema_tokens[i] in ("[E]", "[C]", "[R]"):
            field_names.append(schema_tokens[i + 1])

print(f"\n6. Field names: {field_names}")

# Try _find_valid_spans if we can access it
if hasattr(model, '_find_valid_spans'):
    print("\n7. Model has _find_valid_spans method!")
    # This would need the right parameters
else:
    print("\n7. Model does not have _find_valid_spans as direct method")

print("\n8. Checking for _predict_spans...")
if hasattr(model, '_predict_spans'):
    print("   Model has _predict_spans!")
