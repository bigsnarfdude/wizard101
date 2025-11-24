from gliner2 import GLiNER2

model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")

text = "Contact John Smith at john.smith@company.com or 555-123-4567"
pii_types = ["person name", "email address", "phone number"]

print("=" * 70)
print("Testing GLiNER2 raw output format")
print("=" * 70)

# Test with format_results=True (default)
print("\n1. format_results=True:")
formatted = model.extract_entities(text, pii_types, format_results=True)
print(f"Type: {type(formatted)}")
print(f"Content: {formatted}")

# Test with format_results=False
print("\n2. format_results=False:")
raw = model.extract_entities(text, pii_types, format_results=False)
print(f"Type: {type(raw)}")
print(f"Content: {raw}")

# Inspect structure
if "entities" in raw:
    print(f"\n3. raw['entities'] type: {type(raw['entities'])}")
    print(f"   raw['entities'] content: {raw['entities']}")
    
    if isinstance(raw['entities'], list) and len(raw['entities']) > 0:
        print(f"\n4. raw['entities'][0] type: {type(raw['entities'][0])}")
        print(f"   raw['entities'][0] content: {raw['entities'][0]}")
        
        for key, value in raw['entities'][0].items():
            print(f"\n5. Entity type '{key}':")
            print(f"   Type: {type(value)}")
            print(f"   Content: {value}")
            if isinstance(value, list) and len(value) > 0:
                print(f"   First item type: {type(value[0])}")
                print(f"   First item: {value[0]}")
