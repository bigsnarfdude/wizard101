#!/usr/bin/env python3
"""Minimal GLiNER2 test - single inference."""

from gliner2 import GLiNER2
import time

print("Loading model...")
start = time.time()
model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
print(f"Model loaded in {time.time() - start:.1f}s")

print("\nRunning single inference...")
start = time.time()
result = model.extract_entities(
    "My email is john@example.com",
    ["email address"]
)
print(f"Inference took {(time.time() - start)*1000:.0f}ms")

print(f"\nResult: {result}")
