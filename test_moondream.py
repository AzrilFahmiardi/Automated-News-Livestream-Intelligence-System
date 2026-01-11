#!/usr/bin/env python3
"""
Test script untuk validasi Moondream VLM installation dan basic functionality.
"""

import sys
from pathlib import Path

print("=" * 60)
print("MOONDREAM VLM TEST")
print("=" * 60)

# Test 1: Import dependencies
print("\n[1/5] Testing imports...")
try:
    import torch
    print(f"  ✓ PyTorch {torch.__version__}")
    print(f"  ✓ CUDA available: {torch.cuda.is_available()}")
    print(f"  ✓ Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
except ImportError as e:
    print(f"  ✗ PyTorch import failed: {e}")
    sys.exit(1)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"  ✓ Transformers imported")
except ImportError as e:
    print(f"  ✗ Transformers import failed: {e}")
    sys.exit(1)

try:
    from PIL import Image
    print(f"  ✓ PIL imported")
except ImportError as e:
    print(f"  ✗ PIL import failed: {e}")
    sys.exit(1)

# Test 2: Load Moondream model
print("\n[2/5] Loading Moondream model...")
print("  (This will download ~4GB on first run, please wait...)")

try:
    model_id = "vikhyatk/moondream2"
    revision = "2025-06-21"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    
    print(f"  ✓ Model loaded: {model_id}")
    print(f"  ✓ Revision: {revision}")
    print(f"  ✓ Device: {model.device}")
    
except Exception as e:
    print(f"  ✗ Model loading failed: {e}")
    sys.exit(1)

# Test 3: Create test image
print("\n[3/5] Creating test image...")
try:
    # Create simple test image (red square with white text area)
    test_image = Image.new('RGB', (640, 360), color='red')
    print(f"  ✓ Test image created: {test_image.size}")
except Exception as e:
    print(f"  ✗ Image creation failed: {e}")
    sys.exit(1)

# Test 4: Encode image
print("\n[4/5] Encoding image...")
try:
    enc_image = model.encode_image(test_image)
    print(f"  ✓ Image encoded successfully")
    print(f"  ✓ Encoding shape: {enc_image.shape}")
except Exception as e:
    print(f"  ✗ Image encoding failed: {e}")
    sys.exit(1)

# Test 5: Query model
print("\n[5/5] Testing model query...")
try:
    question = "What color is this image?"
    answer = model.query(enc_image, question, tokenizer)
    
    print(f"  ✓ Query successful")
    print(f"  ✓ Question: {question}")
    print(f"  ✓ Answer: {answer['answer']}")
    
except Exception as e:
    print(f"  ✗ Query failed: {e}")
    sys.exit(1)

# Summary
print("\n" + "=" * 60)
print("TEST SUMMARY")
print("=" * 60)
print("✓ All tests passed!")
print("\nMoondream VLM is ready for use.")
print("You can now run: python main.py --debug --channel KompasTV")
print("=" * 60)
