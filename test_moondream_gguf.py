"""
Test Moondream GGUF Standalone

Test script untuk mencoba Moondream GGUF dengan gambar sample.
"""

import sys
import time
from pathlib import Path
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler

# Paths
TEXT_MODEL = "./models/moondream2-text-model-f16_ct-vicuna.gguf"
MMPROJ = "./models/moondream2-mmproj-f16-20250414.gguf"
TEST_IMAGE = "output/debug/kompastv_20260112_004403/frames/frame_000001.png"  # Screenshot dari user

# Prompt yang kompleks
RIBBON_PROMPT = """Look at the lower part of this news broadcast image. 
Find the MAIN ribbon banner (lower-third graphic) - this is usually a colored bar with important text.
IGNORE any running text or news ticker (text that scrolls horizontally, often shows time/date).
The main ribbon can be:
1. Breaking news headline (e.g., "BREAKING NEWS | [headline]")
2. Speaker identification (e.g., "[Name] | [Title/Role]")
3. News topic label (e.g., "[Topic] | [Description]")

Extract ONLY the text from this MAIN ribbon banner. If there are multiple lines in the ribbon, include all of them.
Do NOT include running text, time, date, or scrolling tickers."""

# Prompt yang simple
SIMPLE_PROMPT = "What text is on the blue banner at the bottom?"

# Prompt super simple
SUPER_SIMPLE = "Read the text at the bottom."


def test_moondream(image_path: str, prompt: str, prompt_name: str):
    """Test Moondream dengan prompt tertentu."""
    print(f"\n{'='*80}")
    print(f"Testing: {prompt_name}")
    print(f"{'='*80}")
    print(f"Prompt: {prompt[:100]}...")
    print(f"Image: {image_path}")
    
    try:
        # Load model
        print("\n[1/4] Loading vision projector...")
        start = time.time()
        chat_handler = Llava15ChatHandler(clip_model_path=MMPROJ)
        print(f"   ✓ Loaded in {time.time() - start:.1f}s")
        
        print("\n[2/4] Loading text model...")
        start = time.time()
        model = Llama(
            model_path=TEXT_MODEL,
            chat_handler=chat_handler,
            n_ctx=2048,
            n_threads=4,
            n_gpu_layers=0,
            logits_all=True,
            verbose=False
        )
        print(f"   ✓ Loaded in {time.time() - start:.1f}s")
        
        # Query
        print("\n[3/4] Processing image...")
        start = time.time()
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"file://{image_path}"}},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        response = model.create_chat_completion(
            messages=messages,
            max_tokens=256,
            temperature=0.1,
            top_p=0.9
        )
        
        processing_time = time.time() - start
        print(f"   ✓ Processed in {processing_time:.1f}s")
        
        # Result
        print("\n[4/4] Result:")
        print("-" * 80)
        answer = response["choices"][0]["message"]["content"]
        print(answer)
        print("-" * 80)
        print(f"\nTotal time: {processing_time:.1f}s")
        print(f"Answer length: {len(answer)} chars, {len(answer.split())} words")
        
        # Cleanup
        del model
        del chat_handler
        
        return answer
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main test function."""
    print("="*80)
    print("MOONDREAM GGUF STANDALONE TEST")
    print("="*80)
    
    # Check files
    if not Path(TEXT_MODEL).exists():
        print(f"❌ Text model not found: {TEXT_MODEL}")
        sys.exit(1)
    if not Path(MMPROJ).exists():
        print(f"❌ MMProj not found: {MMPROJ}")
        sys.exit(1)
    if not Path(TEST_IMAGE).exists():
        print(f"❌ Test image not found: {TEST_IMAGE}")
        print(f"   Please save the screenshot as: {TEST_IMAGE}")
        sys.exit(1)
    
    print(f"✓ Text model: {TEXT_MODEL}")
    print(f"✓ MMProj: {MMPROJ}")
    print(f"✓ Test image: {TEST_IMAGE}")
    
    # Get absolute path
    abs_image = str(Path(TEST_IMAGE).resolve())
    
    # Test dengan berbagai prompt
    tests = [
        ("Super Simple", SUPER_SIMPLE),
        ("Simple", SIMPLE_PROMPT),
        ("Complex", RIBBON_PROMPT),
    ]
    
    results = {}
    for name, prompt in tests:
        result = test_moondream(abs_image, prompt, name)
        results[name] = result
        input("\nPress Enter untuk lanjut ke test berikutnya...")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  {result[:100] if result else 'FAILED'}...")


if __name__ == "__main__":
    main()
