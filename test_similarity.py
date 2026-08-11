import os
import sys

# Ensure project root is in python path
sys.path.insert(0, os.path.dirname(__file__))

from src.corrections import save_correction, find_similar_override

def test_visual_similarity():
    print("Testing visual similarity learning layer...")
    
    # 1. We need a dummy image
    test_img = "test_img.jpg"
    if not os.path.exists(test_img):
        from PIL import Image
        img = Image.new('RGB', (224, 224), color = 'red')
        img.save(test_img)
    
    # 2. Save a correction
    print("Saving correction...")
    save_correction(
        ftir_no="FTIR_TEST_01",
        correction_type="model2_wrong_sbpr",
        original_prediction="SBIN201210B00011",
        correct_label="SBIN202310B06811",
        user_reason="Test manual override with red image",
        image_paths=[test_img],
        metadata={"Subject (English)": "Rust found on door"}
    )
    
    # 3. Search for a similar override using the same image but DIFFERENT text
    print("Searching for similar override...")
    match = find_similar_override(
        metadata={"Subject (English)": "Completely unrelated text"},
        image_paths=[test_img],
        text_threshold=0.6,
        image_threshold=0.90
    )
    
    if match:
        print(f"SUCCESS: Found match! Correct label is: {match['correct_label']}")
    else:
        print("FAILED: No match found.")
        
    # Clean up
    if os.path.exists(test_img):
        os.remove(test_img)

if __name__ == "__main__":
    test_visual_similarity()
