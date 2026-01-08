# ==========================================
# FILE: sequential.py
# ==========================================

import cv2
import numpy as np
import os
import time
import argparse

INPUT_ROOT_DIR = "food-101"
OUTPUT_ROOT_DIR = "food-101_processed_sequential"
SELECTED_CATEGORIES = ["pizza", "sushi", "hamburger", "samosa"]
DEFAULT_LIMIT = 1000

def process_single_image(task):
    """
    Process a single image through all five filters sequentially.
    """
    image_id, input_path, output_path = task
    
    try:
        print(f"[Img {image_id}] START: Loading {os.path.basename(input_path)}")
        
        # Load image
        img = cv2.imread(input_path)
        if img is None:
            print(f"[Img {image_id}] ERROR: Failed to load image")
            return (False, f"Failed to load {input_path}")
        
        # Filter 1: Grayscale Conversion
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(f"[Img {image_id}] Step 1: Grayscale - Shape: {gray.shape}")
        
        # Filter 2: Gaussian Blur (3x3 kernel)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        print(f"[Img {image_id}] Step 2: Gaussian Blur - Kernel: (3,3)")
        
        # Filter 3: Edge Detection (Sobel)
        sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
        edges = cv2.magnitude(sobelx, sobely)
        edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX)
        edges = edges.astype(np.uint8)
        print(f"[Img {image_id}] Step 3: Sobel Edges - Edges computed")
        
        # Filter 4: Image Sharpening
        sharpened = cv2.addWeighted(blur, 1.5, edges, -0.5, 0)
        print(f"[Img {image_id}] Step 4: Sharpening - Weighted add applied")
        
        # Filter 5: Brightness Adjustment
        final_img = cv2.convertScaleAbs(sharpened, alpha=1.0, beta=50)
        print(f"[Img {image_id}] Step 5: Brightness - Beta +50")
        
        # Save output
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, final_img)
        print(f"[Img {image_id}] COMPLETE: Saved to {os.path.basename(output_path)}")
        
        return (True, os.path.basename(input_path))
        
    except Exception as e:
        print(f"[Img {image_id}] EXCEPTION: {str(e)}")
        return (False, f"Error processing {input_path}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Sequential Image Processing")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                       help="Max images per category")
    args = parser.parse_args()

    print("=" * 60)
    print("SEQUENTIAL PROCESSING")
    print("=" * 60)
    print(f"Categories: {SELECTED_CATEGORIES}")
    print(f"Images per category: {args.limit}")
    print("=" * 60)

    # Collect all image tasks
    tasks = []
    image_id = 0
    for category in SELECTED_CATEGORIES:
        src_dir = os.path.join(INPUT_ROOT_DIR, category)
        dst_dir = os.path.join(OUTPUT_ROOT_DIR, category)

        if not os.path.exists(src_dir):
            print(f"Warning: Missing directory {src_dir}")
            continue

        files = [f for f in os.listdir(src_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        
        if args.limit:
            files = files[:args.limit]

        for f in files:
            image_id += 1
            tasks.append((image_id, os.path.join(src_dir, f),
                         os.path.join(dst_dir, f)))

    if not tasks:
        print("No images found. Exiting.")
        return

    print(f"\nTotal images: {len(tasks)}")
    print("\nProcessing...\n")

    start_time = time.time()
    
    # Process images sequentially
    results = []
    for task in tasks:
        result = process_single_image(task)
        results.append(result)
    
    end_time = time.time()

    # Collect errors
    errors = [r[1] for r in results if not r[0]]
    successful = sum(1 for r in results if r[0])
    
    print("\n=============================================")
    print(" Sequential Processing Pipeline")
    print("=============================================\n")
    print(f"Processing mode: Sequential (single-threaded)")
    print(f"Categories: {SELECTED_CATEGORIES}")
    print("\n=============================================")
    print(" FINAL RESULTS")
    print("=============================================")
    print(f"Images processed : {successful}")
    print(f"Execution time   : {end_time - start_time:.4f} seconds")
    print(f"Errors           : {len(errors)}")
    print("=============================================")

if __name__ == "__main__":
    main()