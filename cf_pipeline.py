import cv2
import numpy as np
import os
import time
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

# ==========================================
# CONFIGURATION
# ==========================================

INPUT_ROOT_DIR = "food-101"
OUTPUT_ROOT_DIR = "food-101_processed_cf"

SELECTED_CATEGORIES = ["pizza", "sushi", "hamburger", "samosa"]
DEFAULT_LIMIT = 5 

# ==========================================
# WORKER FUNCTION
# ==========================================
def process_single_image(args):
    """
    Process a single image with verbose logging.
    """
    task, shared_stats, lock, worker_id = args
    image_id, input_path, output_path = task
    
    worker_label = f"Process {worker_id}"

    # Helper function to print safely
    def log_step(step_name, detail=""):
        if lock:  # Only use lock if it exists
            with lock:
                print(f"[{worker_label}] [Img {image_id}] {step_name:<25} : {detail}")
        else:  # Sequential - no lock needed
            print(f"[{worker_label}] [Img {image_id}] {step_name:<25} : {detail}")
    
    try:
        log_step("START", f"Loading {os.path.basename(input_path)}")

        img = cv2.imread(input_path)
        if img is None:
            log_step("ERROR", "Failed to load image")
            return (False, f"Failed to load {input_path}", worker_id)

        # STEP 1: Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        log_step("Step 1: Grayscale", f"Shape: {gray.shape}")

        # STEP 2: Gaussian Blur
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        log_step("Step 2: Gaussian Blur", "Kernel: (3,3)")

        # STEP 3: Sobel Edge Detection
        sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
        edges = cv2.magnitude(sobelx, sobely)
        edges = np.uint8(edges)
        log_step("Step 3: Sobel Edges", "Edges computed")

        # STEP 4: Sharpening
        sharpened = cv2.addWeighted(blur, 1.5, edges, -0.5, 0)
        log_step("Step 4: Sharpening", "Weighted add applied")

        # STEP 5: Brightness Adjustment
        final_img = cv2.convertScaleAbs(sharpened, alpha=1.0, beta=50)
        log_step("Step 5: Brightness", "Beta +50")

        # Save output
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, final_img)
        
        # Update global stats
        if lock:  # Only lock if parallel
            with lock:
                shared_stats["images_processed"] += 1
                current_total = shared_stats["images_processed"]
        else:  # Sequential - no lock needed
            shared_stats["images_processed"] += 1
            current_total = shared_stats["images_processed"]
        
        log_step("COMPLETE", f"Total processed so far: {current_total}")

        return (True, os.path.basename(input_path), worker_id)

    except Exception as e:
        log_step("EXCEPTION", str(e))
        return (False, f"Error processing {input_path}: {str(e)}", worker_id)

# ==========================================
# MAIN DRIVER
# ==========================================
def main():
    parser = argparse.ArgumentParser(
        description="Concurrent Futures Image Processing Pipeline (Verbose Mode)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of worker processes (0 = CPU cores, 1 = sequential)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Max images per category"
    )
    args = parser.parse_args()

    # Determine worker count
    max_cores = multiprocessing.cpu_count()
    num_workers = args.workers if args.workers > 0 else max_cores

    # ==========================================
    # TASK COLLECTION
    # ==========================================
    tasks = []
    image_id = 0

    for category in SELECTED_CATEGORIES:
        src_dir = os.path.join(INPUT_ROOT_DIR, category)
        dst_dir = os.path.join(OUTPUT_ROOT_DIR, category)

        if not os.path.exists(src_dir):
            print(f"Warning: Missing directory {src_dir}")
            continue

        files = [
            f for f in os.listdir(src_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if args.limit:
            files = files[:args.limit]

        for f in files:
            image_id += 1
            tasks.append((
                image_id,
                os.path.join(src_dir, f),
                os.path.join(dst_dir, f)
            ))

    if not tasks:
        print("No images found. Exiting.")
        return

    print(f"Total images queued: {len(tasks)}")
    print("Starting Execution...\n")

    # ==========================================
    # EXECUTION MODE SELECTION
    # ==========================================
    
    if num_workers == 1:
        # ============================================
        # TRUE SEQUENTIAL EXECUTION
        # ============================================
        processed = 0
        errors = []
        start_time = time.time()
        
        # No Manager, no locks needed for sequential
        shared_stats = {"images_processed": 0}
        
        # Process tasks sequentially
        for task in tasks:
            result = process_single_image((task, shared_stats, None, 1))
            status, message, pid = result
            if status:
                processed += 1
            else:
                errors.append(message)

        end_time = time.time()

    else:
        # ============================================
        # PARALLEL EXECUTION
        # ============================================
        manager = multiprocessing.Manager()
        shared_stats = manager.dict()
        shared_stats["images_processed"] = 0
        lock = manager.Lock()

        processed = 0
        errors = []
        start_time = time.time()

        # Create task arguments with round-robin worker IDs
        task_args = []
        for idx, task in enumerate(tasks):
            worker_id = (idx % num_workers) + 1
            task_args.append((task, shared_stats, lock, worker_id))

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_single_image, args) for args in task_args]
            for future in as_completed(futures):
                result = future.result()
                status, message, pid = result 
                
                if status:
                    processed += 1
                else:
                    errors.append(message)
                    print(f"[Process {pid}] FAILED: {message}")

        end_time = time.time()

    # ==========================================
    # FINAL REPORT (same for both)
    # ==========================================
    print("\n=============================================")
    print(" Concurrent.Futures Pipeline")
    print("=============================================\n")

    print(f"Using {num_workers} worker processes")
    print(f"Categories: {SELECTED_CATEGORIES}")

    print("\n=============================================")
    print(" FINAL RESULTS")
    print("=============================================")
    print(f"Images processed : {shared_stats['images_processed']}")
    print(f"Execution time   : {end_time - start_time:.4f} seconds")
    print(f"Errors           : {len(errors)}")
    print("=============================================")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()