import os
import argparse
import glob
import cv2
import numpy as np
import pytesseract
from PIL import Image
import img2pdf
import PyPDF2
import tempfile
import sys

# Update this path if Tesseract is installed somewhere else
# If it's already in the PATH environment variable, this might not be needed.
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def crop_page(image):
    """
    Detects the white page against a black background (and potential gloves).
    Crops and applies perspective transform to extract only the page.
    """
    # 1. Grayscale and blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    
    # 2. Thresholding to isolate white page from black background
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Clean up small noise and bright protrusions (like fingers outside the page)
    kernel_open = np.ones((15, 15), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)
    
    # 3. Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image # fallback
        
    # Find the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)
    
    if cv2.contourArea(largest_contour) < (image.shape[0] * image.shape[1] * 0.05):
        return image
        
    # Use convex hull to bridge dark indentations (like dark gloves/hands on the page)
    hull = cv2.convexHull(largest_contour)
    
    # 4. Approximate a polygon (expecting 4 corners)
    peri = cv2.arcLength(hull, True)
    approx = None
    for eps in range(10, 100, 5):
        epsilon = (eps / 1000.0) * peri
        tmp_approx = cv2.approxPolyDP(hull, epsilon, True)
        if len(tmp_approx) == 4:
            approx = tmp_approx
            break
            
    if approx is None:
        rect = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rect)
        pts = np.int32(box)
    else:
        pts = approx.reshape(4, 2)

    # 5. Perspective Transform
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    (tl, tr, br, bl) = rect
    
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    if maxWidth <= 0 or maxHeight <= 0:
        return image
        
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
        
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped

def deskew(image):
    """
    Detects small text slants and rotates the image by the median angle of all text lines to perfectly align them.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    gray_inv = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray_inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    # Connect text horizontally into solid block lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    angles = []
    for c in contours:
        area = cv2.contourArea(c)
        # Ignore very small noise or massive background/hand blocks
        if area < 500 or area > 100000:
            continue
            
        rect = cv2.minAreaRect(c)
        w, h = rect[1]
        if w < 50 or h < 10:
            continue
            
        # Ignore square-ish blocks which don't have a reliable horizontal angle
        if w/h < 2.0 and h/w < 2.0:
            continue
            
        angle = rect[-1]
        
        # Normalize angle to -45 to 45
        if w < h:
            angle += 90
            
        if -15 < angle < 15:
            angles.append(angle)
            
    if not angles:
        return image
        
    median_angle = np.median(angles)
    
    if abs(median_angle) < 0.2:
        return image
        
    print(f"    Straightened slanted text lines by {median_angle:.2f} degrees.")
    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def split_pages(image):
    """
    Checks if the image contains a two-page spread or a partial second page.
    Splits at the spine and discards 'slivers'.
    """
    h, w = image.shape[:2]
    
    # If it's wider than tall (aspect ratio > 1.1), 
    # it is definitely a two-page spread.
    if w > h * 1.1: 
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # We look for the darkest vertical band, which could be anywhere except margins
        start_x = int(w * 0.25)
        end_x = int(w * 0.75)
        center_region = gray[:, start_x:end_x]
        
        vertical_projection = np.sum(center_region, axis=0)
        
        # Remove background illumination trend
        trend_kernel_size = int(w * 0.2)
        trend_kernel = np.ones(trend_kernel_size) / trend_kernel_size
        trend = np.convolve(vertical_projection, trend_kernel, mode='same')
        detrended = vertical_projection - trend
        
        kernel_size = max(5, int(w*0.02))
        kernel = np.ones(kernel_size) / kernel_size
        smoothed_projection = np.convolve(detrended, kernel, mode='same')
        
        min_idx = np.argmin(smoothed_projection)
        split_x = start_x + min_idx
        
        left_page = image[:, :split_x]
        right_page = image[:, split_x:]
        
        results = []
        lh, lw = left_page.shape[:2]
        if lw > lh * 0.35:  # Require width to be at least 35% of height to be a valid page
            results.append(left_page)
            
        rh, rw = right_page.shape[:2]
        if rw > rh * 0.35:
            results.append(right_page)
            
        return results if results else [image]
    else:
        return [image]

def get_ocr_score(image):
    """
    Returns a score based on Tesseract's confidence in reading words.
    Higher score indicates the text is more likely in the correct reading orientation.
    """
    h, w = image.shape[:2]
    new_w = 600
    new_h = int(new_w * h / w)
    resized = cv2.resize(image, (new_w, new_h))
    
    if len(resized.shape) == 2:
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    else:
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
    pil_img = Image.fromarray(rgb_image)
    try:
        # psm 6 assumes a single uniform block of text
        data = pytesseract.image_to_data(pil_img, config='--psm 6', output_type=pytesseract.Output.DICT)
        confs = [int(data['conf'][i]) for i in range(len(data['text'])) if data['text'][i].strip()]
        if not confs: return 0
        return np.mean(confs) * len(confs)
    except Exception as e:
        return 0

def clean_page(image):
    """
    Cleans up old book pages by correcting illumination and applying binarization.
    """
    # 1. Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. Median blur to remove noise (optional, but helps with small specks)
    gray = cv2.medianBlur(gray, 3)

    # 3. Background estimation: use morphological closing or simple division
    # A large morphological closing kernel helps estimate the background (the paper)
    # kernel size depends on the resolution. 300 DPI means large text. Let's use 51.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (51, 51))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    
    # Divide the original image by the background to normalize illumination
    # Multiply by 255 to bring back to 8-bit scale
    normalized = np.float32(gray) / np.float32(background)
    normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)
    normalized = np.uint8(normalized)

    # 4. Adaptive Thresholding to binarize
    # Block size of 31 or 51 is usually good for 300dpi. C is a constant subtracted from the mean.
    binary = cv2.adaptiveThreshold(
        normalized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15
    )
    
    return binary

def create_pdf(image_paths, output_pdf_path):
    """
    Creates a standard PDF from a list of image paths.
    """
    with open(output_pdf_path, "wb") as f:
        f.write(img2pdf.convert(image_paths))

def create_ocr_pdf(image_paths, output_pdf_path):
    """
    Creates a searchable PDF using Tesseract OCR.
    """
    pdf_writer = PyPDF2.PdfWriter()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for idx, img_path in enumerate(image_paths):
            print(f"Running OCR on {os.path.basename(img_path)}...")
            try:
                # Get specific page pdf bytes using pytesseract
                pdf_bytes = pytesseract.image_to_pdf_or_hocr(img_path, extension='pdf')
                
                # Save single page to temp file
                temp_pdf_path = os.path.join(temp_dir, f"page_{idx}.pdf")
                with open(temp_pdf_path, "wb") as f:
                    f.write(pdf_bytes)
                
                # Read it back with PyPDF2 and append
                pdf_reader = PyPDF2.PdfReader(temp_pdf_path)
                pdf_writer.add_page(pdf_reader.pages[0])
            except Exception as e:
                print(f"Error processing {img_path} with OCR: {e}")
                
        with open(output_pdf_path, "wb") as f:
            pdf_writer.write(f)

def main():
    parser = argparse.ArgumentParser(description="Digitize book pages into a clean, combined PDF.")
    parser.add_argument("input_dir", help="Directory containing raw photos of pages")
    parser.add_argument("output_dir", help="Directory to save the processed files and final PDFs")
    parser.add_argument("--ocr", action="store_true", help="Generate an additional searchable PDF using OCR")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        sys.exit(1)
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Supported image extensions
    extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff")
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(args.input_dir, ext)))
        # case insensitive match
        image_files.extend(glob.glob(os.path.join(args.input_dir, ext.upper())))
        
    # Remove duplicates and sort (so pages are in order)
    image_files = sorted(list(set(image_files)))
    
    if not image_files:
        print(f"No images found in '{args.input_dir}'.")
        sys.exit(1)
        
    print(f"Found {len(image_files)} images. Starting processing...")
    
    processed_image_paths = []
    
    for img_path in image_files:
        print(f"Processing: {os.path.basename(img_path)}")
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue
            
        # 1. Crop page from black background
        cropped_img = crop_page(img)
        
        # 2. Check if it's landscape (w > h * 1.1)
        # It could be a 2-page spread (upright text) or a single page taken sideways (sideways text)
        h, w = cropped_img.shape[:2]
        if w > h * 1.1:
            print(f"  Landscape aspect ratio detected. Voting orientation...")
            temp_clean = clean_page(cropped_img)
            
            score_0 = get_ocr_score(temp_clean)
            
            clean_90 = cv2.rotate(temp_clean, cv2.ROTATE_90_CLOCKWISE)
            score_90 = get_ocr_score(clean_90)
            
            clean_270 = cv2.rotate(temp_clean, cv2.ROTATE_90_COUNTERCLOCKWISE)
            score_270 = get_ocr_score(clean_270)
            
            # If 90 or 270 is significantly higher than 0, it's a sideways single page.
            if score_90 > score_0 * 1.5 and score_90 >= score_270:
                print(f"  Rotated spread 90 degrees via OCR voting (Score 90: {score_90:.0f} vs 0: {score_0:.0f}).")
                cropped_img = cv2.rotate(cropped_img, cv2.ROTATE_90_CLOCKWISE)
            elif score_270 > score_0 * 1.5 and score_270 > score_90:
                print(f"  Rotated spread 270 degrees via OCR voting (Score 270: {score_270:.0f} vs 0: {score_0:.0f}).")
                cropped_img = cv2.rotate(cropped_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                print(f"  Kept at 0 degrees. Likely a 2-page spread.")
        
        # 3. Split pages if applicable (will only split if STILL w > h * 1.1)
        pages = split_pages(cropped_img)
        
        # Process each resulting page
        for sub_idx, page_img in enumerate(pages):
            page_label = chr(65 + sub_idx) if len(pages) > 1 else "" # e.g. "A" or "B"
            if page_label:
                print(f"  -> Processing sub-page {page_label}")
                
            # 4. Universal OCR Voting for Final Orientation
            # To handle square pages or pages where text is rotated despite portrait geometry.
            temp_clean = clean_page(page_img)
            score_0 = get_ocr_score(temp_clean)
            
            clean_90 = cv2.rotate(temp_clean, cv2.ROTATE_90_CLOCKWISE)
            score_90 = get_ocr_score(clean_90)
            
            clean_180 = cv2.rotate(temp_clean, cv2.ROTATE_180)
            score_180 = get_ocr_score(clean_180)
            
            clean_270 = cv2.rotate(temp_clean, cv2.ROTATE_90_COUNTERCLOCKWISE)
            score_270 = get_ocr_score(clean_270)
            
            scores = {0: score_0, 90: score_90, 180: score_180, 270: score_270}
            best_angle = max(scores, key=scores.get)
            best_score = scores[best_angle]
            
            # We enforce a rotation if the best angle strongly beats the current 0 angle
            if best_angle != 0 and best_score > (score_0 * 1.5):
                print(f"    Rotated sub-page {best_angle} degrees via universal OCR voting (Score {best_angle}: {best_score:.0f} vs 0: {score_0:.0f}).")
                if best_angle == 90:
                    page_img = cv2.rotate(page_img, cv2.ROTATE_90_CLOCKWISE)
                elif best_angle == 180:
                    page_img = cv2.rotate(page_img, cv2.ROTATE_180)
                elif best_angle == 270:
                    page_img = cv2.rotate(page_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    
            # 5. Deskew small slants (e.g. 1 or 2 degrees)
            page_img = deskew(page_img)
                
            # 6. Final Cleaning
            final_clean = clean_page(page_img)
            
            # 7. Save processed PNG
            out_filename = f"page_{len(processed_image_paths)+1:04d}.png"
            out_path = os.path.join(args.output_dir, out_filename)
            cv2.imwrite(out_path, final_clean)
            processed_image_paths.append(out_path)
        
    if not processed_image_paths:
        print("No pages were successfully processed.")
        sys.exit(1)
        
    # Generate Output PDF (No OCR)
    pdf_no_ocr_path = os.path.join(args.output_dir, "output_no_ocr.pdf")
    print(f"Generating clean PDF without OCR: {pdf_no_ocr_path}")
    create_pdf(processed_image_paths, pdf_no_ocr_path)
    print("Done generating standard PDF.")
    
    # Generate OCR PDF if requested
    if args.ocr:
        pdf_ocr_path = os.path.join(args.output_dir, "output_ocr.pdf")
        print(f"\nGenerating searchable PDF with OCR: {pdf_ocr_path}")
        create_ocr_pdf(processed_image_paths, pdf_ocr_path)
        print("Done generating OCR PDF.")
        
    print("All tasks completed successfully!")

if __name__ == "__main__":
    main()
