import cv2
import numpy as np
import pytesseract
import os
import time
import requests

# 🔹 Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def download_image(url, save_path):
    """Download image from URL."""
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(response.content)
        print(f"✅ Image downloaded: {save_path}")
        return True
    else:
        print("❌ Failed to download image:", response.status_code)
        return False

def remove_bubbles_edges(panel, margin=10):
    """Crop white speech bubble parts from top and bottom of the panel."""
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)  # white = 255

    h, w = mask.shape

    # 🔹 Top trim
    top = 0
    for i in range(h // 3):  # scan only top third
        white_ratio = np.sum(mask[i, :] == 255) / w
        if white_ratio > 0.7:  # row mostly white
            top = i
        else:
            break
    top = min(top + margin, h - 1)

    # 🔹 Bottom trim
    bottom = h
    for i in range(h - 1, 2 * h // 3, -1):  # scan only bottom third
        white_ratio = np.sum(mask[i, :] == 255) / w
        if white_ratio > 0.7:
            bottom = i
        else:
            break
    bottom = max(bottom - margin, top + 1)

    return panel[top:bottom, :]

def is_useful_panel(panel):
    """Check if panel should be kept."""
    h, w = panel.shape[:2]
    if h < 100 or w < 100:
        return False

    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    non_white_ratio = np.sum(mask > 0) / (h * w)
    if non_white_ratio < 0.15:
        return False

    text = pytesseract.image_to_string(panel).strip()
    if text:
        color_std = np.std(panel.reshape(-1, 3), axis=0).mean()
        if color_std < 35:
            return False

    return True

def split_and_save(img, output_dir, page_idx):
    """Split comic into panels and save useful ones."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    row_sum = np.sum(binary, axis=1)
    gap_threshold = 0.95 * np.max(row_sum)
    gaps = np.where(row_sum > gap_threshold)[0]

    splits = []
    prev = -2
    for y in gaps:
        if y != prev + 1:
            splits.append(y)
        prev = y
    splits.append(img.shape[0])

    start = 0
    count_good = 0

    for cut in splits:
        if cut - start > 80:
            panel = img[start:cut, :]

            gray_panel = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_panel, 240, 255, cv2.THRESH_BINARY_INV)
            coords = cv2.findNonZero(mask)

            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                cropped = panel[y:y+h, x:x+w]

                # 🔹 Remove bubble edges
                cropped = remove_bubbles_edges(cropped)

                if is_useful_panel(cropped):
                    filename = f"page{page_idx}_panel{count_good+1}.jpg"
                    cv2.imwrite(os.path.join(output_dir, filename), cropped)
                    count_good += 1

        start = cut

    print(f"✅ Page {page_idx}: Saved {count_good} clean panels")

def process_images(urls):
    """Process multiple image URLs into one folder."""
    base_dir = "AUTO-SS"
    os.makedirs(base_dir, exist_ok=True)

    input_dir = os.path.join(base_dir, "input")
    os.makedirs(input_dir, exist_ok=True)

    panels_dir = os.path.join(base_dir, "panels")
    os.makedirs(panels_dir, exist_ok=True)

    for idx, url in enumerate(urls, start=1):
        img_path = os.path.join(input_dir, f"comic_{idx}.webp")

        # Download image
        if download_image(url, img_path):
            img = cv2.imread(img_path)

            if img is None:
                print(f"❌ Error: OpenCV could not read image {idx}")
            else:
                split_and_save(img, panels_dir, page_idx=idx)

if __name__ == "__main__":
    # 🔹 List of comic page URLs
    urls = [
        "https://en-thunderscans.com/wp-content/uploads/2025/09/2025-09-20-04-58-17-1758387497373.webp",
        "https://en-thunderscans.com/wp-content/uploads/2025/09/2025-09-20-04-58-31-1758387511318.webp",
        "https://en-thunderscans.com/wp-content/uploads/2025/09/2025-09-20-04-58-34-1758387514123.webp",
        "https://en-thunderscans.com/wp-content/uploads/2025/09/2025-09-20-04-58-36-1758387516450.webp",
        "https://en-thunderscans.com/wp-content/uploads/2025/09/2025-09-20-04-58-38-1758387518528.webp"
        # add more links here...
    ]

    process_images(urls)
