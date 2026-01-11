"""
ROI Visualization Tool

Creates visual markers on frames to verify OCR region positioning.
Saves both the full frame with ROI marked and the cropped region.
"""

import sys
from pathlib import Path
from PIL import Image, ImageDraw

sys.path.insert(0, '.')

from src.utils import load_config

# Load config
config = load_config('./config/settings.yaml')
ocr_config = config.get("ocr", {})
ribbon_roi = ocr_config.get("ribbon_roi", {})

# Get specific frame
frame_path = Path("output/debug/kompastv_20260111_145305/frames/frame_000071.png")

if not frame_path.exists():
    print(f"Frame not found: {frame_path}")
    sys.exit(1)

print(f"Using frame: {frame_path.name}")

# Load frame
frame = Image.open(frame_path)
print(f"Frame size: {frame.size}")
print(f"ROI config: {ribbon_roi}")

# Create copy for drawing
frame_with_roi = frame.copy()
draw = ImageDraw.Draw(frame_with_roi)

# Extract ROI coordinates
x1 = ribbon_roi.get("x1", 0)
y1 = ribbon_roi.get("y1", 0)
x2 = ribbon_roi.get("x2", frame.width)
y2 = ribbon_roi.get("y2", frame.height)

print(f"\nROI coordinates: ({x1}, {y1}) to ({x2}, {y2})")
print(f"ROI size: {x2-x1}x{y2-y1}")

# Draw red rectangle on ROI
draw.rectangle([x1, y1, x2, y2], outline="red", width=5)

# Draw corner markers for better visibility
corner_size = 20
draw.line([x1, y1, x1 + corner_size, y1], fill="yellow", width=3)
draw.line([x1, y1, x1, y1 + corner_size], fill="yellow", width=3)

draw.line([x2, y1, x2 - corner_size, y1], fill="yellow", width=3)
draw.line([x2, y1, x2, y1 + corner_size], fill="yellow", width=3)

draw.line([x1, y2, x1 + corner_size, y2], fill="yellow", width=3)
draw.line([x1, y2, x1, y2 - corner_size], fill="yellow", width=3)

draw.line([x2, y2, x2 - corner_size, y2], fill="yellow", width=3)
draw.line([x2, y2, x2, y2 - corner_size], fill="yellow", width=3)

# Save frame with ROI marked
output_marked = Path("roi_visualization_marked.png")
frame_with_roi.save(output_marked)
print(f"\nSaved marked frame: {output_marked}")

# Crop and save ROI region
roi_crop = frame.crop((x1, y1, x2, y2))
output_crop = Path("roi_visualization_crop.png")
roi_crop.save(output_crop)
print(f"Saved cropped ROI: {output_crop}")

print("\nCHECK FILES:")
print("1. roi_visualization_marked.png - Full frame with red rectangle showing ROI")
print("2. roi_visualization_crop.png - Only the cropped region")
print("\nIf ribbon text is NOT in the red box, update ribbon_roi in config/settings.yaml")
