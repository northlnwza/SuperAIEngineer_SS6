"""
OCR Function - Extract Tables from Images using Typhoon
Focused on table extraction only, minimal dependencies
"""

from typhoon.ocr import Reader
from PIL import Image
import cv2
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
import re


def OCR(image_path: str) -> Optional[pd.DataFrame]:
    """
    Extract table from image using Typhoon OCR
    
    Parameters:
    -----------
    image_path : str
        Path to image file
    
    Returns:
    --------
    pd.DataFrame or None
        Extracted table as DataFrame, or None if no table found
    """
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Convert to RGB for Typhoon
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Initialize Typhoon OCR reader
    reader = Reader()
    
    # Extract text using Typhoon
    results = reader.readtext(rgb_img)
    
    # Parse results into table
    text = _extract_text_from_results(results)
    df = _parse_table(text)
    
    return df


def _extract_text_from_results(results: List) -> str:
    """
    Extract text from Typhoon OCR results
    
    Parameters:
    -----------
    results : List
        Typhoon readtext results
    
    Returns:
    --------
    str
        Concatenated text lines
    """
    
    # Group by vertical position (rows)
    rows = {}
    for (bbox, text, confidence) in results:
        if confidence < 0.3:  # Skip low confidence
            continue
        
        # Get y-coordinate of first point (row position)
        y_pos = int(bbox[0][1])
        
        # Group by row (within 10 pixels)
        row_key = (y_pos // 15) * 15
        
        if row_key not in rows:
            rows[row_key] = []
        
        rows[row_key].append((bbox[0][0], text))
    
    # Sort by row, then by x-coordinate
    sorted_rows = []
    for row_key in sorted(rows.keys()):
        sorted_texts = sorted(rows[row_key], key=lambda x: x[0])
        line_text = ' '.join([text for _, text in sorted_texts])
        sorted_rows.append(line_text)
    
    return '\n'.join(sorted_rows)
    """
    Parse OCR text into table structure
    
    Parameters:
    -----------
    text : str
        Raw OCR text
    
    Returns:
    --------
    pd.DataFrame or None
        Structured table
    """
    
    lines = text.strip().split('\n')
    
    # Remove empty lines and non-table lines
    lines = [line.strip() for line in lines if line.strip()]
    
    if not lines:
        return None
    
    # Detect separator pattern (spaces, tabs, pipes)
    table_rows = []
    
    for line in lines:
        # Skip lines that are too short
        if len(line) < 5:
            continue
        
        # Split by multiple spaces or tabs
        row = re.split(r'\s{2,}|\t|\|', line)
        row = [cell.strip() for cell in row if cell.strip()]
        
        # Filter out lines with too many or too few columns
        if 2 <= len(row) <= 20:
            table_rows.append(row)
    
    if not table_rows:
        return None
    
    # Detect and align columns
    df = _align_columns(table_rows)
    
    return df


def _align_columns(rows: List[List[str]]) -> pd.DataFrame:
    """
    Align columns and create DataFrame
    
    Parameters:
    -----------
    rows : List[List[str]]
        List of parsed rows
    
    Returns:
    --------
    pd.DataFrame
        Aligned table
    """
    
    if not rows:
        return None
    
    # Find max column count
    max_cols = max(len(row) for row in rows)
    
    # Pad rows to have same column count
    aligned_rows = []
    for row in rows:
        while len(row) < max_cols:
            row.append('')
        aligned_rows.append(row[:max_cols])
    
    # First row as header
    header = aligned_rows[0]
    data = aligned_rows[1:]
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=header)
    
    # Remove rows with all empty values
    df = df.dropna(how='all')
    
    # Reset index
    df = df.reset_index(drop=True)
    
    return df


def OCR_batch(image_paths: List[str]) -> List[pd.DataFrame]:
    """
    Extract tables from multiple images
    
    Parameters:
    -----------
    image_paths : List[str]
        List of image file paths
    
    Returns:
    --------
    List[pd.DataFrame]
        List of extracted tables
    """
    
    results = []
    for path in image_paths:
        try:
            df = OCR(path)
            if df is not None:
                results.append(df)
                print(f"✓ Extracted: {path} ({df.shape[0]} rows × {df.shape[1]} cols)")
            else:
                print(f"⚠ No table found: {path}")
        except Exception as e:
            print(f"✗ Error processing {path}: {str(e)}")
    
    return results


def OCR_to_csv(image_path: str, output_path: str) -> bool:
    """
    Extract table from image and save as CSV
    
    Parameters:
    -----------
    image_path : str
        Input image path
    output_path : str
        Output CSV path
    
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    
    try:
        df = OCR(image_path)
        if df is None:
            print(f"No table found in: {image_path}")
            return False
        
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"✓ Saved: {output_path}")
        return True
    
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False


# Example usage
if __name__ == "__main__":
    # Example: OCR_df = OCR("path/to/image.jpg")
    print("OCR Module loaded successfully")
    print("\nUsage:")
    print("  df = OCR('image.jpg')")
    print("  dfs = OCR_batch(['image1.jpg', 'image2.jpg'])")
    print("  OCR_to_csv('image.jpg', 'output.csv')")
