"""
PDF text and image extraction utilities
"""
import fitz  # PyMuPDF
fitz.TOOLS.mupdf_display_errors(False)    # Suppress error messages on stderr
fitz.TOOLS.mupdf_display_warnings(False)  # Suppress warning messages on stderr
from typing import Tuple, Optional, Dict
from io import BytesIO
from PIL import Image


def make_pdf_rect(page: fitz.Page, region: Dict[str, float]) -> fitz.Rect:
    """
    Convert ratio-based region to PDF coordinates
    
    Args:
        page: PyMuPDF page object
        region: Dict with x_ratio, y_ratio, w_ratio, h_ratio (0.0-1.0)
        
    Returns:
        fitz.Rect
    """
    w, h = page.rect.width, page.rect.height
    
    x0 = region['x_ratio'] * w
    y0 = region['y_ratio'] * h
    x1 = x0 + region['w_ratio'] * w
    y1 = y0 + region['h_ratio'] * h
    
    # Clamp to page bounds and ensure valid rectangle
    x0, x1 = sorted((max(0, min(int(round(x0)), int(w))),
                     max(0, min(int(round(x1)), int(w)))))
    y0, y1 = sorted((max(0, min(int(round(y0)), int(h))),
                     max(0, min(int(round(y1)), int(h)))))
    
    return fitz.Rect(x0, y0, x1, y1)


def extract_text_from_region(
    page: fitz.Page,
    region: Dict[str, float]
) -> str:
    """
    Extract native PDF text from a specific region
    
    Args:
        page: PyMuPDF page object
        region: Region dict with ratios
        
    Returns:
        Extracted text string
    """
    rect = make_pdf_rect(page, region)
    text = page.get_text('text', clip=rect).strip()
    return text


def bulk_extract_text_from_regions(
    pdf_path: str,
    region: Dict[str, float]
) -> Dict[int, str]:
    """
    Extract text from region for all pages in a single pass.

    More efficient than calling extract_text_from_region per page
    because we keep the document handle open.

    Args:
        pdf_path: Path to PDF file
        region: Region dict with x_ratio, y_ratio, w_ratio, h_ratio

    Returns:
        Dict mapping page_num (0-indexed) to extracted text
    """
    doc = fitz.open(pdf_path)
    results = {}

    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        rect = make_pdf_rect(page, region)
        text = page.get_text('text', clip=rect).strip()
        results[page_num] = text

    doc.close()
    return results


def extract_image_from_region(
    page: fitz.Page,
    region: Dict[str, float],
    zoom: float = 4.0,
    format: str = 'PNG'
) -> bytes:
    """
    Extract high-res image from a specific region
    
    Args:
        page: PyMuPDF page object
        region: Region dict with ratios
        zoom: Zoom factor for rendering (higher = better quality, larger file)
        format: Image format (PNG, JPEG)
        
    Returns:
        Image bytes
    """
    rect = make_pdf_rect(page, region)
    
    # Render at high resolution
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)
    
    # Convert to bytes
    img_bytes = pix.tobytes(output=format.lower())
    
    return img_bytes


def optimize_image_for_api(
    img_bytes: bytes,
    max_dimension: int = 2048,
    quality: int = 85
) -> bytes:
    """
    Optimize image size for API transmission while maintaining readability
    
    Args:
        img_bytes: Original image bytes
        max_dimension: Maximum width or height
        quality: JPEG quality (1-100)
        
    Returns:
        Optimized image bytes
    """
    img = Image.open(BytesIO(img_bytes))
    
    # Resize if too large
    if max(img.size) > max_dimension:
        ratio = max_dimension / max(img.size)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    
    # Convert to JPEG for smaller size
    output = BytesIO()
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    img.save(output, format='JPEG', quality=quality, optimize=True)
    
    return output.getvalue()


def extract_full_page_image(
    page: fitz.Page,
    zoom: float = 3.0,
    max_dimension: int = 4096,
    quality: int = 95
) -> bytes:
    """
    Render a full PDF page as a high-quality JPEG image.

    Args:
        page: PyMuPDF page object
        zoom: Zoom factor for rendering
        max_dimension: Max width or height in pixels
        quality: JPEG quality (1-100)

    Returns:
        JPEG image bytes
    """
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)

    img = Image.open(BytesIO(pix.tobytes('png')))

    # Downscale if needed
    if max(img.size) > max_dimension:
        ratio = max_dimension / max(img.size)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    output = BytesIO()
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    img.save(output, format='JPEG', quality=quality, optimize=True)
    return output.getvalue()


def get_pdf_page_count(pdf_path: str) -> int:
    """Get total number of pages in PDF"""
    doc = fitz.open(pdf_path)
    count = doc.page_count
    doc.close()
    return count


def load_pdf_page(pdf_path: str, page_num: int) -> fitz.Page:
    """
    Load a specific page from PDF
    
    Args:
        pdf_path: Path to PDF file
        page_num: Page number (0-indexed)
        
    Returns:
        PyMuPDF Page object
    """
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num)
    return page
