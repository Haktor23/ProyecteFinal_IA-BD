# pdf_parser.py
import fitz # PyMuPDF
import re
import os
import time
import logging
from typing import List, Dict, Tuple, Any, Generator
from ollama_api_client import OllamaApiClient # For image description
from config_loader import Config
from utils import MetricsCollector # For type hinting

logger = logging.getLogger(__name__)

def extract_pdf_pages_text_and_images(pdf_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Extracts text and image references from PDF pages."""
    start_time = time.time()
    pdf_filename = os.path.basename(pdf_path)
    logger.info(f"🔄 Extracting content from: {pdf_filename}")
    
    page_level_metrics = {
        'extraction_time': 0.0, 'pages_extracted': 0, 'images_found': 0,
        'errors': [] # List of {'type': str, 'msg': str, 'context': str}
    }
    
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        error_msg = f"Failed to open PDF {pdf_filename}: {e}"
        logger.error(f"❌ {error_msg}")
        page_level_metrics['errors'].append({'type': 'pdf_open', 'msg': error_msg, 'context': pdf_path})
        return [], page_level_metrics

    pages_data = []
    total_images_in_pdf = 0
    
    # Regex patterns for cleaning
    header_footer_patterns = [
        r"\bpage\s+\d+\s*(of\s+\d+)?\b", r"\bconfidential\b", r"©.*?\d{4}",
        r"^\s*\d+\s*$", r"\b(draft|internal use only)\b"
    ] # Add more as needed
    repeated_short_line_pattern = r"^(.{1,30})\n\1\n" # Detects repeated short lines (often headers/footers)
    whitespace_pattern = re.compile(r'\s+', re.UNICODE)
    multiple_newlines_pattern = re.compile(r'\n{3,}', re.UNICODE)

    for page_num_human, page in enumerate(doc, 1):
        try:
            text = page.get_text("text", sort=True) # Get sorted text
            
            # Basic cleaning
            text = re.sub(repeated_short_line_pattern, "\n", text, flags=re.IGNORECASE) # Remove repeated lines
            cleaned_lines = []
            for line in text.splitlines():
                is_header_footer = any(re.search(patt, line, re.IGNORECASE) for patt in header_footer_patterns)
                # MODIFIED LINE HERE: Keep non-empty, non-header/footer lines
                if not is_header_footer and line.strip(): 
                    cleaned_lines.append(line)
            
            cleaned_text = "\n".join(cleaned_lines)
            cleaned_text = multiple_newlines_pattern.sub('\n\n', cleaned_text) # Consolidate newlines
            cleaned_text = whitespace_pattern.sub(' ', cleaned_text).strip() # Consolidate whitespace

            images_on_page_info = []
            try:
                images_on_page = page.get_images(full=True)
                total_images_in_pdf += len(images_on_page)
                for img_info in images_on_page:
                    xref = img_info[0]
                    # Basic filter for very small images (likely icons or lines)
                    if img_info[3] > 50 and img_info[4] > 50: # width > 50px and height > 50px
                        images_on_page_info.append({'xref': xref, 'page_num': page_num_human})
            except Exception as img_ex:
                logger.warning(f"Could not extract images from page {page_num_human} of {pdf_filename}: {img_ex}")

            if images_on_page_info:
                logger.debug(f"🖼️ Page {page_num_human} ({pdf_filename}): Found {len(images_on_page_info)} relevant images.")
            
            pages_data.append({
                "page_num": page_num_human,
                "text": cleaned_text,
                "image_refs": images_on_page_info, # Store refs for later processing
                "page_obj_for_image_extraction": page, # Keep page object if needed for image bytes
                "char_count": len(cleaned_text)
            })
        except Exception as page_ex:
            error_msg = f"Error processing page {page_num_human} in {pdf_filename}: {page_ex}"
            logger.error(f"❌ {error_msg}")
            page_level_metrics['errors'].append({'type': 'page_processing', 'msg': error_msg, 'context': f"{pdf_path} p{page_num_human}"})

    processing_time = time.time() - start_time
    page_level_metrics.update({
        'extraction_time': processing_time,
        'pages_extracted': len(pages_data),
        'images_found': total_images_in_pdf # Total images detected before filtering
    })
    logger.info(f"✅ Extracted {len(pages_data)} pages, found {total_images_in_pdf} raw images from {pdf_filename} in {processing_time:.2f}s")
    
    return pages_data, page_level_metrics


def describe_page_images(
    page_data: Dict[str, Any],
    ollama_client: OllamaApiClient,
    metrics_collector: MetricsCollector,
    pdf_filename: str
) -> str:
    """Describes images for a single page using Ollama."""
    if not page_data['image_refs'] or Config.VISION_MODEL.lower() == "disabled" or not ollama_client:
        return ""

    page_image_descriptions = []
    successful_descriptions_count = 0
    
    page_obj = page_data['page_obj_for_image_extraction']
    doc_obj = page_obj.parent # Get document from page. Using different name to avoid confusion with 'doc' above.

    logger.info(f"🖼️ Describing {len(page_data['image_refs'])} images for page {page_data['page_num']} of {pdf_filename} using {Config.VISION_MODEL}...")
    
    for idx, img_ref in enumerate(page_data['image_refs']):
        img_process_start_time = time.time()
        try:
            img_bytes_dict = doc_obj.extract_image(img_ref['xref'])
            if not img_bytes_dict or 'image' not in img_bytes_dict:
                logger.warning(f"Could not extract bytes for image xref {img_ref['xref']} on page {page_data['page_num']}.")
                continue
            
            img_bytes = img_bytes_dict['image']
            description = ollama_client.describe_image(img_bytes, Config.VISION_MODEL)
            
            if description:
                page_image_descriptions.append(f"[Image {idx+1} on page {page_data['page_num']} description: {description}]")
                successful_descriptions_count += 1
                img_time = time.time() - img_process_start_time
                metrics_collector.metrics['image_processing_stats']['count'] += 1
                metrics_collector.metrics['image_processing_stats']['total_time'] += img_time
            else:
                logger.warning(f"Empty description for image {idx+1} on page {page_data['page_num']} of {pdf_filename}.")

        except Exception as e:
            error_msg = f"Error processing image {idx+1} (xref {img_ref['xref']}) on page {page_data['page_num']} of {pdf_filename}: {e}"
            logger.error(f"❌ {error_msg}")
            metrics_collector.log_error('image_description', error_msg, f"{pdf_filename}, page {page_data['page_num']}, img_idx {idx+1}")
            
    metrics_collector.metrics['pdf_stats']['total_images_described'] += successful_descriptions_count
    if page_image_descriptions:
        logger.info(f"✅ Described {successful_descriptions_count}/{len(page_data['image_refs'])} images for page {page_data['page_num']} of {pdf_filename}.")
    return "\n".join(page_image_descriptions)


def generate_page_context_windows(pages_data: List[Dict[str, Any]]) -> Generator[List[Dict[str, Any]], None, None]:
    """Applies a sliding window over extracted page data."""
    if not pages_data:
        return
    
    window_size = Config.WINDOW_SIZE
    overlap = Config.OVERLAP
    step = window_size - overlap
    if step <= 0:
        step = 1
    
    logger.debug(f"🪟 Sliding window: size={window_size}, overlap={overlap}, step={step}")
    
    for i in range(0, len(pages_data), step):
        window = pages_data[i : i + window_size]
        if window: # Ensure window is not empty
            yield window