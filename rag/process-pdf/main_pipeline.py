# main_pipeline.py
import os
import json
import argparse
import time
import logging # Main logger setup will be here
from typing import List, Dict, Any, Generator

# Modular imports
from config_loader import Config
from utils import setup_logging, MetricsCollector
from ollama_api_client import OllamaApiClient
from elasticsearch_api_client import ElasticsearchApiClient
from pdf_parser import (
    extract_pdf_pages_text_and_images, 
    describe_page_images,
    generate_page_context_windows
)
from text_chunker import SimpleTextSplitter
from embedding_processor import deduplicate_chunks_by_similarity
from index_manager import create_es_index_if_not_exists, index_documents_to_elasticsearch

# Initialize logger and metrics (globally accessible within this script context after setup)
# Specific module loggers will be children of this root logger.
logger = setup_logging() # Sets up the root logger
metrics = MetricsCollector()


def process_single_pdf(
    pdf_path: str,
    ollama_client: OllamaApiClient, # Pass the client
    text_splitter: SimpleTextSplitter # Pass the splitter
) -> List[Dict[str, Any]]:
    """Processes a single PDF: extract, describe images, chunk."""
    pdf_filename = os.path.basename(pdf_path)
    raw_chunks_for_pdf = []
    
    pdf_process_start_time = time.time()
    
    # 1. Extract text and image references from PDF
    pages_data, pdf_extraction_metrics = extract_pdf_pages_text_and_images(pdf_path)
    
    # Update global metrics from this PDF's extraction
    metrics.metrics['pdf_stats']['total_pages_extracted'] += pdf_extraction_metrics['pages_extracted']
    metrics.metrics['pdf_stats']['total_images_found'] += pdf_extraction_metrics['images_found']
    for err in pdf_extraction_metrics['errors']:
        metrics.log_error(err['type'], err['msg'], err['context'])

    if not pages_data:
        logger.warning(f"No data extracted from {pdf_filename}. Skipping further processing for this PDF.")
        metrics.metrics['pdf_stats']['failed_pdfs_processed'] += 1
        return []
    
    # 2. Process pages in windows, describe images, and chunk text
    for window_idx, page_window_data in enumerate(generate_page_context_windows(pages_data)):
        window_texts_and_img_desc = []
        window_contributing_page_nums = []

        for page_content in page_window_data:
            window_texts_and_img_desc.append(page_content['text'])
            window_contributing_page_nums.append(page_content['page_num'])
            
            # Describe images for this page if Ollama client is available
            if ollama_client and Config.VISION_MODEL.lower() != "disabled":
                image_descriptions = describe_page_images(page_content, ollama_client, metrics, pdf_filename)
                if image_descriptions:
                    window_texts_and_img_desc.append(f"\n[Image Context from page {page_content['page_num']}:\n{image_descriptions}\n]")
        
        combined_text_for_window = "\n\n".join(filter(None, window_texts_and_img_desc)).strip()
        
        if not combined_text_for_window: # Skip if window resulted in no text
            continue

        split_chunk_texts = text_splitter.split_text(combined_text_for_window)
        
        for chunk_text_content in split_chunk_texts:
            if chunk_text_content: # Ensure chunk is not empty
                raw_chunks_for_pdf.append({
                    # ID will be properly assigned during/after deduplication
                    "text": chunk_text_content,
                    "pdf_filename": pdf_filename,
                    "page_numbers": sorted(list(set(window_contributing_page_nums))),
                    "metadata": {
                        "char_count": len(chunk_text_content),
                        "source_window_index": window_idx
                    }
                })
    
    pdf_duration = time.time() - pdf_process_start_time
    logger.info(f"📄 PDF '{pdf_filename}' processed in {pdf_duration:.2f}s, yielding {len(raw_chunks_for_pdf)} raw chunks.")
    metrics.metrics['pdf_stats']['successful_pdfs_processed'] += 1
    return raw_chunks_for_pdf


def run_ingestion_pipeline(pdf_dir: str, output_json_path: str, es_client: ElasticsearchApiClient, ollama_client: OllamaApiClient):
    all_raw_chunks_from_all_pdfs = []
    
    try:
        pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    except FileNotFoundError:
        logger.error(f"❌ PDF directory not found: {pdf_dir}")
        return
        
    metrics.metrics['pdf_stats']['total_pdfs_found'] = len(pdf_files)
    
    if not pdf_files:
        logger.info(f"No PDF files found in directory: {pdf_dir}")
        return

    logger.info(f"📂 Found {len(pdf_files)} PDF(s) in '{pdf_dir}'. Starting processing...")

    # Initialize text splitter
    splitter = SimpleTextSplitter(chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP)

    for filename in pdf_files: # Using tqdm here can be nice: from tqdm import tqdm; for filename in tqdm(pdf_files):
        pdf_full_path = os.path.join(pdf_dir, filename)
        chunks_from_this_pdf = process_single_pdf(pdf_full_path, ollama_client, splitter)
        all_raw_chunks_from_all_pdfs.extend(chunks_from_this_pdf)

    logger.info(f"📊 Total raw chunks from all PDFs: {len(all_raw_chunks_from_all_pdfs)}")

    # 3. Deduplicate all collected chunks (this now returns list of {'chunk': data, 'embedding': vector})
    # It uses ollama_client internally for embeddings
    final_chunks_with_embeddings = deduplicate_chunks_by_similarity(all_raw_chunks_from_all_pdfs, ollama_client, metrics)
    
    # 4. Save deduplicated chunk text to JSON (optional)
    final_chunks_for_json = [item['chunk'] for item in final_chunks_with_embeddings] # Extract only chunk data
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(final_chunks_for_json, f, ensure_ascii=False, indent=4)
        logger.info(f"💾 Deduplicated chunks ({len(final_chunks_for_json)}) saved to: {output_json_path}")
    except IOError as e:
        logger.error(f"❌ Error saving chunks to JSON '{output_json_path}': {e}")
        metrics.log_error('json_save_error', str(e), output_json_path)

    # 5. Index to Elasticsearch (if client is available)
    if es_client and Config.ES_INDEX:
        if create_es_index_if_not_exists(es_client, Config.ES_INDEX, metrics): # Ensures Config.EMBEDDING_DIM is ready
             index_documents_to_elasticsearch(es_client, final_chunks_with_embeddings, Config.ES_INDEX, metrics)
        else:
            logger.error(f"Skipping Elasticsearch indexing due to index creation/check failure for '{Config.ES_INDEX}'.")
    elif not Config.ES_INDEX:
        logger.info("Elasticsearch index name not specified (ES_INDEX). Skipping indexing.")
    else: # es_client is None
        logger.info("Elasticsearch client not available. Skipping indexing.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Ingestion Pipeline using Ollama Embeddings")
    parser.add_argument("--pdf_folder", type=str, default=Config.PDF_FOLDER, help="Directory with PDFs.")
    parser.add_argument("--output_json", type=str, default=Config.JSON_OUTPUT, help="Output JSON file for chunks.")
    parser.add_argument("--recreate_index", action='store_true', help="Delete and recreate Elasticsearch index if it exists.")
    
    args = parser.parse_args()

    pipeline_start_time = time.time()
    logger.info("🚀 RAG Ingestion Pipeline (Ollama Embeddings) Started 🚀")

    # Initialize API clients
    ollama_cli = None
    try:
        ollama_cli = OllamaApiClient(base_url=Config.OLLAMA_URL)
        if not ollama_cli.health_check():
            logger.error(f"❌ Ollama health check failed at {Config.OLLAMA_URL}. Ensure Ollama is running and accessible.")
            ollama_cli = None # Disable if not healthy
        else:
            logger.info(f"✅ Ollama client initialized and healthy ({Config.OLLAMA_URL}). Embedding model: {Config.EMBED_MODEL_OLLAMA}")
            # Attempt to get embedding dimension early if not set
            if not Config.EMBEDDING_DIM:
                try:
                    logger.info("Attempting to pre-fetch embedding dimension from Ollama...")
                    # This call to generate_embedding will set Config.EMBEDDING_DIM on success
                    test_emb = ollama_cli.generate_embedding("dimension test", Config.EMBED_MODEL_OLLAMA)
                    if not test_emb and not Config.EMBEDDING_DIM: # If it returned empty and dim still not set
                        logger.warning("Could not pre-fetch embedding dimension. Will attempt during processing.")
                except Exception as e:
                    logger.warning(f"Could not pre-fetch embedding dimension: {e}. Will attempt during processing.")

    except Exception as e:
        logger.error(f"❌ Failed to initialize Ollama client: {e}")
        ollama_cli = None
    
    es_cli = None
    if Config.ES_HOST and Config.ES_INDEX : # Only initialize if ES is configured
        try:
            es_cli = ElasticsearchApiClient(host=Config.ES_HOST, username=Config.ES_USERNAME, password=Config.ES_PASSWORD)
            if not es_cli.health_check():
                logger.error(f"❌ Elasticsearch health check failed at {Config.ES_HOST}. Ensure Elasticsearch is running.")
                es_cli = None # Disable if not healthy
            else:
                logger.info(f"✅ Elasticsearch client initialized and healthy ({Config.ES_HOST}). Index: {Config.ES_INDEX}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Elasticsearch client: {e}")
            es_cli = None
    else:
        logger.info("Elasticsearch host or index not configured. Skipping Elasticsearch client initialization.")


    if not ollama_cli:
        logger.critical("Ollama client is not available. Cannot proceed with embedding-dependent tasks. Exiting.")
        exit(1)

    if args.recreate_index and es_cli and Config.ES_INDEX:
        logger.warning(f"🔥 Option --recreate_index used. Attempting to delete index '{Config.ES_INDEX}'...")
        if es_cli.delete_index(Config.ES_INDEX):
            logger.info(f"🗑️ Index '{Config.ES_INDEX}' deleted or did not exist.")
        else:
            logger.error(f"⚠️ Failed to ensure deletion of index '{Config.ES_INDEX}'.")
            # Potentially exit if deletion is critical and failed. For now, we continue.

    # Run the main processing logic
    run_ingestion_pipeline(args.pdf_folder, args.output_json, es_cli, ollama_cli)

    pipeline_duration = time.time() - pipeline_start_time
    logger.info(f"🏁 RAG Ingestion Pipeline finished in {pipeline_duration:.2f} seconds.")
    
    # Display metrics summary
    summary = metrics.get_summary()
    logger.info("📊 Processing Metrics Summary:")
    for key, value in summary.items():
        formatted_value = f"{value:.2f}" if isinstance(value, float) else value
        logger.info(f"   ├── {key}: {formatted_value}")
    
    if metrics.metrics['errors']:
        logger.warning(f"⚠️ Encountered {len(metrics.metrics['errors'])} errors during execution. Check 'logs/errors.log' and general logs.")