import time
import logging
from datetime import datetime
from typing import List, Dict, Any
from config_loader import Config
from elasticsearch_api_client import ElasticsearchApiClient
from utils import MetricsCollector # For type hinting

logger = logging.getLogger(__name__)

def create_es_index_if_not_exists(
    es_client: ElasticsearchApiClient, 
    index_name: str,
    metrics_collector: MetricsCollector
):
    if not Config.EMBEDDING_DIM:
        msg = "Embedding dimension is not set. Cannot create Elasticsearch index mapping."
        logger.error(f"❌ {msg}")
        metrics_collector.log_error("es_index_creation", msg, index_name)
        return False

    if es_client.index_exists(index_name):
        logger.info(f"ℹ️ Elasticsearch index '{index_name}' already exists.")
        # TODO: Optionally, verify existing mapping's embedding dimension
        return True

    mapping = {
        "mappings": {
            "properties": {
                "text": {"type": "text", "analyzer": "standard"},
                "source_pdf": {"type": "keyword"},
                "page_numbers": {"type": "integer"},
                "chunk_id": {"type": "keyword"},
                "timestamp": {"type": "date"},
                "metadata": {"type": "object", "enabled": True}, # Store extra metadata
                "embedding": {
                    "type": "dense_vector",
                    "dims": Config.EMBEDDING_DIM,
                    "index": True, 
                    "similarity": "cosine" # Important for semantic search
                }
            }
        }
    }
    logger.info(f"🚀 Creating Elasticsearch index '{index_name}' with embedding dim {Config.EMBEDDING_DIM}...")
    if es_client.create_index(index_name, mapping):
        logger.info(f"✅ Index '{index_name}' created successfully.")
        return True
    else:
        msg = f"Failed to create Elasticsearch index '{index_name}'."
        logger.error(f"❌ {msg}")
        metrics_collector.log_error("es_index_creation", msg, index_name)
        return False

def index_documents_to_elasticsearch(
    es_client: ElasticsearchApiClient, 
    chunks_with_embeddings: List[Dict[str, Any]], 
    index_name: str,
    metrics_collector: MetricsCollector
):
    if not chunks_with_embeddings:
        logger.info("No chunks to index into Elasticsearch.")
        return

    logger.info(f"🔄 Indexing {len(chunks_with_embeddings)} chunks into Elasticsearch index '{index_name}'...")
    
    operations = []
    for item in chunks_with_embeddings:
        chunk_doc = item['chunk']
        embedding_vector = item['embedding'] # This can be None if embedding failed

        doc_body = {
            "text": chunk_doc['text'],
            "source_pdf": chunk_doc.get('pdf_filename', 'unknown_source'),
            "page_numbers": chunk_doc.get('page_numbers', []),
            "chunk_id": chunk_doc['id'], # Should be set during deduplication
            "timestamp": datetime.now().isoformat(),
            "metadata": chunk_doc.get('metadata', {})
        }
        if embedding_vector: # Only include embedding if it exists and is valid
             if Config.EMBEDDING_DIM and len(embedding_vector) == Config.EMBEDDING_DIM:
                doc_body["embedding"] = embedding_vector
             else:
                logger.warning(f"Chunk {chunk_doc['id']} has an invalid or mismatched dimension embedding (len: {len(embedding_vector) if embedding_vector else 'N/A'}). Indexing without embedding.")
        else:
            logger.warning(f"Chunk {chunk_doc['id']} has no embedding. Indexing without it.")
            
        operations.append({
            "index": {"_index": index_name, "_id": chunk_doc['id']},
            "body": doc_body
        })

        # Process batch
        if len(operations) >= Config.BATCH_SIZE: # Each item is one operation for ES bulk
            _perform_bulk_index(es_client, operations, index_name, metrics_collector)
            operations = []

    # Process any remaining operations
    if operations:
        _perform_bulk_index(es_client, operations, index_name, metrics_collector)

    successful = metrics_collector.metrics['elasticsearch_stats']['successful_indexes']
    failed = metrics_collector.metrics['elasticsearch_stats']['failed_indexes']
    logger.info(f"✅ Elasticsearch indexing complete. Successful: {successful}, Failed: {failed}")

def _perform_bulk_index(
    es_client: ElasticsearchApiClient, 
    batch_ops: List[Dict[str, Any]], 
    index_name: str,
    metrics_collector: MetricsCollector
):
    start_batch_time = time.time()
    response = es_client.bulk_index(batch_ops)
    batch_duration = time.time() - start_batch_time
    metrics_collector.metrics['elasticsearch_stats']['batch_times'].append(batch_duration)

    items_in_batch = len(batch_ops)
    if response.get('errors'):
        errors_in_batch = 0
        for item_resp in response.get('items', []):
            action_result = item_resp.get('index', item_resp.get('create', {})) # ES bulk can have different actions
            if action_result.get('error'):
                errors_in_batch += 1
                logger.debug(f"Error indexing doc ID {action_result.get('_id', 'N/A')}: {action_result['error']}")
                metrics_collector.log_error(
                    "es_doc_index_error", 
                    str(action_result['error']), 
                    f"Index: {index_name}, Doc ID: {action_result.get('_id', 'N/A')}"
                )
        
        metrics_collector.metrics['elasticsearch_stats']['failed_indexes'] += errors_in_batch
        metrics_collector.metrics['elasticsearch_stats']['successful_indexes'] += (items_in_batch - errors_in_batch)
        logger.error(f"❌ Errors in Elasticsearch bulk indexing batch ({errors_in_batch}/{items_in_batch} failed). Batch took {batch_duration:.2f}s.")
    else:
        metrics_collector.metrics['elasticsearch_stats']['successful_indexes'] += items_in_batch
        logger.info(f"📦 Batch of {items_in_batch} chunks indexed to ES in {batch_duration:.2f}s.")