# embedding_processor.py
import numpy as np
import faiss
import uuid
import time
import logging
import re
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from ollama_api_client import OllamaApiClient
from config_loader import Config
from utils import MetricsCollector
import requests

logger = logging.getLogger(__name__)

@dataclass
class ChunkRefinementStats:
    """Statistics for chunk refinement operations"""
    contextual_refined: int = 0
    individual_refined: int = 0
    failed_refinements: int = 0
    total_processing_time: float = 0.0

@dataclass
class RefinementConfig:
    """Configuration for text refinement operations"""
    contextual_timeout: int = 150
    individual_timeout: int = 90
    min_length_factor: float = 0.80
    max_length_factor: float = 1.20
    fixed_max_addition: int = 150
    context_window_size: int = 500

class TextRefinementEngine:
    """Handles text refinement operations using LLM"""
    
    def __init__(self, ollama_client: OllamaApiClient, refinement_model: str):
        self.ollama_client = ollama_client
        self.refinement_model = refinement_model
        self.config = RefinementConfig()
        self.stats = ChunkRefinementStats()
        
        # Compiled regex patterns for better performance
        self.think_tag_pattern = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
        self.think_single_pattern = re.compile(r"<\/?think>", re.IGNORECASE)
        
        # Predefined patterns for efficiency
        self.common_cut_patterns = {
            "nes ", "cion ", "ción ", "mpeten", " pa)", " pa ", "e ", "s ", "d "
        }
        self.preambles_to_remove = [
            "okay, let's tackle this query.", "okay, let's tackle this.", 
            "here's the refined chunk:", "refined chunk:", "el fragmento refinado es:",
            "refined current chunk (output only this):", "refined current chunk:", 
            "output del 'fragmento actual' refinado:", "fragmento refinado (solo esta salida):"
        ]

    def needs_start_refinement(self, text: str) -> bool:
        """
        Determines if text needs refinement at the start
        
        Args:
            text: Text to analyze
            
        Returns:
            True if text needs refinement, False otherwise
        """
        if not text or len(text) < 3:
            return False
            
        first_char = text[0]
        
        # Check for lowercase start (excluding vowels)
        if first_char.islower() and first_char not in "aioeuyáéíóú":
            if any(text.lower().startswith(cut) for cut in self.common_cut_patterns):
                return True
                
            first_word = text.split(' ', 1)[0]
            if (len(first_word) <= 4 and 
                sum(1 for char in first_word if char.lower() in "aeiouáéíóú") == 0):
                return True
        
        # Check for punctuation start
        if first_char in "),.;:%":
            return True
            
        return False

    def _clean_llm_output(self, output: str, original_text: str) -> str:
        """
        Clean and validate LLM output
        
        Args:
            output: Raw LLM output
            original_text: Original text for fallback
            
        Returns:
            Cleaned output or original text if cleaning fails
        """
        if not output:
            return original_text
            
        # Remove think tags
        cleaned = self.think_tag_pattern.sub("", output).strip()
        cleaned = self.think_single_pattern.sub("", cleaned).strip()
        
        # Remove common preambles
        cleaned_lower = cleaned.lower()
        for preamble in self.preambles_to_remove:
            if cleaned_lower.startswith(preamble):
                cleaned = cleaned[len(preamble):].lstrip(" :-\n")
                break
        
        # Remove surrounding quotes
        if ((cleaned.startswith('"') and cleaned.endswith('"')) or
            (cleaned.startswith("'") and cleaned.endswith("'"))):
            if len(cleaned) > 1:
                cleaned = cleaned[1:-1].strip()
        
        return cleaned if cleaned.strip() else original_text

    def _validate_refinement_length(self, original: str, refined: str, 
                                   is_contextual: bool = False) -> bool:
        """
        Validate that refined text length is within acceptable bounds
        
        Args:
            original: Original text
            refined: Refined text
            is_contextual: Whether this is contextual refinement
            
        Returns:
            True if length is acceptable, False otherwise
        """
        if len(original) < 15:  # Skip validation for very short texts
            return True
            
        # Adjust factors for short texts
        if len(original) < 50:
            min_factor, max_factor = 0.5, 2.0
            max_addition = 100
        else:
            min_factor = self.config.min_length_factor
            max_factor = self.config.max_length_factor
            max_addition = self.config.fixed_max_addition if is_contextual else 0
        
        min_len = min_factor * len(original)
        max_len = (max_factor * len(original)) + max_addition
        
        return min_len <= len(refined) <= max_len

    def refine_chunk_with_llm(self, text: str, is_contextual: bool = False,
                             original_for_fallback: Optional[str] = None,
                             timeout_seconds: Optional[int] = None) -> str:
        """
        Refine text chunk using LLM
        
        Args:
            text: Text to refine (or prompt if is_contextual)
            is_contextual: Whether this is contextual refinement
            original_for_fallback: Original text for fallback
            timeout_seconds: Timeout for the request
            
        Returns:
            Refined text or original if refinement fails
        """
        start_time = time.time()
        
        original_text = original_for_fallback if original_for_fallback else text
        timeout = timeout_seconds or (
            self.config.contextual_timeout if is_contextual 
            else self.config.individual_timeout
        )
        
        if is_contextual:
            prompt = text  # Already constructed prompt
        else:
            prompt = self._build_individual_refinement_prompt(original_text)
        
        log_prefix = "Contextual" if is_contextual else "Individual"
        log_sample = original_text[:70]
        
        try:
            payload = {
                "model": self.refinement_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_ctx": 4096}
            }
            
            logger.debug(f"{log_prefix} LLM refinement for '{log_sample}...' "
                        f"(Timeout: {timeout}s)")
            
            response = self.ollama_client.session.post(
                f"{self.ollama_client.base_url}/api/generate",
                json=payload,
                timeout=timeout
            )
            response.raise_for_status()
            
            llm_output = response.json().get('response', '').strip()
            cleaned_output = self._clean_llm_output(llm_output, original_text)
            
            # Validate length
            if not self._validate_refinement_length(
                original_text, cleaned_output, is_contextual
            ):
                logger.warning(
                    f"{log_prefix} refinement for '{log_sample}...' "
                    f"failed length validation (Original: {len(original_text)}, "
                    f"Refined: {len(cleaned_output)}). Using original."
                )
                self.stats.failed_refinements += 1
                return original_text
            
            # Log successful refinement
            if cleaned_output != original_text:
                refinement_type = "contextual" if is_contextual else "individual"
                logger.info(
                    f"Chunk refined ({refinement_type}): "
                    f"'{log_sample}...' -> '{cleaned_output[:70]}...'"
                )
                if is_contextual:
                    self.stats.contextual_refined += 1
                else:
                    self.stats.individual_refined += 1
            
            return cleaned_output
            
        except requests.Timeout:
            logger.error(
                f"TIMEOUT during {log_prefix} refinement for '{log_sample}...' "
                f"(timeout: {timeout}s)"
            )
            self.stats.failed_refinements += 1
            return original_text
            
        except requests.RequestException as e:
            logger.error(
                f"Network error during {log_prefix} refinement for "
                f"'{log_sample}...': {e}"
            )
            self.stats.failed_refinements += 1
            return original_text
            
        except Exception as e:
            logger.error(
                f"Unexpected error during {log_prefix} refinement for "
                f"'{log_sample}...': {e}"
            )
            self.stats.failed_refinements += 1
            return original_text
            
        finally:
            self.stats.total_processing_time += time.time() - start_time

    def _build_individual_refinement_prompt(self, text: str) -> str:
        """Build prompt for individual chunk refinement"""
        return (
            f"Eres un asistente de corrección de texto altamente preciso. "
            f"Tu tarea es corregir el siguiente fragmento de texto, "
            f"que podría comenzar o terminar de forma abrupta. "
            f"Asegúrate de que la salida comience y termine como una pieza "
            f"de texto completa y coherente. "
            f"Realiza ÚNICAMENTE LOS CAMBIOS MÍNIMOS ABSOLUTAMENTE NECESARIOS. "
            f"Si el fragmento ya parece completo, devuélvelo SIN CAMBIOS. "
            f"NO incluyas NINGUNA explicación ni texto que no sea el propio "
            f"fragmento refinado. "
            f"NO uses etiquetas como <think>. Tu respuesta debe ser ÚNICAMENTE "
            f"el fragmento de texto.\n\n"
            f"Fragmento original:\n\"\"\"\n{text}\n\"\"\"\n\n"
            f"Fragmento refinado (solo esta salida):"
        )

    def _build_contextual_refinement_prompt(self, prev_chunk: str, 
                                          current_chunk: str, 
                                          next_chunk: str) -> str:
        """Build prompt for contextual chunk refinement"""
        return (
            f"ACTÚA COMO UN EDITOR DE TEXTO PRECISO. Tu única tarea es refinar "
            f"el 'FRAGMENTO ACTUAL' dado el contexto del 'FRAGMENTO ANTERIOR' "
            f"y 'FRAGMENTO SIGUIENTE'.\n"
            f"OBJETIVO PRINCIPAL: El 'FRAGMENTO ACTUAL' refinado DEBE preservar "
            f"casi todo su contenido original. SOLO realiza ediciones mínimas "
            f"en el inicio o final del 'FRAGMENTO ACTUAL' para asegurar que "
            f"comience y termine con una palabra/frase completa, formando una "
            f"unidad semántica coherente.\n"
            f"REGLAS ESTRICTAS:\n"
            f"1. CAMBIOS MÍNIMOS: Si el 'FRAGMENTO ACTUAL' ya es coherente en "
            f"sus bordes dado el contexto, DEVUÉLVELO EXACTAMENTE IGUAL.\n"
            f"2. AJUSTE DE BORDES: Si el 'FRAGMENTO ACTUAL' está cortado, puedes:\n"
            f"   a. Precederlo con una PEQUEÑA parte final del 'FRAGMENTO ANTERIOR' "
            f"ÚNICAMENTE si es la continuación directa y necesaria.\n"
            f"   b. Añadirle una PEQUEÑA parte inicial del 'FRAGMENTO SIGUIENTE' "
            f"ÚNICAMENTE si es la continuación directa y necesaria.\n"
            f"3. NO RESUMIR NI ELIMINAR: NO elimines partes sustanciales del "
            f"'FRAGMENTO ACTUAL'. NO lo resumas. NO lo parafrasees extensamente.\n"
            f"4. NO FUSIONAR CHUNKS: Tu salida es SOLO el 'FRAGMENTO ACTUAL' "
            f"refinado, NO una fusión de los tres fragmentos.\n"
            f"5. SALIDA LIMPIA: Tu respuesta DEBE SER ÚNICA Y EXCLUSIVAMENTE "
            f"el texto del 'FRAGMENTO ACTUAL' refinado. SIN explicaciones, "
            f"SIN preámbulos, SIN etiquetas <think> o similares, SIN NADA MÁS.\n\n"
            f"CONTEXTO:\n"
            f"FRAGMENTO ANTERIOR (última parte, solo para contexto del inicio "
            f"del FRAGMENTO ACTUAL):\n\"\"\"\n{prev_chunk}\n\"\"\"\n\n"
            f"FRAGMENTO ACTUAL (este es el que debes editar y devolver):\n"
            f"\"\"\"\n{current_chunk}\n\"\"\"\n\n"
            f"FRAGMENTO SIGUIENTE (primera parte, solo para contexto del final "
            f"del FRAGMENTO ACTUAL):\n\"\"\"\n{next_chunk}\n\"\"\"\n\n"
            f"DEVUELVE ÚNICAMENTE EL TEXTO REFINADO DEL 'FRAGMENTO ACTUAL':"
        )

    def refine_chunk_in_contextual_window(self, prev_chunk: str, 
                                        current_chunk: str, 
                                        next_chunk: str) -> str:
        """
        Refine chunk using contextual window
        
        Args:
            prev_chunk: Previous chunk text (limited)
            current_chunk: Current chunk to refine
            next_chunk: Next chunk text (limited)
            
        Returns:
            Refined current chunk text
        """
        prompt = self._build_contextual_refinement_prompt(
            prev_chunk, current_chunk, next_chunk
        )
        
        return self.refine_chunk_with_llm(
            prompt,
            is_contextual=True,
            original_for_fallback=current_chunk,
            timeout_seconds=self.config.contextual_timeout
        )


class EmbeddingProcessor:
    """Main class for processing embeddings and deduplication"""
    
    def __init__(self, ollama_client: OllamaApiClient, 
                 metrics_collector: MetricsCollector):
        self.ollama_client = ollama_client
        self.metrics_collector = metrics_collector
        self.refinement_engine = TextRefinementEngine(
            ollama_client, Config.LLM_MODEL
        )

    def generate_embeddings_with_ollama(self, texts: List[str]) -> Tuple[List[List[float]], int]:
        """
        Generate embeddings using Ollama API
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Tuple of (embeddings_list, valid_count)
        """
        if not texts:
            return [], 0
        
        start_time = time.time()
        logger.info(f"🤖 Generating {len(texts)} embeddings using "
                   f"Ollama model: {Config.EMBED_MODEL_OLLAMA}...")
        
        embeddings = self.ollama_client.generate_embeddings_batch(
            texts, Config.EMBED_MODEL_OLLAMA
        )
        
        valid_embeddings_count = 0
        processed_embeddings = []
        
        for i, emb in enumerate(embeddings):
            if emb and (not Config.EMBEDDING_DIM or len(emb) == Config.EMBEDDING_DIM):
                valid_embeddings_count += 1
                processed_embeddings.append(emb)
            elif Config.EMBEDDING_DIM:
                logger.warning(
                    f"Using placeholder for text index {i} due to embedding "
                    f"issue (length: {len(emb) if emb else 'N/A'}, "
                    f"expected: {Config.EMBEDDING_DIM}). "
                    f"Text: '{texts[i][:50]}...'"
                )
                processed_embeddings.append([0.0] * Config.EMBEDDING_DIM)
            else:
                logger.warning(
                    f"Failed embedding for text index {i} with unknown dimension. "
                    f"Adding empty list. Text: '{texts[i][:50]}...'"
                )
                processed_embeddings.append([])

        duration = time.time() - start_time
        self.metrics_collector.log_processing_time('ollama_embedding_generation', duration)
        
        logger.info(f"✅ Generated {valid_embeddings_count}/{len(texts)} "
                   f"valid embeddings in {duration:.2f}s.")
        
        if valid_embeddings_count < len(texts):
            self.metrics_collector.log_error(
                'embedding_failure',
                f"{len(texts) - valid_embeddings_count} embeddings failed",
                f"Model: {Config.EMBED_MODEL_OLLAMA}"
            )
        
        return processed_embeddings, valid_embeddings_count

    def _perform_exact_deduplication(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove exact duplicate chunks
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            List of chunks with exact duplicates removed
        """
        unique_chunks_map = {}
        for chunk_dict in chunks:
            text = chunk_dict['text']
            if text not in unique_chunks_map:
                unique_chunks_map[text] = chunk_dict
        
        return list(unique_chunks_map.values())

    def _perform_contextual_refinement(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply contextual refinement to chunks
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            List of contextually refined chunks
        """
        if not chunks:
            return []
            
        logger.info(f"✨ Applying contextual refinement to {len(chunks)} "
                   f"chunks using sliding window and model '{Config.LLM_MODEL}'...")
        
        refined_chunks = []
        
        for i in range(len(chunks)):
            current_chunk = chunks[i]
            original_text = current_chunk['text']
            
            # Get context with size limits
            prev_text = ""
            if i > 0:
                prev_text = chunks[i-1]['text'][-self.refinement_engine.config.context_window_size:]
            
            next_text = ""
            if i < len(chunks) - 1:
                next_text = chunks[i+1]['text'][:self.refinement_engine.config.context_window_size:]
            
            # Refine with context
            refined_text = self.refinement_engine.refine_chunk_in_contextual_window(
                prev_text, original_text, next_text
            )
            
            # Create refined chunk
            refined_chunk = current_chunk.copy()
            refined_chunk['text'] = refined_text
            
            # Track original if changed
            if refined_text != original_text:
                refined_chunk['original_text_if_context_refined'] = original_text
            
            refined_chunks.append(refined_chunk)
        
        stats = self.refinement_engine.stats
        if stats.contextual_refined > 0:
            logger.info(f"✨ Contextually refined {stats.contextual_refined} "
                       f"of {len(chunks)} chunks.")
        
        return refined_chunks

    def _perform_individual_refinement(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply individual refinement to chunks that need it
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            List of individually refined chunks
        """
        if not chunks:
            return []
            
        logger.info(f"✨ Applying individual refinement to chunks...")
        
        refined_chunks = []
        initial_individual_count = self.refinement_engine.stats.individual_refined
        
        for chunk_dict in chunks:
            original_text = chunk_dict['text']
            refined_text = original_text
            
            if self.refinement_engine.needs_start_refinement(original_text):
                refined_text = self.refinement_engine.refine_chunk_with_llm(
                    original_text,
                    is_contextual=False,
                    original_for_fallback=original_text
                )
            
            # Create refined chunk
            refined_chunk = chunk_dict.copy()
            refined_chunk['text'] = refined_text
            
            # Track original if changed
            if refined_text != original_text:
                refined_chunk['original_text_if_individually_refined'] = original_text
            
            refined_chunks.append(refined_chunk)
        
        individual_refined_count = (
            self.refinement_engine.stats.individual_refined - initial_individual_count
        )
        
        if individual_refined_count > 0:
            logger.info(f"✨ Individually refined {individual_refined_count} "
                       f"additional chunks.")
        
        return refined_chunks

    def _build_faiss_index(self, embeddings: List[List[float]]) -> Optional[faiss.Index]:
        """
        Build FAISS index from embeddings
        
        Args:
            embeddings: List of embedding vectors
            
        Returns:
            FAISS index or None if creation fails
        """
        if not embeddings:
            return None
            
        vectors = np.array(embeddings, dtype='float32')
        
        # Confirm embedding dimension if not set
        if not Config.EMBEDDING_DIM and vectors.ndim == 2 and vectors.shape[1] > 0:
            Config.EMBEDDING_DIM = vectors.shape[1]
            logger.info(f"📐 Embedding dimension confirmed from FAISS vectors: "
                       f"{Config.EMBEDDING_DIM}")
        
        if (vectors.shape[0] == 0 or vectors.ndim != 2 or 
            not Config.EMBEDDING_DIM or Config.EMBEDDING_DIM == 0):
            logger.error("❌ No valid vectors for FAISS index. "
                        "Semantic deduplication skipped.")
            return None
        
        try:
            index = faiss.IndexFlatIP(Config.EMBEDDING_DIM)
            index.add(vectors)
            return index
        except Exception as e:
            logger.error(f"❌ Error creating FAISS index: {e}. "
                        "Skipping semantic deduplication.")
            return None

    def _find_semantic_duplicates(self, index: faiss.Index, 
                                 vectors: np.ndarray) -> set:
        """
        Find semantic duplicates using FAISS index
        
        Args:
            index: FAISS index
            vectors: Embedding vectors
            
        Returns:
            Set of indices to keep (duplicates removed)
        """
        kept_indices = set(range(vectors.shape[0]))
        k_search = min(5, vectors.shape[0])
        
        if k_search <= 1:
            return kept_indices
            
        logger.info(f"🔎 Searching for semantic duplicates "
                   f"(k={k_search}, threshold={Config.SIMILARITY_THRESHOLD})...")
        
        similarities, indices = index.search(vectors, k=k_search)
        
        for i in range(vectors.shape[0]):
            if i not in kept_indices:
                continue
                
            for neighbor_idx in range(1, k_search):
                original_neighbor_idx = indices[i, neighbor_idx]
                
                if (original_neighbor_idx == -1 or 
                    original_neighbor_idx == i or
                    original_neighbor_idx not in kept_indices):
                    continue
                
                similarity = similarities[i, neighbor_idx]
                if similarity >= Config.SIMILARITY_THRESHOLD:
                    kept_indices.remove(original_neighbor_idx)
        
        return kept_indices

    def deduplicate_chunks_by_similarity(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Main function to deduplicate and refine chunks
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            List of deduplicated and refined chunks
        """
        if not chunks:
            return []

        start_time = time.time()
        original_count = len(chunks)
        
        self.metrics_collector.metrics['chunk_stats']['total_chunks_raw'] += original_count
        logger.info(f"🔄 Starting deduplication and refinement for "
                   f"{original_count} raw chunks...")

        # Step 1: Exact deduplication
        chunks_after_exact = self._perform_exact_deduplication(chunks)
        exact_removed = original_count - len(chunks_after_exact)
        
        if exact_removed > 0:
            logger.info(f"ℹ️ Removed {exact_removed} exact duplicates. "
                       f"{len(chunks_after_exact)} remaining.")

        if not chunks_after_exact:
            self.metrics_collector.metrics['chunk_stats']['duplicate_chunks_removed'] = exact_removed
            return []

        # Step 2: Contextual refinement
        contextually_refined = self._perform_contextual_refinement(chunks_after_exact)
        
        # Optional: Individual refinement (currently commented out in original)
        # final_refined = self._perform_individual_refinement(contextually_refined)
        final_refined = contextually_refined

        # Step 3: Generate embeddings for semantic deduplication
        texts_to_embed = [chunk['text'] for chunk in final_refined]
        if not texts_to_embed:
            logger.info("No texts for embedding generation after refinement.")
            self.metrics_collector.metrics['chunk_stats']['duplicate_chunks_removed'] = exact_removed
            self.metrics_collector.metrics['chunk_stats']['total_chunks_final'] = 0
            return []

        embeddings_list, _ = self.generate_embeddings_with_ollama(texts_to_embed)

        # Step 4: Prepare chunks with valid embeddings
        embeddable_chunks = []
        skipped_chunks = []

        for i, chunk_data in enumerate(final_refined):
            embedding = embeddings_list[i]
            is_valid_embedding = (
                embedding and
                (not Config.EMBEDDING_DIM or len(embedding) == Config.EMBEDDING_DIM) and
                (not Config.EMBEDDING_DIM or not all(v == 0.0 for v in embedding))
            )

            if is_valid_embedding:
                embeddable_chunks.append({'chunk': chunk_data, 'embedding': embedding})
            else:
                logger.warning(f"Skipping chunk for FAISS due to invalid embedding: "
                             f"'{chunk_data['text'][:50]}...'")
                chunk_data['id'] = chunk_data.get('id', str(uuid.uuid4()))
                skipped_chunks.append({'chunk': chunk_data, 'embedding': None})

        if not embeddable_chunks:
            logger.warning("No valid embeddings for semantic deduplication.")
            self.metrics_collector.metrics['chunk_stats']['duplicate_chunks_removed'] = exact_removed
            return [item['chunk'] for item in skipped_chunks]

        # Step 5: Semantic deduplication using FAISS
        embeddings_for_faiss = [item['embedding'] for item in embeddable_chunks]
        faiss_index = self._build_faiss_index(embeddings_for_faiss)

        if not faiss_index:
            # Fallback without semantic deduplication
            result_chunks = []
            for item in embeddable_chunks:
                item['chunk']['id'] = item['chunk'].get('id', str(uuid.uuid4()))
                result_chunks.append(item['chunk'])
            result_chunks.extend([item['chunk'] for item in skipped_chunks])
            
            self.metrics_collector.metrics['chunk_stats']['duplicate_chunks_removed'] = exact_removed
            self.metrics_collector.metrics['chunk_stats']['total_chunks_final'] = len(result_chunks)
            return result_chunks

        # Find and remove semantic duplicates
        vectors = np.array(embeddings_for_faiss, dtype='float32')
        kept_indices = self._find_semantic_duplicates(faiss_index, vectors)

        # Build final result
        final_chunks = []
        for i in sorted(kept_indices):
            item = embeddable_chunks[i]
            item['chunk']['id'] = item['chunk'].get('id', str(uuid.uuid4()))
            final_chunks.append(item['chunk'])

        # Add skipped chunks
        final_chunks.extend([item['chunk'] for item in skipped_chunks])

        # Update metrics
        semantic_removed = len(embeddable_chunks) - len(kept_indices)
        total_removed = exact_removed + semantic_removed
        
        self.metrics_collector.metrics['chunk_stats']['duplicate_chunks_removed'] = total_removed
        self.metrics_collector.metrics['chunk_stats']['total_chunks_final'] = len(final_chunks)

        # Log final results
        duration = time.time() - start_time
        logger.info(f"✅ Deduplication and refinement completed in {duration:.2f}s. "
                   f"Original: {original_count}, Final: {len(final_chunks)} "
                   f"(Removed: {total_removed})")
        
        # Log refinement stats
        stats = self.refinement_engine.stats
        logger.info(f"📊 Refinement stats - Contextual: {stats.contextual_refined}, "
                   f"Individual: {stats.individual_refined}, "
                   f"Failed: {stats.failed_refinements}, "
                   f"Total time: {stats.total_processing_time:.2f}s")

        return final_chunks


# Convenience functions to maintain backward compatibility
def generate_embeddings_with_ollama(texts: List[str], 
                                   ollama_client: OllamaApiClient,
                                   metrics_collector: MetricsCollector) -> Tuple[List[List[float]], int]:
    """Backward compatibility wrapper"""
    processor = EmbeddingProcessor(ollama_client, metrics_collector)
    return processor.generate_embeddings_with_ollama(texts)


def deduplicate_chunks_by_similarity(chunks: List[Dict[str, Any]], 
                                   ollama_client: OllamaApiClient,
                                   metrics_collector: MetricsCollector) -> List[Dict[str, Any]]:
    """Backward compatibility wrapper"""
    processor = EmbeddingProcessor(ollama_client, metrics_collector)
    return processor.deduplicate_chunks_by_similarity(chunks)