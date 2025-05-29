import os
import json
import logging
from typing import List, Dict, Any
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch, helpers
import requests
from tqdm import tqdm
import gc
# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SemanticPDFProcessor:
    def __init__(
        self,
        pdf_folder: str = ".",
        output_folder: str = "output",
        es_host: str = "http://localhost:9200",
        es_index: str = "document_chunks",
        llm_api_url: str = " http://127.0.0.1:1234/v1/embeddings",  # URL de tu API Llama 3.2
        embedding_model_name: str = 'all-MiniLM-L6-v2',
        batch_size: int = 10

    ):
        """Inicializa el procesador de PDFs con chunking semántico"""
        # Directorios y configuración básica
        self.pdf_folder = pdf_folder
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Modelo de embeddings
        logger.info(f"Cargando modelo de embeddings: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Conexión a Elasticsearch
        logger.info(f"Conectando a Elasticsearch: {es_host}")
        self.es_index = es_index
        self.es = Elasticsearch(es_host)
        
        self._setup_elasticsearch()
       
        # Configuración de Llama 3.2
        self.llm_api_url = llm_api_url
        self.batch_size = batch_size

    def _setup_elasticsearch(self):
        """Configura el índice de Elasticsearch si no existe"""
        try:
            if not self.es.indices.exists(index=self.es_index):
                logger.info(f"Creando índice '{self.es_index}' en Elasticsearch")
                
                # Detectar dimensiones automáticamente desde el modelo
                dims = len(self.embedding_model.encode("test"))

                mapping = {
                    "mappings": {
                        "properties": {
                            "title": {"type": "text"},
                            "description": {"type": "text"},
                            "content": {"type": "text"},
                            "highlight": {"type": "text"},
                            "page_number": {"type": "integer"},
                            "document_name": {"type": "keyword"},
                            "embedding": {
                                "type": "dense_vector",
                                "dims": dims,
                                "index": True,
                                "similarity": "cosine"
                            }
                        }
                    },
                    "settings": {
                        "number_of_shards": 1,
                        "number_of_replicas": 0
                    }
                }

                self.es.indices.create(index=self.es_index, body=mapping)

        except Exception as e:
            logger.error(f"Error al configurar el índice en Elasticsearch: {e}")

    def llama_generate(self, prompt: str) -> str:
        """Genera texto usando la API de Llama 3.2"""
        try:
            payload = {
                "prompt": prompt,
                "max_tokens": 300,
                "temperature": 0.1,
                "top_p": 0.9,
                "stream": False
            }
            response = requests.post("http://localhost:1234/v1/chat/completions", json={
                "model": "meta-llama-3.1-8b-instruct",  # Reemplaza con el nombre del modelo cargado en LM Studio
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "top_p": 0.9
            })

            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()

        except Exception as e:
            logger.error(f"Error al generar texto con Llama 3.2: {e}")
            return f"Error de generación: {str(e)[:100]}..."

    def read_pdf_text_by_pages(self, path: str) -> List[Dict[str, Any]]:
        """Lee un PDF y devuelve una lista de páginas con su texto y número"""
        try:
            reader = PdfReader(path)
            pages = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ''
                if text.strip():  # Solo incluir páginas con texto
                    pages.append({"text": text, "page_number": i + 1})
            return pages
        except Exception as e:
            logger.error(f"Error al leer el PDF {path}: {e}")
            return []

    def semantic_chunking(self, page_text: str) -> List[str]:
        """Divide el texto de una página en chunks semánticos usando Llama 3.2"""
        prompt = f"""
REGLAS IMPORTANTES:
1. Usa EXACTAMENTE "**CHUNK :**" para marcar los límites de cada fragmento.
2. NO MODIFIQUES el texto original en absoluto - copia y pega exactamente.
3. Cada fragmento debe ser una parte TEXTUAL EXACTA del texto original.
4. No agregues texto explicativo, numeración, o cualquier contenido que no esté en el texto original.
5. Cada fragmento debe ser semánticamente completo (una idea, concepto o sección).
6. No cortes oraciones o párrafos a la mitad.

TEXTO A DIVIDIR:
{page_text}


TEXTO:
{page_text}

CHUNKS:
"""
        try:
            response = self.llama_generate(prompt)
            chunks = []
            for line in response.split("CHUNK:"):
                if line.strip():
                    chunks.append(line.strip())
            
            # Si Llama no devuelve chunks de manera adecuada, usar página completa
            if not chunks or len(chunks) == 1 and len(chunks[0]) < 50:
                logger.warning("Chunking semántico fallido, usando página completa")
                return [page_text]
                
            return chunks
        except Exception as e:
            logger.error(f"Error en chunking semántico: {e}")
            # Fallback: devolver la página como un solo chunk
            return [page_text]

    def process_chunk(self, chunk: str, filename: str, page_number: int, chunk_id: int) -> Dict[str, Any]:
        """Procesa un chunk individual para extraer insights y embeddings"""
        try:
            # Crear embedding
            embedding = self.embedding_model.encode(chunk).tolist()

            # Crear documento
            doc = {
                "content": chunk,
                "chunk": f"idChunk_{chunk_id}",
                "embedding": embedding,
                "page_number": page_number,
                "document_name": filename
            }

            # Guardar en JSON
            json_path = os.path.join(self.output_folder, f"{filename}_p{page_number}_c{chunk_id}.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(doc, f, ensure_ascii=False, indent=2)

            # Liberar memoria interna
            del embedding
            del json_path
            del chunk

            gc.collect()

            return doc

        except Exception as e:
            logger.error(f"Error al procesar chunk: {e}")
            return {}


    def process_documents(self):
        """Procesa todos los PDFs en la carpeta"""
        pdf_files = [f for f in os.listdir(self.pdf_folder) if f.lower().endswith(".pdf")]
        if not pdf_files:
            logger.warning(f"No se encontraron archivos PDF en {self.pdf_folder}")
            return
            
        logger.info(f"Encontrados {len(pdf_files)} archivos PDF para procesar")
        
        for filename in tqdm(pdf_files, desc="Procesando PDFs"):
            filepath = os.path.join(self.pdf_folder, filename)
            logger.info(f"Procesando {filepath}")
            
            # Leer PDF por páginas
            pages = self.read_pdf_text_by_pages(filepath)
            logger.info(f"Extraídas {len(pages)} páginas con texto")
            
            # Procesar cada página
            for page in tqdm(pages, desc=f"Páginas de {filename}"):
                page_text = page["text"]
                page_number = page["page_number"]
                
                # Chunking semántico usando Llama 3.2
                chunks = self.semantic_chunking(page_text)
                logger.info(f"Página {page_number}: Generados {len(chunks)} chunks semánticos")
                
                # Procesar cada chunk
                for i, chunk in enumerate(chunks):
                    if len(chunk.strip()) < 20:
                        continue
                        
                    self.process_chunk(chunk, filename, page_number, i)

                    # Libera memoria tras cada chunk
                    del chunk
                    gc.collect()
                
                # Libera memoria tras cada página
                del page_text, page, chunks
                gc.collect()
            
            # Libera memoria tras cada archivo PDF
            del pages
            gc.collect()

        logger.info("Procesamiento completado")



def main():
    """Función principal"""
    # Configuración
    config = {
        "pdf_folder": ".",  # Carpeta de PDFs
        "output_folder": "chunks_json",  # Carpeta de salida para los chunks en formato JSON
        "es_host": "http://localhost:9200",
        "es_index": "document_chunks",
        "llm_api_url": "http://127.0.0.1:1234/v1/embeddings",  # URL para la API de embeddings de Llama 3.2
        "embedding_model_name": 'all-MiniLM-L6-v2'
    }
    
    # Crear y ejecutar procesador
    processor = SemanticPDFProcessor(**config)
    processor.process_documents()


if __name__ == "__main__":
    main()
