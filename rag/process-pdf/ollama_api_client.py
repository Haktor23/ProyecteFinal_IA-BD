import requests
import base64
import logging
from tqdm import tqdm
from typing import List, Dict, Any
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry # Asegúrate de importar Retry
from config_loader import Config

logger = logging.getLogger(__name__)

class OllamaApiClient:
    def __init__(self, base_url: str = Config.OLLAMA_URL, 
                 num_retries: int = Config.OLLAMA_NUM_RETRIES, # Usar desde Config
                 backoff_factor: float = Config.OLLAMA_RETRY_BACKOFF): # Usar desde Config
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()

        # Estrategia de Reintentos
        retry_strategy = Retry(
            total=num_retries,
            status_forcelist=[429, 500, 502, 503, 504], # Códigos HTTP para reintentar
            allowed_methods=["POST", "GET"], # Métodos HTTP permitidos para reintentar
            backoff_factor=backoff_factor, # Factor de espera entre reintentos (ej. 0s, 1s, 2s para 0.5)
            respect_retry_after_header=True
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter) # Si usas HTTPS para Ollama

        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })

    def health_check(self) -> bool:
        try:
            # Usar un timeout corto para el health check, no sujeto a reintentos largos
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def generate_embedding(self, text: str, model: str) -> List[float]:
        try:
            payload = {"model": model, "prompt": text}
            # El timeout aquí es por intento. La estrategia de reintentos manejará el total.
            response = self.session.post(f"{self.base_url}/api/embeddings", json=payload, timeout=45)
            response.raise_for_status()
            embedding = response.json().get('embedding', [])
            if not Config.EMBEDDING_DIM and embedding:
                Config.EMBEDDING_DIM = len(embedding)
                logger.info(f"📐 Dimensión de embedding (Ollama: {model}) auto-detectada: {Config.EMBEDDING_DIM}")
            return embedding
        except requests.Timeout:
            logger.error(f"Timeout (después de reintentos) generando embedding con Ollama para modelo {model}. Prompt: '{text[:50]}...'")
            raise Exception(f"Ollama API timeout (después de reintentos) para modelo {model}")
        except requests.HTTPError as http_err:
            logger.error(f"HTTP error (después de reintentos) generando embedding con Ollama modelo {model}: {http_err} - {response.text if response else 'N/A'}")
            raise Exception(f"Ollama API HTTP error (después de reintentos) para modelo {model}: {response.text if response else 'N/A'}")
        except Exception as e:
            logger.error(f"Error (después de reintentos) generando embedding con Ollama modelo {model}: {str(e)}")
            raise Exception(f"Fallo al generar embedding (después de reintentos) con Ollama modelo {model}: {str(e)}")

    def generate_embeddings_batch(self, texts: List[str], model: str) -> List[List[float]]:
        embeddings = []
        for text in tqdm(texts, desc=f"Generando Embeddings (Ollama {model})", unit="text", disable=len(texts)<5):
            try:
                embedding = self.generate_embedding(text, model) # Ya usa la sesión con reintentos
                embeddings.append(embedding)
            except Exception: 
                logger.warning(f"Usando placeholder para embedding fallido. Texto: '{text[:50]}...'")
                embeddings.append([0.0] * Config.EMBEDDING_DIM if Config.EMBEDDING_DIM else [])
        return embeddings

    def describe_image(self, image_bytes: bytes, model: str, custom_prompt: str = None) -> str:
        # Esta función también usará la sesión con reintentos
        try:
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            prompt = custom_prompt or (
                "Describe esta imagen en detalle, enfocándote en elementos técnicos, gráficos, "
                "tablas o diagramas. Si no hay elementos significativos o decorativos, indícalo así. Sé conciso."
            )
            payload = {
                "model": model, "prompt": prompt, "images": [image_b64], 
                "stream": False, "options": {"temperature": 0.2, "num_ctx": 2048}
            }
            response = self.session.post(f"{self.base_url}/api/generate", json=payload, timeout=90) # Timeout por intento
            response.raise_for_status()
            return response.json().get('response', '').strip()
        # ... (manejo de errores similar a generate_embedding, indicando "después de reintentos") ...
        except requests.Timeout:
            logger.error(f"Timeout (después de reintentos) describiendo imagen con Ollama modelo {model}.")
            return "" 
        except requests.HTTPError as http_err:
            logger.error(f"HTTP error (después de reintentos) describiendo imagen con Ollama modelo {model}: {http_err} - {response.text if response else 'N/A'}")
            return ""
        except Exception as e:
            logger.error(f"Error (después de reintentos) describiendo imagen con Ollama modelo {model}: {str(e)}")
            return ""
        
    def generate_text(self, prompt: str, model: str, temperature: float = 0.2, timeout: int = 45) -> str:
        """Generates text using the specified Ollama model."""
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature, "num_ctx": 4096} # Adjust num_ctx as needed
            }
            response = self.session.post(f"{self.base_url}/api/generate", json=payload, timeout=timeout)
            response.raise_for_status()
            generated_text = response.json().get('response', '').strip()
            # Clean common LLM quoting artifacts
            if generated_text.startswith('"') and generated_text.endswith('"'):
                generated_text = generated_text[1:-1]
            if generated_text.startswith("'") and generated_text.endswith("'"):
                generated_text = generated_text[1:-1]
            return generated_text
        except requests.Timeout:
            logger.error(f"Timeout generating text with Ollama model {model}.")
            return ""
        except requests.HTTPError as http_err:
            logger.error(f"HTTP error generating text with Ollama model {model}: {http_err} - {response.text if response else 'N/A'}")
            return ""
        except Exception as e:
            logger.error(f"Error generating text with Ollama model {model}: {str(e)}")
            return ""
        