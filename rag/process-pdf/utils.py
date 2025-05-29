# utils.py
import os
import logging
from datetime import datetime
from typing import Dict, List, Any

try:
    import colorlog
    HAS_COLOR = True
except ImportError:
    HAS_COLOR = False

def setup_logging():
    """Configuración avanzada de logging con colores y múltiples handlers"""
    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger() # Root logger
    logger.setLevel(logging.DEBUG)
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    file_formatter = logging.Formatter(
        '%(asctime)s | %(name)-20s | %(levelname)-8s | %(funcName)-15s:%(lineno)-4d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    if HAS_COLOR:
        console_formatter = colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s | %(name)-12s | %(levelname)-8s | %(reset)s%(message)s',
            datefmt='%H:%M:%S',
            log_colors={
                'DEBUG': 'cyan', 'INFO': 'green', 'WARNING': 'yellow',
                'ERROR': 'red', 'CRITICAL': 'red,bg_white',
            }
        )
    else:
        console_formatter = logging.Formatter(
            '%(asctime)s | %(name)-12s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
    
    file_handler = logging.FileHandler(f"logs/rag_system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    error_handler = logging.FileHandler("logs/errors.log", encoding='utf-8')
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(error_handler)
    logger.addHandler(console_handler)
    
    return logger

class MetricsCollector:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.metrics = {
            'processing_times': {},
            'chunk_stats': {'total_chunks_raw': 0, 'total_chunks_final': 0, 'duplicate_chunks_removed': 0},
            'pdf_stats': {'total_pdfs_found': 0, 'successful_pdfs_processed': 0, 'failed_pdfs_processed': 0,
                          'total_pages_extracted': 0, 'total_images_found': 0, 'total_images_described': 0},
            'elasticsearch_stats': {'successful_indexes': 0, 'failed_indexes': 0, 'batch_times': []},
            'image_processing_stats': {'total_time': 0.0, 'count': 0},
            'errors': []
        }
    
    def log_processing_time(self, operation: str, duration: float):
        self.metrics['processing_times'].setdefault(operation, []).append(duration)
    
    def log_error(self, error_type: str, error_msg: str, context: str = ""):
        self.metrics['errors'].append({
            'timestamp': datetime.now().isoformat(), 'type': error_type,
            'message': error_msg, 'context': context
        })
    
    def get_summary(self) -> Dict[str, Any]:
        summary = {}
        for op, times in self.metrics['processing_times'].items():
            if times:
                summary[f'{op}_avg_time_s'] = sum(times) / len(times)
                summary[f'{op}_total_time_s'] = sum(times)
                summary[f'{op}_count'] = len(times)
        
        summary.update(self.metrics['chunk_stats'])
        summary.update(self.metrics['pdf_stats'])
        summary.update(self.metrics['elasticsearch_stats'])

        if self.metrics['image_processing_stats']['count'] > 0:
            summary['image_desc_avg_time_s'] = self.metrics['image_processing_stats']['total_time'] / self.metrics['image_processing_stats']['count']
        
        summary['total_errors_logged'] = len(self.metrics['errors'])
        return summary