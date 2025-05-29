# text_chunker.py
from typing import List

class SimpleTextSplitter:
    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 256):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # More comprehensive list of separators, ordered by preference
        self.separators = [
            "\n\n\n", "\n\n", "\n", ". ", "! ", "? ", "; ", ": ", 
            "。 ", "！ ", "？ ", "； ", "： ", 
            ", ", " ", ""
        ] 

    def split_text(self, text: str) -> List[str]:
        if not text or len(text.strip()) == 0 :
            return []
        if len(text) <= self.chunk_size:
            return [text.strip()]

        chunks = []
        current_pos = 0
        while current_pos < len(text):
            end_pos = min(current_pos + self.chunk_size, len(text))
            
            # If the remaining text is small enough, take it all
            if len(text) - current_pos <= self.chunk_size :
                chunk = text[current_pos:].strip()
                if chunk: chunks.append(chunk)
                break

            # Find the best split point by looking for separators from end_pos backwards
            split_pos = -1
            if end_pos < len(text): # Only search for split point if not at the very end
                for sep in self.separators:
                    # Search for separator in the part of the text that could be a valid end
                    # We search from (current_pos + chunk_overlap) up to end_pos
                    # to ensure the chunk is not too small and to respect overlap.
                    # A simpler approach is to search backwards from end_pos in the current prospective chunk.
                    search_start_boundary = current_pos # Maximize chance to find a separator
                    
                    sep_idx = text.rfind(sep, search_start_boundary, end_pos)
                    if sep_idx != -1 and sep_idx > current_pos : # Found a valid separator
                        split_pos = sep_idx + len(sep) # Split after the separator
                        break
            
            if split_pos == -1 or split_pos <= current_pos : # No good separator found, or found too early
                split_pos = end_pos # Force split at chunk_size

            chunk = text[current_pos:split_pos].strip()
            if chunk:
                chunks.append(chunk)
            
            # Next chunk starts after overlap from the previous chunk's start
            # but ensuring we make progress
            next_start = current_pos + self.chunk_size - self.chunk_overlap
            if next_start <= current_pos : # Ensure progress
                 next_start = current_pos +1 if len(chunk) > 0 else split_pos # If chunk was tiny or empty, jump to split_pos
            if next_start >= len(text) and len(text) > current_pos: # Avoid infinite loop if stuck
                 if current_pos < len(text): # If there's remaining text not covered by next_start
                     final_chunk = text[current_pos:].strip()
                     if final_chunk and (not chunks or chunks[-1] != final_chunk): # Avoid adding empty or duplicate final chunk
                         chunks.append(final_chunk)
                 break


            current_pos = next_start if next_start < split_pos else split_pos


        # Post-process to remove empty strings just in case
        return [c for c in chunks if c]