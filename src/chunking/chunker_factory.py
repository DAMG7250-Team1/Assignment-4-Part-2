"""
Factory class for creating and retrieving document chunking strategies.
"""

import logging
from typing import Dict, Any, Optional, List, Union, Type

# Import chunking strategies
from src.chunking.strategies.fixed_size import FixedSizeChunker
from src.chunking.strategies.sliding_window import SlidingWindowChunker
from src.chunking.strategies.semantic import SemanticChunker

logger = logging.getLogger(__name__)

class ChunkerFactory:
    """
    Factory class for creating document chunking strategies.
    
    This factory provides a unified interface for selecting and using
    different chunking strategies for text data.
    """
    
    AVAILABLE_CHUNKERS = {
        "fixed_size": FixedSizeChunker,
        "sliding_window": SlidingWindowChunker,
        "semantic": SemanticChunker
    }
    
    @classmethod
    def get_chunker(cls, strategy_name: str, **kwargs) -> Union[FixedSizeChunker, SlidingWindowChunker, SemanticChunker]:
        """
        Get a chunker instance based on the strategy name.
        
        Args:
            strategy_name: Name of the chunking strategy
            **kwargs: Additional parameters to pass to the chunker constructor
            
        Returns:
            An instance of the requested chunker
            
        Raises:
            ValueError: If the requested strategy is not available
        """
        strategy_name = strategy_name.lower()
        
        if strategy_name not in cls.AVAILABLE_CHUNKERS:
            available_strategies = ", ".join(cls.AVAILABLE_CHUNKERS.keys())
            raise ValueError(f"Unknown chunking strategy: '{strategy_name}'. "
                            f"Available strategies: {available_strategies}")
        
        chunker_class = cls.AVAILABLE_CHUNKERS[strategy_name]
        logger.info(f"Creating chunker with strategy '{strategy_name}'")
        
        try:
            return chunker_class(**kwargs)
        except Exception as e:
            logger.error(f"Error creating chunker with strategy '{strategy_name}': {str(e)}")
            raise
    
    @classmethod
    def get_available_strategies(cls) -> List[str]:
        """
        Get a list of available chunking strategies.
        
        Returns:
            List of available chunking strategy names
        """
        return list(cls.AVAILABLE_CHUNKERS.keys())
    
    @classmethod
    def chunk_text(cls, 
                  text: str, 
                  strategy_name: str = "fixed_size", 
                  metadata: Optional[Dict[str, Any]] = None,
                  **kwargs) -> List[Dict[str, Any]]:
        """
        Convenience method to chunk text with a specified strategy.
        
        Args:
            text: Text to chunk
            strategy_name: Name of the chunking strategy to use
            metadata: Optional metadata to include with each chunk
            **kwargs: Additional parameters for the chunker
            
        Returns:
            List of chunks with their metadata
        """
        chunker = cls.get_chunker(strategy_name, **kwargs)
        
        logger.info(f"Chunking text using '{strategy_name}' strategy")
        chunks = chunker.create_chunks(text, metadata)
        logger.info(f"Created {len(chunks)} chunks using '{strategy_name}' strategy")
        
        return chunks 