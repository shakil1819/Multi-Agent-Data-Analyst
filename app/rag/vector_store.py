"""
Pinecone Vector Store Integration for RAG System
"""
import os
from typing import Dict, List, Optional, Any

import pinecone
from pinecone.core.client.configuration import Configuration as PineconeConfiguration
from pinecone.core.client.models import QueryResponse

class PineconeVectorStore:
    """
    Vector store implementation using Pinecone 5.4.2
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        environment: str = "gcp-starter",
        index_name: str = "rag-index",
        namespace: str = "default",
        dimension: int = 1536,  # Default for many embedding models
        metric: str = "cosine",
    ):
        """
        Initialize Pinecone vector store
        
        Args:
            api_key: Pinecone API key (defaults to PINECONE_API_KEY env var)
            environment: Pinecone environment
            index_name: Name of the Pinecone index
            namespace: Namespace within the index
            dimension: Dimension of the embedding vectors
            metric: Distance metric for similarity search
        """
        self.api_key = api_key or os.environ.get("PINECONE_API_KEY")
        if not self.api_key:
            raise ValueError("Pinecone API key is required")
            
        self.environment = environment
        self.index_name = index_name
        self.namespace = namespace
        self.dimension = dimension
        self.metric = metric
        
        # Initialize Pinecone
        pc_config = PineconeConfiguration(
            api_key=self.api_key,
            host=f"https://controller.{self.environment}.pinecone.io"
        )
        pinecone.init(api_key=self.api_key, environment=self.environment)
        
        # Create index if it doesn't exist
        self._create_index_if_not_exists()
        
        # Connect to the index
        self.index = pinecone.Index(self.index_name)
    
    def _create_index_if_not_exists(self) -> None:
        """Create Pinecone index if it doesn't exist"""
        existing_indexes = pinecone.list_indexes()
        
        if self.index_name not in existing_indexes:
            pinecone.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric=self.metric
            )
            print(f"Created new Pinecone index: {self.index_name}")
    
    def upsert(
        self, 
        vectors: List[tuple],
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """
        Upsert vectors to Pinecone
        
        Args:
            vectors: List of (id, vector, metadata) tuples
            batch_size: Batch size for upserting
            
        Returns:
            Response from Pinecone
        """
        # Process vectors in batches
        total_vectors = len(vectors)
        results = {}
        
        for i in range(0, total_vectors, batch_size):
            batch = vectors[i:i + batch_size]
            
            # Format for Pinecone
            ids = [item[0] for item in batch]
            embeddings = [item[1] for item in batch]
            metadatas = [item[2] for item in batch]
            
            # Upsert to Pinecone
            result = self.index.upsert(
                vectors=zip(ids, embeddings, metadatas),
                namespace=self.namespace
            )
            
            results[f"batch_{i//batch_size}"] = result
            
        return results
    
    def query(
        self,
        query_vector: List[float],
        top_k: int = 5,
        include_metadata: bool = True,
        filter: Optional[Dict[str, Any]] = None
    ) -> QueryResponse:
        """
        Query Pinecone index
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            include_metadata: Whether to include metadata in results
            filter: Filter to apply to the query
            
        Returns:
            Query response from Pinecone
        """
        return self.index.query(
            namespace=self.namespace,
            vector=query_vector,
            top_k=top_k,
            include_metadata=include_metadata,
            filter=filter
        )
    
    def delete(
        self,
        ids: Optional[List[str]] = None,
        delete_all: bool = False,
        filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Delete vectors from Pinecone
        
        Args:
            ids: List of vector IDs to delete
            delete_all: Whether to delete all vectors
            filter: Filter to apply to deletion
            
        Returns:
            Response from Pinecone
        """
        if delete_all:
            return self.index.delete(
                delete_all=True,
                namespace=self.namespace
            )
        elif ids:
            return self.index.delete(
                ids=ids,
                namespace=self.namespace
            )
        elif filter:
            return self.index.delete(
                filter=filter,
                namespace=self.namespace
            )
        else:
            raise ValueError("Either ids, delete_all, or filter must be specified")