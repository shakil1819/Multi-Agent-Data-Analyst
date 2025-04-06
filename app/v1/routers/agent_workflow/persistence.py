import json
import os
import uuid

from typing import Any, Optional  # noqa: UP035

import chromadb

from chromadb.config import Settings
from pydantic_graph.persistence import BaseStatePersistence


def get_chroma_client(persist_directory: Optional[str] = None) -> chromadb.Client:
    """Get or create a ChromaDB client.

    Args:
        persist_directory: Optional directory to persist the ChromaDB data.
            If not provided, an in-memory client will be used.

    Returns:
        chromadb.Client: A ChromaDB client instance

    """
    if persist_directory:
        # Create the directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)

        # Return a persistent client
        return chromadb.PersistentClient(path=persist_directory, settings=Settings(anonymized_telemetry=False))
    # Return an in-memory client
    return chromadb.Client(Settings(anonymized_telemetry=False, is_persistent=False))


class ChromaDBStatePersistence(BaseStatePersistence):
    """Custom implementation of state persistence using ChromaDB."""

    def __init__(
        self,
        chroma_client: chromadb.Client,
        collection_name: str = "graph_states",
    ):
        """Initialize the ChromaDB state persistence.

        Args:
            chroma_client: ChromaDB client instance
            collection_name: Name of the collection to store states

        """
        self.client = chroma_client
        self.collection = self.client.get_or_create_collection(collection_name)

    async def save_state(self, state_id: str, state: dict[str, Any], metadata: Optional[dict[str, Any]] = None) -> None:
        """Save a state to ChromaDB.

        Args:
            state_id: Unique identifier for the state
            state: The state to save
            metadata: Optional metadata to store with the state

        """
        # Convert state to JSON string
        state_json = json.dumps(state)

        # Prepare metadata
        meta = metadata or {}
        meta["state_id"] = state_id

        # Use a unique document ID
        doc_id = f"{state_id}_{uuid.uuid4()}"

        # Add the state to ChromaDB
        self.collection.add(documents=[state_json], metadatas=[meta], ids=[doc_id])

    async def load_state(self, state_id: str) -> Optional[dict[str, Any]]:
        """Load a state from ChromaDB.

        Args:
            state_id: Unique identifier for the state

        Returns:
            The state if found, None otherwise

        """
        # Query ChromaDB for the state
        results = self.collection.query(where={"state_id": state_id}, limit=1)

        # Return the state if found
        if results["documents"] and results["documents"][0]:
            return json.loads(results["documents"][0])

        return None

    async def delete_state(self, state_id: str) -> None:
        """Delete a state from ChromaDB.

        Args:
            state_id: Unique identifier for the state

        """
        # Find all documents with the given state_id
        results = self.collection.query(where={"state_id": state_id}, limit=100)

        # Delete the documents if found
        if results["ids"]:
            self.collection.delete(ids=results["ids"])
