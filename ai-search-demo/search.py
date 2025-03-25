from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
import os
import json
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


class AzureSearch:
    def __init__(self, endpoint, index_name, api_key):
        self.endpoint = endpoint
        self.index_name = index_name
        self.api_key = api_key
        self.client = SearchClient(
            endpoint=self.endpoint,
            index_name=self.index_name,
            credential=AzureKeyCredential(self.api_key),
        )
        logger.info("Azure Search client initialized.")

    def ingest_embeddings(self, documents):
        try:
            logger.info(
                "Ingesting embeddings into Azure Search. Length: %s", len(documents)
            )
            self.client.upload_documents(documents=documents)
            logger.info("Embeddings ingested successfully.")
        except Exception as e:
            logger.error(f"Error ingesting embeddings: {e}")

    def query_index(self, query_embedding):
        try:
            logger.info(
                f"Querying Azure Search index with query: {len(query_embedding)}"
            )
            vector_query = VectorizedQuery(
                vector=query_embedding, k_nearest_neighbors=5, fields="embedding"
            )
            results = self.client.search(
                search_text="*",
                top=5,
                vector_queries=[vector_query],
                select="id,type,content",
                # filter="type eq 'image'",
            )
            # logging.info("Results keys. %s", results[0])
            response = [doc for doc in results]
            # response = [(doc["id"], doc["content"]) for doc in response]
            logger.info("Query executed successfully.")
            return response
        except Exception as e:
            logger.error(f"Error querying index: {e}")
            return []

    def clear_index(self):
        try:
            logger.info("Clearing all documents in the index.")
            results = self.client.search(search_text="*", top=1000, select="id")
            docs_to_delete = []
            for doc in results:
                docs_to_delete.append(
                    {
                        "id": doc["id"],
                        # "@search.action": "delete",
                    }
                )
            if docs_to_delete:
                self.client.delete_documents(documents=docs_to_delete)
            logger.info("Index cleared successfully.")
        except Exception as e:
            logger.error(f"Error clearing the index: {e}")


# Example usage
if __name__ == "__main__":
    endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
    api_key = os.getenv("AZURE_SEARCH_API_KEY")

    azure_search = AzureSearch(endpoint, index_name, api_key)
    # Add further functionality as needed for testing or integration.
