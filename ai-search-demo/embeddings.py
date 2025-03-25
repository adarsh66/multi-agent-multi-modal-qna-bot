from azure.ai.inference import EmbeddingsClient
from azure.ai.inference import ImageEmbeddingsClient
from azure.ai.inference.models import ImageEmbeddingInput, EmbeddingInputType
from azure.core.credentials import AzureKeyCredential
import numpy as np
import os
import logging
from dotenv import load_dotenv

# Initialize logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler("app.log")
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

load_dotenv()

embedding_api_key = os.getenv("AZURE_MAAS_API_KEY")
embedding_endpoint = os.getenv("AZURE_MAAS_ENDPOINT")
text_embedding_client = EmbeddingsClient(
    endpoint=embedding_endpoint, credential=AzureKeyCredential(embedding_api_key)
)
image_embedding_client = ImageEmbeddingsClient(
    endpoint=embedding_endpoint,
    credential=AzureKeyCredential(embedding_api_key),
    input_type=EmbeddingInputType.TEXT,
)


def generate_query_embedding(query):
    logger.info("Generating embeddings for the query.")
    query_embedding = text_embedding_client.embed(input=[query]).data[0].embedding
    logger.info("Query embedding generated successfully.")
    return query_embedding


def chunk_text(text, chunk_size=512):
    words = text.split()
    chunks = [
        " ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)
    ]
    return chunks


def generate_embeddings(text, tables, images):
    logger.info("Generating embeddings for texts, tables, and images.")

    # Chunk the text before generating embeddings
    text_chunks = chunk_text(text)

    # Generate embeddings for text chunks
    text_embeddings = []
    logger.info("Generating embeddings for text chunks. %s", text_chunks)
    _embedding = text_embedding_client.embed(input=text_chunks)
    for e in _embedding.data:
        text_embeddings.append(e.embedding)
    logging.info(
        "Text embeddings generated successfully. %s", len(_embedding.data[0].embedding)
    )

    # Generate embeddings for tables
    table_embeddings = []
    for table in tables:
        _embedding = text_embedding_client.embed(input=[table])
        table_embeddings.append(_embedding.data[0].embedding)
    logging.info("Table embeddings generated successfully. %s", len(table_embeddings))

    # Generate embeddings for images
    image_embeddings = []
    for image in images:
        _embedding = image_embedding_client.embed(
            input=[ImageEmbeddingInput.load(image_file=image, image_format="jpg")]
        )
        # logging.info("Image embedding: %s", _embedding.data[0].embedding)
        image_embeddings.append(_embedding.data[0].embedding)
    logging.info("Image embeddings generated successfully.")

    logger.info("Embeddings generated successfully.")
    return text_embeddings, text_chunks, table_embeddings, image_embeddings


def save_embeddings_to_index(doc_name, text, tables, images, index_client):
    logger.info("Generating embeddings for text, tables, and images.")
    text_embeddings, text_chunks, table_embeddings, image_embeddings = (
        generate_embeddings(text, tables, images)
    )
    logger.info("Ingesting embeddings into the Azure AI search index.")
    try:
        docs = []
        # Prepare text embeddings documents
        for i, text_emb in enumerate(text_embeddings):
            docs.append(
                {
                    "id": f"text_{i}_{doc_name}",
                    "embedding": text_emb,
                    "type": "text",
                    "content": text_chunks[i],
                }
            )
        # Prepare table embeddings documents
        for i, table_emb in enumerate(table_embeddings):
            docs.append(
                {
                    "id": f"table_{i}_{doc_name}",
                    "embedding": table_emb,
                    "type": "table",
                    "content": tables[i],
                }
            )
        # Prepare image embeddings documents
        for i, image_emb in enumerate(image_embeddings):
            docs.append(
                {
                    "id": f"image_{i}_{doc_name}",
                    "embedding": image_emb,
                    "type": "image",
                    "content": images[i],
                }
            )
        index_client.ingest_embeddings(documents=docs)
        logger.info("Embeddings ingested successfully.")
    except Exception as e:
        logger.error(f"Error ingesting embeddings: {e}")
        raise


def load_embeddings_from_index(index_client, query):
    logger.info("Querying the Azure AI search index.")
    try:
        results = index_client.query(query)[:5]
        logger.info("Query executed successfully.")
        return [r["content"] for r in results]
    except Exception as e:
        logger.error(f"Error querying the index: {e}")
        raise
