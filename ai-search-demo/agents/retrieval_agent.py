from search import AzureSearch
from embeddings import generate_query_embedding
from utils.load_prompt import get_system_prompt
from utils.image_handler import encode_image
import os
from io import BytesIO
from dotenv import load_dotenv
from agents.model_client import GPT_4o as model_client
from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ChatCompletionClient
from autogen_core.model_context import (
    UnboundedChatCompletionContext,
    BufferedChatCompletionContext,
)
from typing import AsyncGenerator, List, Sequence, Callable
from autogen_core import Image as AGImage
from PIL import Image
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import (
    AgentEvent,
    ChatMessage,
    TextMessage,
    MultiModalMessage,
    ToolCallSummaryMessage,
)
from autogen_core import CancellationToken
import json
import ast
from utils.logger import setup_logger

# Setup logger
logger = setup_logger(name="Cohere Multi model search demo")
load_dotenv()


def get_structured_output(content):
    # Convert the text content into a list of dictionaries
    # Try to parse as JSON first
    try:
        parsed_content = json.loads(content)
    except json.JSONDecodeError:
        # If JSON fails, try ast.literal_eval for Python dict-like strings
        try:
            parsed_content = ast.literal_eval(content)
        except (SyntaxError, ValueError):
            # If both fail, return the raw content
            parsed_content = content
    if not isinstance(parsed_content, list) and isinstance(parsed_content, dict):
        parsed_content = [parsed_content]
    return parsed_content


def get_search_client():
    """
    Initialize and return the any Search clien given the connection parameters
    In this case, we are returning an instance of the AzureSearch client.
    However, this can easily be modified to return other types of clients
    (e.g., Pinecone, Weaviate, etc.) based on the connection parameters.
    The connection parameters are expected to be set as environment variables.
    Ensure that the following environment variables are set:
        - AZURE_SEARCH_ENDPOINT
        - AZURE_SEARCH_INDEX_NAME
        - AZURE_SEARCH_API_KEY
    This function uses the AzureSearch class to create a client instance
    with the provided endpoint, index name, and API key.

    Returns:
        AzureSearch: An instance of the AzureSearch client.
    """
    azure_search = AzureSearch(
        endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
        index_name=os.getenv("AZURE_SEARCH_INDEX_NAME"),
        api_key=os.getenv("AZURE_SEARCH_API_KEY"),
    )
    return azure_search


def query_vector_search_db(query: str):
    """
    Query the vector search database with the given query.
    This function will generate the embeddings using the provided query
    and then use the search client to query the index with the generated embeddings.

    Args:
        query (str): The query string.

    Returns:
        List[Dict[str, Any]]: A list of documents matching the query.
    """
    # This function should be implemented to query your vector search database
    # and return the results.
    search_client = get_search_client()
    query_embedding = generate_query_embedding(query)
    context = search_client.query_index(query_embedding=query_embedding)

    return context


class MultiModalRetrievalAgent(BaseChatAgent):
    """
    Retrieval agent that retrieves relevant documents from a vector search database.
    It will return a MultiModalMessage with the retrieved documents.
    This agent is designed to work with multimodal data, including text, images, and tables.
    """

    def __init__(
        self,
        name: str,
        description: str,
        retrieval_agent: AssistantAgent,
    ) -> None:
        super().__init__(name=name, description=description)
        self._retrieval_agent = retrieval_agent

    @property
    def produced_message_types(self) -> Sequence[type[ChatMessage]]:
        return (TextMessage, MultiModalMessage)

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        await self._retrieval_agent.on_reset(cancellation_token=cancellation_token)

    async def on_messages(
        self,
        messages: Sequence[ChatMessage],
        cancellation_token: CancellationToken,
        # ) -> AsyncGenerator[AgentEvent | ChatMessage | Response, None]:
    ) -> Response:
        try:
            result = await self._retrieval_agent.run(
                task=messages, cancellation_token=cancellation_token
            )
            last_message = result.messages[-1]
            print("Last message:", last_message)
            img_list = []
            if isinstance(last_message, ToolCallSummaryMessage):
                content = last_message.content
                structured_output = get_structured_output(content)
                logger.info(
                    f"Structured output of retrieved content: {structured_output} of type {type(structured_output)}"
                )
                if isinstance(structured_output, list):
                    for item in structured_output:
                        if isinstance(item, dict):
                            if item["type"] == "image":
                                img_path = item["content"]
                                # Check if the image path is a URL or a local path
                                if img_path.startswith("http"):
                                    # If it's a URL, download the image
                                    img = AGImage(encode_image(img_path))
                                else:
                                    # If it's a local path, open the image
                                    if os.path.exists(img_path):
                                        # Open the image using PIL
                                        img = AGImage(Image.open(img_path))
                                    else:
                                        logger.warning(
                                            f"Image path {img_path} does not exist."
                                        )
                                        continue
                                img_list.append(img)
                logger.info(f"Image list size: {len(img_list)}")
                return Response(
                    chat_message=MultiModalMessage(
                        content=[content] + img_list,
                        source=self.name,  # last_message.source,
                        models_usage=last_message.models_usage,
                    ),
                    inner_messages=result.messages[len(messages) : -1],
                )
            else:
                content = "No relevant documents found."
                return Response(
                    chat_message=TextMessage(
                        content=content,
                        source=self.name,  # messages[0].source
                    ),
                    inner_messages=[],
                )
        except BaseException:
            content = "Error occured while processing MultiModalRetrievalAgent. Do Not call this agent again."
            raise
            return Response(
                chat_message=TextMessage(content=content, source=self.name),
                inner_messages=[],
            )


RETRIEVAL_AGENT = AssistantAgent(
    name="RetrievalAgent",
    description="Retrieval agent that retrieves relevant documents from a vector search database",
    model_client=model_client,
    tools=[query_vector_search_db],
    system_message=get_system_prompt(
        prompt_template="./prompts/retrieval_agent_prompt.txt"
    ),
    reflect_on_tool_use=False,
)

MULTI_MODAL_RETRIEVAL_AGENT = MultiModalRetrievalAgent(
    name="MultiModalRetrievalAgent",
    description="Retrieval agent that retrieves relevant documents from a vector search database",
    retrieval_agent=RETRIEVAL_AGENT,
)

RETRIEVAL_AGENT_WITH_REFLECTION = AssistantAgent(
    name="RetrievalAgent",
    description="Retrieval agent that retrieves relevant documents from a vector search database",
    model_client=model_client,
    tools=[query_vector_search_db],
    system_message=get_system_prompt(
        prompt_template="./prompts/retrieval_agent_prompt.txt"
    ),
    reflect_on_tool_use=True,
)
