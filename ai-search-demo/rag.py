import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import (
    SystemMessage,
    UserMessage,
    TextContentItem,
    ImageContentItem,
    ImageUrl,
)
from azure.core.credentials import AzureKeyCredential
import base64
from dotenv import load_dotenv
import logging

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure your Azure Form Recognizer endpoint and key
endpoint = os.getenv("AZURE_INFERENCE_ENDPOINT")
inference_key = os.getenv("AZURE_INFERENCE_KEY")
MODEL_NAME = "gpt-4o"

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(inference_key),
)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string


def prepare_rag_context(question, search_context):
    text_context = "Text Context: "
    table_context = "Table Context: "
    context_item_list = []
    for msg in search_context:
        if msg["type"] == "text":
            text_context += msg["content"]
            text_context += "\n\n"
        elif msg["type"] == "table":
            table_context += msg["content"]
            table_context += "\n\n"
        elif msg["type"] == "image":
            encoded_image = encode_image(msg["content"])
            image_url = ImageUrl(url=f"data:image/jpeg;base64,{encoded_image}")
            image_context = ImageContentItem(image_url=image_url)
            context_item_list.append(image_context)

    context_prompt = f"""
        #  **Provided  Context**
        {text_context}

        {table_context}

        # **Question**
        Question: "{question}"

        # **Answer**
        """
    text_context_item = TextContentItem(text=context_prompt)
    context_item_list.append(text_context_item)
    return context_item_list


def get_system_prompt(prompt_template="answer_gen_prompt.txt"):
    # Read system prompt from file
    try:
        with open(prompt_template, "r") as file:
            system_prompt = file.read()
    except FileNotFoundError:
        logger.warning(f"{prompt_template} not found. Using default system message.")
        system_prompt = "You are a helpful assistant."
    return system_prompt


def generate_answer(
    question,
    search_context,
    model_name=MODEL_NAME,
    stream=False,
    max_tokens=4096,
    temperature=1.0,
    top_p=1.0,
):
    rag_context = prepare_rag_context(question, search_context)
    response = client.complete(
        messages=[
            SystemMessage(content=get_system_prompt()),
            UserMessage(content=rag_context),
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        model=model_name,
        stream=stream,
    )
    if stream:
        for update in response:
            if update.choices:
                return update.choices[0].delta.content
    else:
        return response.choices[0].message.content
