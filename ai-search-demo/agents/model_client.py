from azure.core.credentials import AzureKeyCredential
import os
from dotenv import load_dotenv
from autogen_ext.models.azure import AzureAIChatCompletionClient

load_dotenv()

GPT_4o = AzureAIChatCompletionClient(
    endpoint=os.getenv("AZURE_INFERENCE_ENDPOINT"),
    credential=AzureKeyCredential(os.getenv("AZURE_INFERENCE_KEY")),
    model="gpt-4o",
    model_info={
        "family": "gpt-4o",
        "vision": True,
        "function_calling": True,
        "json_output": True,
    },
)
