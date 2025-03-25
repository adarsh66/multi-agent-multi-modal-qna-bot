from autogen_agentchat.agents import AssistantAgent
from agents.model_client import GPT_4o as model_client
from utils.load_prompt import get_system_prompt

VALIDATION_AGENT = AssistantAgent(
    name="ValidationAgent",
    description="Validate the answer or rewrite the query to get a better answer",
    model_client=model_client,
    system_message=get_system_prompt(
        prompt_template="./prompts/validation_agent_prompt.txt"
    ),
)
