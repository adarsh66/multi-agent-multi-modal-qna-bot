from autogen_agentchat.agents import AssistantAgent
from agents.model_client import GPT_4o as model_client
from utils.load_prompt import get_system_prompt

ANSWER_AGENT = AssistantAgent(
    name="AnswerAgent",
    description="Answer agent that answers questions based on the retrieved documents",
    model_client=model_client,
    system_message=get_system_prompt(prompt_template="./prompts/answer_gen_prompt.txt"),
)
