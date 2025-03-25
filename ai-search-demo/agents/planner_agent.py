from agents.model_client import GPT_4o as model_client
from autogen_agentchat.agents import AssistantAgent
from utils.load_prompt import get_system_prompt

PLANNER_AGENT = AssistantAgent(
    name="PlanningAgent",
    description="Planning agent that breaks down tasks into subtasks",
    model_client=model_client,
    system_message=get_system_prompt("./prompts/planner_prompt.txt"),
)
