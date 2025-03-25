from typing import List, AsyncGenerator, Any
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import (
    MaxMessageTermination,
    TextMentionTermination,
    HandoffTermination,
)
from autogen_core import CancellationToken
from autogen_agentchat.teams import Swarm, SelectorGroupChat, RoundRobinGroupChat
from autogen_agentchat.ui import Console
from agents.retrieval_agent import (
    RETRIEVAL_AGENT,
    MULTI_MODAL_RETRIEVAL_AGENT,
    RETRIEVAL_AGENT_WITH_REFLECTION,
)
from autogen_agentchat.messages import TextMessage
from agents.answer_agent import ANSWER_AGENT
from agents.planner_agent import PLANNER_AGENT
from agents.validation_agent import VALIDATION_AGENT
from agents.model_client import GPT_4o as model_client
from utils.load_prompt import get_system_prompt
import logging

from autogen_agentchat import EVENT_LOGGER_NAME, TRACE_LOGGER_NAME

# For structured message logging, such as low-level messages between agents.
event_logger = logging.getLogger(EVENT_LOGGER_NAME)
event_logger.addHandler(logging.StreamHandler())
event_logger.setLevel(logging.WARN)
logging.getLogger().setLevel(logging.ERROR)


def get_selector_team(team: List[AssistantAgent], max_turns=5) -> SelectorGroupChat:
    """
    Create a team of agents for the SelectorGroupChat.
    """
    selector_prompt = get_system_prompt(
        prompt_template="./prompts/selector_team_prompt.txt"
    )
    text_mention_termination = TextMentionTermination("TERMINATE")
    max_messages_termination = MaxMessageTermination(max_messages=max_turns)
    termination = text_mention_termination | max_messages_termination

    team = SelectorGroupChat(
        participants=team,
        model_client=model_client,
        termination_condition=termination,
        selector_prompt=selector_prompt,
        allow_repeated_speaker=False,  # Allow an agent to speak multiple turns in a row.
        max_selector_attempts=2,  # Try to answer twice, then terminate.
    )

    return team


def get_round_robin_team(
    team: List[AssistantAgent], max_turns=5
) -> RoundRobinGroupChat:
    """
    Create a team of agents for the RoundRobinGroupChat.
    """
    text_mention_termination = TextMentionTermination("TERMINATE")
    termination = text_mention_termination

    team = RoundRobinGroupChat(
        participants=team,
        termination_condition=termination,
        max_turns=max_turns,
    )

    return team


def get_swarm_team(team: List[AssistantAgent], max_turns=10) -> Swarm:
    """
    Create a team of agents for the Swarm.
    """
    text_mention_termination = TextMentionTermination("TERMINATE")
    hand_off_termination = HandoffTermination(target="user")
    termination = text_mention_termination | hand_off_termination

    team = Swarm(
        participants=team,
        termination_condition=termination,
        max_turns=max_turns,
    )

    return team


async def run_agent_team(task: str, team_type="selector") -> AsyncGenerator[Any, None]:
    """
    Run the team of agents with the given task.
    """
    # Create a team of agents
    agents = [
        PLANNER_AGENT,
        MULTI_MODAL_RETRIEVAL_AGENT,
        ANSWER_AGENT,
        VALIDATION_AGENT,
    ]
    # Create a team of agents based on the specified type
    if team_type == "selector":
        team = get_selector_team(agents)
    elif team_type == "round_robin":
        team = get_round_robin_team([MULTI_MODAL_RETRIEVAL_AGENT, ANSWER_AGENT])
    elif team_type == "swarm":
        team = get_swarm_team(agents)
    else:
        raise ValueError(f"Unknown team type: {team_type}")

    # Run the team with the given task
    await team.reset()
    # response = team.run_stream(task=task)
    # yield response
    # result = await Console(team.run_stream(task=task), output_stats=True)
    async for message in team.run_stream(task=task):
        yield message


async def run_single_agent(task: str) -> AsyncGenerator[Any, None]:
    """
    Run a single agent with the given task.
    """
    # Run the agent with the given task
    agent = RETRIEVAL_AGENT_WITH_REFLECTION
    response = agent.run_stream(task=task)

    yield response


if __name__ == "__main__":
    # Example usage
    asyncio.run(
        run_agent_team(
            # task="Whats the price for a double room in Japan?", team_type="selector"
            # task="How many onsens do you observe in the images in Sakura Onsen?",
            task="How do i filter for green accommodations on the app",
            team_type="selector",
        )
    )
