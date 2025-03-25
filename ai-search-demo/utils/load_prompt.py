import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_system_prompt(prompt_template="./prompts/retrieval_agent_prompt.txt"):
    # Read system prompt from file
    try:
        with open(prompt_template, "r") as file:
            system_prompt = file.read()
    except FileNotFoundError:
        logger.warning(f"{prompt_template} not found. Using default system message.")
        system_prompt = "You are a helpful assistant."
    return system_prompt
