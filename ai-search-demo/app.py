import streamlit as st
import os
from extractor import extract_document_content
from embeddings import save_embeddings_to_index, generate_query_embedding
from search import AzureSearch
from agentic_rag import run_agent_team, run_single_agent
from rag import generate_answer
from utils.logger import setup_logger
from dotenv import load_dotenv
import asyncio
from autogen_agentchat.base import TaskResult
from autogen_core.models import RequestUsage
from autogen_agentchat.messages import TextMessage, MultiModalMessage
import io
from PIL import Image


# Setup logger
logger = setup_logger(name="Cohere Multi model search demo")

# Load environment variables
load_dotenv()
endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
api_key = os.getenv("AZURE_SEARCH_API_KEY")
azure_search = AzureSearch(endpoint, index_name, api_key)
# Streamlit app title
st.title("AI Search Demo")

# Sidebar navigation
selection = st.sidebar.radio("Navigation", ["Manage Documents", "Search"])

if selection == "Manage Documents":
    st.markdown("## Manage Documents")
    extract_button = st.button("Extract Text and Images")
    embeddings_button = st.button("Generate Embeddings")
    clear_index_button = st.button("Clear Index")

    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file is not None:
        # Save the uploaded file to local storage
        file_path = os.path.join("uploads", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("File uploaded successfully!")
        doc_name = file_path.split("/")[-1].split(".")[0]

        if extract_button:
            with st.spinner("Extracting text and images..."):
                try:
                    # After successful extraction...
                    text, tables, images = extract_document_content(
                        file_path, doc_name=doc_name
                    )
                    st.session_state["extracted_text"] = text
                    st.session_state["extracted_tables"] = tables
                    st.session_state["extracted_images"] = images
                    if text is None:
                        raise Exception("Extraction failed.")
                    st.success("Extraction complete!")
                    st.markdown("### Extracted Text:")
                    st.write(text)
                    # Display tables if needed (omitted here for brevity)
                    if tables:
                        st.markdown("### Extracted Tables:")
                        for table in tables:
                            st.markdown(table)
                    if images:
                        st.markdown("### Extracted Images:")
                        for img in images:
                            st.image(img, caption="Extracted Image")
                except Exception as e:
                    logger.error(f"Error during extraction: {e}")
                    st.error("Error during extraction.")

        if embeddings_button:
            with st.spinner("Generating embeddings..."):
                try:
                    text = st.session_state.get("extracted_text")
                    tables = st.session_state.get("extracted_tables")
                    images = st.session_state.get("extracted_images")
                    if not text:
                        raise Exception(
                            "No text available. Please extract the document first."
                        )
                    save_embeddings_to_index(
                        doc_name, text, tables, images, azure_search
                    )
                    st.success("Embeddings generated and ingested successfully!")
                except Exception as e:
                    logger.error(f"Error during embedding generation: {e}")
                    st.error("Error during embedding generation.")

    if clear_index_button:
        with st.spinner("Clearing the index..."):
            try:
                # Initialize a new AzureSearch client instance
                azure_search.clear_index()
                st.success("Index cleared successfully!")
            except Exception as e:
                logger.error(f"Error during clearing the index: {e}")
                st.error("Error during clearing the index.")

elif selection == "Search":
    # Chat window for querying the AI Search index
    st.markdown("### Query the AI Search Index")
    team_config = st.selectbox(
        "Select Team Configuration",
        ["selector", "round_robin", "swarm", "single_agent"],
    )

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input for query
    if prompt := st.chat_input(
        "Enter your query",
    ):
        # Append user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process the query and get the response
        with st.spinner("Querying the AI search index..."):
            # try:
            with st.chat_message("assistant"):
                with st.expander("Agent thought process", expanded=False):
                    if team_config == "single_agent":
                        # Run a single agent
                        stream = run_single_agent(task=prompt)
                    else:
                        stream = run_agent_team(task=prompt, team_type=team_config)
                    response = st.write_stream(stream)

    # user_query = st.text_input("Enter your query:")
    # if st.button("Submit Query"):
    #     with st.spinner("Querying the AI search index..."):
    #         try:
    #             # query_embedding = generate_query_embedding(user_query)
    #             # context = azure_search.query_index(query_embedding=query_embedding)
    #             # answer = generate_answer(user_query, context)
    #             try:
    #                 loop = asyncio.get_event_loop()
    #             except RuntimeError:
    #                 loop = asyncio.new_event_loop()
    #                 asyncio.set_event_loop(loop)
    #             response = loop.run_until_complete(
    #                 run_agent_team(task=user_query, team_type="selector")
    #             )
    #             answer = ""
    #             if isinstance(response, TaskResult):
    #                 total_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
    #                 for msg in response.messages:
    #                     if isinstance(msg, TextMessage):
    #                         # total_usage.prompt_tokens += msg.models_usage.prompt_tokens
    #                         # total_usage.completion_tokens += (
    #                         #     msg.models_usage.completion_tokens
    #                         # )
    #                         print(f"TextMessage: {msg.content}")
    #                         answer += f"\n\n### {msg.source}\n\n"
    #                         answer += msg.content + "\n"
    #                     elif isinstance(msg, MultiModalMessage):
    #                         # total_usage.prompt_tokens += msg.models_usage.prompt_tokens
    #                         # total_usage.completion_tokens += (
    #                         #     msg.models_usage.completion_tokens
    #                         # )
    #                         img_list = []
    #                         answer += f"\n\n### {msg.source}\n\n"
    #                         for c in msg.content:
    #                             if isinstance(c, str):
    #                                 answer += c + "\n"
    #                             else:
    #                                 # Convert autogen_core.image to BytesIO for display in Streamlit
    #                                 if hasattr(c, "image") and isinstance(
    #                                     c.image, Image.Image
    #                                 ):
    #                                     img_bytes = io.BytesIO()
    #                                     c.image.save(img_bytes, format="PNG")
    #                                     img_bytes.seek(0)
    #                                     img_list.append(img_bytes)
    #                                     answer += (
    #                                         "[Image]"  # Add placeholder text in answer
    #                                     )
    #                                 img_list.append(c)
    #                         for i, img in enumerate(img_list):
    #                             st.image(img, caption=f"Retrieved Image {i}")

    #                 # output = (
    #                 #     f"{'-' * 10} Summary {'-' * 10}\n"
    #                 #     f"Number of messages: {len(answer.messages)}\n"
    #                 #     f"Finish reason: {answer.stop_reason}\n"
    #                 #     f"Total prompt tokens: {total_usage.prompt_tokens}\n"
    #                 #     f"Total completion tokens: {total_usage.completion_tokens}\n"
    #                 #     # f"Duration: {duration:.2f} seconds\n"
    #                 # )

    #             st.success("Query completed!")
    #             st.markdown("### Answer:")
    #             st.markdown(answer)
    #         except Exception as e:
    #             logger.error(f"Error during querying: {e}")
    #             st.error("Error during querying.")
