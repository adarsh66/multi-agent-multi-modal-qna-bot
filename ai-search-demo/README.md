# AI Search Demo

This project demonstrates how to use Cohere multimodal embeddings to ingest both text and images from PDF documents into an AI search index, and then query it at runtime from a Streamlit-based UI.

## Project Structure

```
ai-search-demo
├── app.py               # Main entry point for the Streamlit application
├── extractor.py         # Functions for extracting text, tables, and images from PDFs
├── embeddings.py        # Functions to generate multimodal embeddings using Cohere
├── search.py            # Manages interactions with the Azure AI
├── rag.py               # Uses traditional, non agentic RAG workflow
├── agentic_rag.py       # Deploys multi-agent/ single-agent workflows 
├── agents
│   └── planner_agent.py        # Central decision maker in the Selector group chat responsible for planning steps for task execution
│   └── retrieval_agent.py        # Nested AssistantAgent to retrieve text and multi-modal content from retrieval engine of choice
│   └── answer_agent.py        # Answer the original quesiton using context retrieved
│   └── validation_agent.py        # Validate if answer meets original 
questions intent
├── agents              # Prompt library for each agent
├── utils
│   └── logger.py        # Logging functionality for the application
│   └── image_handler.py        # Image encoding
│   └── load_prompt.py        # load prompt based on file path
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd ai-search-demo
   ```

2. **Install dependencies:**
   Ensure you have Python installed, then run:
   ```
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   You may need to set up environment variables for your Azure AI search and Cohere API keys. Refer to the respective documentation for details.

## Usage Guidelines

1. **Run the application:**
   Start the Streamlit application by executing:
   ```
   streamlit run app.py
   ```

2. **Upload a PDF file:**
   Use the file upload feature in the UI to upload your PDF document. The application will extract text, tables, and images.

3. **Generate embeddings:**
   After uploading, click the button to trigger document intelligence, which will extract the necessary data and generate embeddings using the Cohere multimodal model.

4. **Ingest into Azure AI Search:**
   The generated embeddings will be ingested into the Azure AI search index for querying.

5. **Query the AI search index:**
   Select the type of query you want to test - single agent query, multi-agent using selector team etc. 
   Use the chat window in the UI to ask questions. The application will retrieve relevant answers based on the indexed data using the RAG (Retrieval-Augmented Generation) pattern.

## Logging and Tracing

The application includes logging functionality to provide status messages and trace the application's behavior. Check the logs for any issues or status updates during the execution.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.