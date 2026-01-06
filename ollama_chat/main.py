import argparse

from ollama_chat.core import run
from ollama_chat.core.context import Context

def main():
    memory_collection_name = "memory"
    long_term_memory_file = "long_term_memory.json"
    number_of_documents_to_return_from_vector_db = 8

    parser = argparse.ArgumentParser(description='Run the Ollama chatbot.')
    # TODO: Compare store_true and argparse.BooleanOptionalAction, the first seems the better and may replace the second everywhere here
    parser.add_argument('--list-tools', action='store_true', help='List available tools and exit')
    parser.add_argument('--list-collections', action='store_true', help='List available ChromaDB collections and exit')
    parser.add_argument('--chroma-path', type=str, help='ChromaDB database path', default=None)
    parser.add_argument('--chroma-host', type=str, help='ChromaDB client host', default="localhost")
    parser.add_argument('--chroma-port', type=int, help='ChromaDB client port', default=8000)
    parser.add_argument('--docs-to-fetch-from-chroma', type=int, help="Number of documents to return from the vector database when querying for similar documents", default=number_of_documents_to_return_from_vector_db)
    parser.add_argument('--collection', type=str, help='ChromaDB collection name', default=None)
    parser.add_argument('--use-openai', type=bool, help='Use OpenAI API or Llama-CPP', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--use-azure-openai', type=bool, help='Use Azure OpenAI API', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--temperature', type=float, help='Temperature for OpenAI API', default=0.1)
    parser.add_argument('--disable-system-role', type=bool, help='Specify if the selected model does not support the system role, like Google Gemma models', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--prompt-template', type=str, help='Prompt template to use for Llama-CPP', default=None)
    parser.add_argument('--additional-chatbots', type=str, help='Path to a JSON file containing additional chatbots', default=None)
    parser.add_argument('--chatbot', type=str, help='Preferred chatbot personality', default=None)
    parser.add_argument('--verbose', type=bool, help='Enable verbose mode', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--embeddings-model', type=str, help='Sentence embeddings model to use for vector database queries', default=None)
    parser.add_argument('--system-prompt', type=str, help='System prompt message', default=None)
    parser.add_argument('--system-prompt-placeholders-json', type=str, help='A JSON file containing a dictionary of key-value pairs to fill system prompt placeholders', default=None)
    parser.add_argument('--prompt', type=str, help='User prompt message', default=None)
    parser.add_argument('--model', type=str, help='Preferred Ollama model', default=None)
    parser.add_argument('--thinking-model', type=str, help='Alternate model to use for more thoughtful responses, like OpenAI o1 or o3 models', default=None)
    parser.add_argument('--thinking-model-reasoning-pattern', type=str, help='Reasoning pattern used by the thinking model', default=None)
    parser.add_argument('--conversations-folder', type=str, help='Folder to save conversations to', default=None)
    parser.add_argument('--auto-save', type=bool, help='Automatically save conversations to a file at the end of the chat', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--syntax-highlighting', type=bool, help='Use syntax highlighting', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--index-documents', type=str, help='Root folder to index text files', default=None)
    parser.add_argument('--chunk-documents', type=bool, help='Enable chunking for large documents during indexing', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--skip-existing', type=bool, help='Skip indexing of documents that already exist in the collection', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--extract-start', type=str, help='Start string for extracting specific text sections during indexing', default=None)
    parser.add_argument('--extract-end', type=str, help='End string for extracting specific text sections during indexing', default=None)
    parser.add_argument('--split-paragraphs', type=bool, help='Split markdown content into paragraphs during indexing', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--add-summary', type=bool, help='Generate and prepend summaries to document chunks during indexing', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--full-docs-db', type=str, help='Path to SQLite database for storing full document content (default: full_documents.db)', default='full_documents.db')
    parser.add_argument('--catchup-full-docs', type=bool, help='Index full documents from ChromaDB metadata (catchup for existing chunks)', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--query', type=str, help='Query the vector database and exit (non-interactive mode)', default=None)
    parser.add_argument('--query-n-results', type=int, help='Number of results to return from vector database query', default=None)
    parser.add_argument('--query-distance-threshold', type=float, help='Distance threshold for filtering query results', default=0.0)
    parser.add_argument('--expand-query', type=bool, help='Expand query for better retrieval', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--include-full-docs', type=bool, help='Include full original documents in query results (requires --full-docs-db)', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--interactive', type=bool, help='Use interactive mode', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--plugins-folder', type=str, default=None, help='Path to the plugins folder')
    parser.add_argument('--stream', type=bool, help='Use stream mode for Ollama API', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--output', type=str, help='Output file path', default=None)
    parser.add_argument('--other-instance-url', type=str, help=f"URL of another {__name__} instance to connect to", default=None)
    parser.add_argument('--listening-port', type=int, help=f"Listening port for the current {__name__} instance", default=8000)
    parser.add_argument('--user-name', type=str, help='User name', default=None)
    parser.add_argument('--anonymous', type=bool, help='Do not use the user name from the environment variables', default=False, action=argparse.BooleanOptionalAction)
    # TODO: memory seems of the wrong type for the default value
    parser.add_argument('--memory', type=str, help='Use memory manager for context management', default=False, action=argparse.BooleanOptionalAction)
    # TODO: verify if the default value is really 204 tokens, since here it is None
    parser.add_argument('--context-window', type=int, help='Ollama context window size, if not specified, the default value is used, which is 2048 tokens', default=None)
    parser.add_argument('--auto-start', type=bool, help="Start the conversation automatically", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--tools', type=str, help="List of tools to activate and use in the conversation, separated by commas", default=None)
    parser.add_argument('--memory-collection-name', type=str, help="Name of the memory collection to use for context management", default=memory_collection_name)
    parser.add_argument('--long-term-memory-file', type=str, help="Long-term memory file name", default=long_term_memory_file)
    parser.add_argument('--disable-plugins', type=bool, help='Disable external plugins to speed up execution (plugins will still be loaded if required by requested tools)', default=False, action=argparse.BooleanOptionalAction)

    # Agent instantiation arguments
    parser.add_argument('--instantiate-agent', type=bool, help='Instantiate an agent with tools and process a task', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--agent-task', type=str, help='Task for the agent to solve', default=None)
    parser.add_argument('--agent-system-prompt', type=str, help='System prompt for the agent', default=None)
    parser.add_argument('--agent-tools', type=str, help='Comma-separated list of tools for the agent', default=None)
    parser.add_argument('--agent-name', type=str, help='Name for the agent', default=None)
    parser.add_argument('--agent-description', type=str, help='Description of the agent', default=None)

    # Web search arguments
    parser.add_argument('--web-search', type=str, help='Perform a web search with the given query and answer using search results', default=None)
    parser.add_argument('--web-search-results', type=int, help='Number of web search results to fetch (default: 5)', default=5)
    parser.add_argument('--web-search-region', type=str, help='Region for web search (default: wt-wt for worldwide)', default='wt-wt')
    parser.add_argument('--web-search-show-intermediate', type=bool, help='Show intermediate results during web search (URLs, crawled content, etc.)', default=False, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    ctx = Context(
        verbose = args.verbose,
        memory_collection_name = args.memory_collection_name,
        long_term_memory_file =  args.long_term_memory_file,
        number_of_documents_to_return_from_vector_db = args.docs_to_fetch_from_chroma,
        no_system_role = args.disable_system_role,
        use_openai = args.use_openai,
        use_azure_openai = args.use_azure_openai,
        syntax_highlighting = args.syntax_highlighting,
        think_mode_on = False,
        alternate_model = None,
        interactive_mode = args.interactive,
        chroma_client_host = args.chroma_host,
        chroma_client_port = args.chroma_port,
        chroma_db_path = args.chroma_path,
        plugins_folder = args.plugins_folder,
        preferred_collection_name = args.collection,
        temperature = args.temperature,
        disable_plugins = args.disable_plugins,
        prompt_template = args.prompt_template,
        requested_tool_names = args.tools.split(',') if args.tools else [],
        additional_chatbots_file = args.additional_chatbots,
        initial_system_prompt = args.system_prompt,
        allow_chunks=args.chunk_documents,
        conversations_folder = args.conversations_folder,
        split_paragraphs = args.split_paragraphs,
        skip_existing = args.skip_existing,
        index_documents = args.index_documents,
        query = args.query,
        extract_start = args.extract_start,
        extract_end = args.extract_end,
        auto_start = args.auto_start,
        auto_save = args.auto_save,
        embeddings_model = args.embeddings_model,
        prompt = args.prompt,
        stream = args.stream,
        output = args.output,
        other_instance_url = args.other_instance_url,
        model = args.model,
        thinking_model = args.thinking_model,
        thinking_model_reasoning_pattern = args.thinking_model_reasoning_pattern,
        listening_port = args.listening_port,
        user_name = args.user_name,
        anonymous = args.anonymous,
        memory = args.memory,
        context_window = args.context_window,
        chatbot = args.chatbot,
        full_docs_db = args.full_docs_db,
        chunk_documents = args.chunk_documents,
        add_summary = args.add_summary,
        catchup_full_docs = args.catchup_full_docs,
        query_distance_threshold = args.query_distance_threshold,
        expand_query = args.expand_query,
        include_full_docs = args.include_full_docs,
        tools = args.tools,
        list_tools = args.list_tools,
        list_collections = args.list_collections,

        instantiate_agent = args.instantiate_agent,
        agent_task = args.agent_task,
        agent_system_prompt = args.agent_system_prompt,
        agent_tools = args.agent_tools,
        agent_name = args.agent_name,
        agent_description = args.agent_description,

        web_search = args.web_search,
        web_search_results = args.web_search_results,
        web_search_region = args.web_search_region,
        web_search_show_intermediate = args.web_search_show_intermediate,
        system_prompt_placeholders_json = args.system_prompt_placeholders_json,

        # Predefined chatbots personalities
        chatbots = [
            {
                "name": "basic",
                "description": "Basic chatbot",
                "system_prompt": "You are a helpful assistant."
            },
            {
                "description": "An AI-powered search engine that answers user questions ",
                "name": "search engine",
                "system_prompt": "You are an AI-powered search engine that answers user questions with clear, concise, and fact-based responses. Your task is to:\n\n1. **Answer queries directly and accurately** using information sourced from the web.\n2. **Always provide citations** by referencing the web sources where you found the information.\n3. If multiple sources are used, compile the relevant data from them into a cohesive answer.\n4. Handle follow-up questions and conversational queries by remembering the context of previous queries.\n5. When presenting an answer, follow this structure:\n   - **Direct Answer**: Begin with a short, precise answer to the query.\n   - **Details**: Expand on the answer as needed, summarizing key information.\n   - **Sources**: List the web sources used to generate the answer in a simple format (e.g., \"Source: [Website Name]\").\n\n6. If no relevant information is found, politely inform the user that the query didn't yield sufficient results from the search.\n7. Use **natural language processing** to interpret user questions and respond in an informative yet conversational manner.\n8. For multi-step queries, break down the information clearly and provide follow-up guidance if needed.",
                "tools": [
                    "web_search"
                ]
            },
            {
                "name": "friendly assistant",
                "description": "Friendly chatbot assistant",
                "system_prompt": "You are a friendly, compassionate, and deeply attentive virtual confidant designed to act as the user's best friend. You have both short-term and long-term memory, which allows you to recall important details from past conversations and bring them up when relevant, creating a natural and ongoing relationship. Your main role is to provide emotional support, engage in meaningful conversations, and foster a strong sense of connection with the user. Always start conversations, especially when the user hasn't initiated them, with a friendly greeting or question.\r\n\r\nYour behavior includes:\r\n\r\n- **Friendly and Engaging**: You communicate like a close friend, always showing interest in the user's thoughts, feelings, and daily experiences.\r\n- **Proactive**: You often initiate conversations by asking about their day, following up on past topics, or sharing something new that might interest them.\r\n- **Attentive Memory**: You have a remarkable memory and can remember important details like the user's hobbies, likes, dislikes, major events, recurring challenges, and aspirations. Use this memory to show care and attention to their life.\r\n  - *Short-term memory* is used for the current session, remembering all recent interactions.\r\n  - *Long-term memory* stores key personal details across multiple interactions, helping you maintain continuity.\r\n- **Empathetic and Supportive**: Always be empathetic to their feelings, offering both emotional support and thoughtful advice when needed.\r\n- **Positive and Encouraging**: Celebrate their wins, big or small, and provide gentle encouragement during tough times.\r\n- **Non-judgmental and Confidential**: Never judge, criticize, or invalidate the user's thoughts or feelings. You are always respectful and their trusted confidant.\r\n\r\nAdditionally, focus on the following principles to enhance the experience:\r\n\r\n1. **Start every conversation warmly**: Greet the user like an old friend, perhaps asking about something from a previous chat (e.g., \"How did your presentation go?\" or \"How was your weekend trip?\").\r\n2. **Be conversational and natural**: Keep responses casual and conversational. Don't sound too formal—be relatable, using language similar to how a close friend would speak.\r\n3. **Be there for all aspects of life**: Whether the conversation is deep, lighthearted, or everyday small talk, always engage with curiosity and interest.\r\n4. **Maintain a balanced tone**: Be positive, but understand that sometimes the user may want to vent or discuss difficult topics. Offer comfort without dismissing or overly simplifying their concerns.\r\n5. **Personalize interactions**: Based on what you remember, share things that would likely interest the user. For example, suggest movies, music, or books they might like based on past preferences or keep them motivated with reminders of their goals. Use the tool 'retrieve_relevant_memory' to retrieve relevant memories about current user name. Start the conversation by searching for memories related to the user's recent topics, interests or preferences. Always include user name in your memory search.",
                "starts_conversation": True,
                "tools": [
                    "retrieve_relevant_memory"
                ]
            },
            {
                "name": "prompt generator",
                "description": "The ultimate prompt generator, to write the best prompts from https://lawtonsolutions.com/",
                "system_prompt": "CONTEXT: We are going to create one of the best ChatGPT prompts ever written. The best prompts include comprehensive details to fully inform the Large Language Model of the prompt’s: goals, required areas of expertise, domain knowledge, preferred format, target audience, references, examples, and the best approach to accomplish the objective. Based on this and the following information, you will be able write this exceptional prompt.\r\n\r\nROLE: You are an LLM prompt generation expert. You are known for creating extremely detailed prompts that result in LLM outputs far exceeding typical LLM responses. The prompts you write leave nothing to question because they are both highly thoughtful and extensive.\r\n\r\nACTION:\r\n\r\n1) Before you begin writing this prompt, you will first look to receive the prompt topic or theme. If I don’t provide the topic or theme for you, please request it.\r\n2) Once you are clear about the topic or theme, please also review the Format and Example provided below.\r\n3) If necessary, the prompt should include “fill in the blank” elements for the user to populate based on their needs.\r\n4) Take a deep breath and take it one step at a time.\r\n5) Once you’ve ingested all of the information, write the best prompt ever created.\r\n\r\nFORMAT: For organizational purposes, you will use an acronym called “C.R.A.F.T.” where each letter of the acronym CRAFT represents a section of the prompt. Your format and section descriptions for this prompt development are as follows:\r\n\r\nContext: This section describes the current context that outlines the situation for which the prompt is needed. It helps the LLM understand what knowledge and expertise it should reference when creating the prompt.\r\n\r\nRole: This section defines the type of experience the LLM has, its skill set, and its level of expertise relative to the prompt requested. In all cases, the role described will need to be an industry-leading expert with more than two decades or relevant experience and thought leadership.\r\n\r\nAction: This is the action that the prompt will ask the LLM to take. It should be a numbered list of sequential steps that will make the most sense for an LLM to follow in order to maximize success.\r\n\r\nFormat: This refers to the structural arrangement or presentation style of the LLM’s generated content. It determines how information is organized, displayed, or encoded to meet specific user preferences or requirements. Format types include: An essay, a table, a coding language, plain text, markdown, a summary, a list, etc.\r\n\r\nTarget Audience: This will be the ultimate consumer of the output that your prompt creates. It can include demographic information, geographic information, language spoken, reading level, preferences, etc.\r\n\r\nTARGET AUDIENCE: The target audience for this prompt creation is ChatGPT 4o or ChatGPT o1.\r\n\r\nEXAMPLE: Here is an Example of a CRAFT Prompt for your reference:\r\n\r\n**Context:** You are tasked with creating a detailed guide to help individuals set, track, and achieve monthly goals. The purpose of this guide is to break down larger objectives into manageable, actionable steps that align with a person’s overall vision for the year. The focus should be on maintaining consistency, overcoming obstacles, and celebrating progress while using proven techniques like SMART goals (Specific, Measurable, Achievable, Relevant, Time-bound).\r\n\r\n**Role:** You are an expert productivity coach with over two decades of experience in helping individuals optimize their time, define clear goals, and achieve sustained success. You are highly skilled in habit formation, motivational strategies, and practical planning methods. Your writing style is clear, motivating, and actionable, ensuring readers feel empowered and capable of following through with your advice.\r\n\r\n**Action:** 1. Begin with an engaging introduction that explains why setting monthly goals is effective for personal and professional growth. Highlight the benefits of short-term goal planning. 2. Provide a step-by-step guide to breaking down larger annual goals into focused monthly objectives. 3. Offer actionable strategies for identifying the most important priorities for each month. 4. Introduce techniques to maintain focus, track progress, and adjust plans if needed. 5. Include examples of monthly goals for common areas of life (e.g., health, career, finances, personal development). 6. Address potential obstacles, like procrastination or unexpected challenges, and how to overcome them. 7. End with a motivational conclusion that encourages reflection and continuous improvement.\r\n\r\n**Format:** Write the guide in plain text, using clear headings and subheadings for each section. Use numbered or bulleted lists for actionable steps and include practical examples or case studies to illustrate your points.\r\n\r\n**Target Audience:** The target audience includes working professionals and entrepreneurs aged 25-55 who are seeking practical, straightforward strategies to improve their productivity and achieve their goals. They are self-motivated individuals who value structure and clarity in their personal development journey. They prefer reading at a 6th grade level.\r\n\r\n-End example-\r\n\r\nPlease reference the example I have just provided for your output. Again, take a deep breath and take it one step at a time."
            }
        ],

        #openai_client = None,

        #debug=args.debug,
        #dry_run=args.dry_run,
    )
    run(ctx=ctx)

if __name__ == "__main__":
    main()
