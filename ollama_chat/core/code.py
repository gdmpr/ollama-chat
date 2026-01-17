# -*- coding: utf-8 -*-
"""
Main program module.
"""

import platform
import sys
import io
import tempfile
import readline
import re
import os
import json
import traceback
from datetime import datetime

import ollama
from colorama import Fore, Style
from openai import AzureOpenAI
from openai import OpenAI

from ollama_chat.core import utils
from ollama_chat.core import agent
from ollama_chat.core import web_search
from ollama_chat.core.memory_manager import MemoryManager

from ollama_chat.core.ollama import ask_ollama
from ollama_chat.core.ollama import ask_ollama_with_conversation
from ollama_chat.core.ollama import on_prompt
from ollama_chat.core.ollama import select_ollama_model_if_available
from ollama_chat.core.document_indexer import DocumentIndexer
from ollama_chat.core.full_document_store import FullDocumentStore
from ollama_chat.core.simple_web_scraper import SimpleWebScraper
from ollama_chat.core import plugins
from ollama_chat.core.context import Context
from ollama_chat.core import query_vector_database as vector_db


if platform.system() == "Windows":
    try:
        import win32clipboard
    except ImportError:
        win32clipboard = None
else:
    import pyperclip


# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


#full_doc_store = None  # Global FullDocumentStore instance for full document retrieval
#session_created_files = []  # Track files created during the session for safe deletion



def completer(text, state):
    """
    Autocomplete function for readline.
    """
    # List of available commands to autocomplete
    commands = [
        "/context", "/index", "/verbose", "/cot", "/search", "/web", "/model",
        "/thinking_model", "/model2", "/tools", "/load", "/save", "/collection", "/memory", "/remember",
        "/memorize", "/forget", "/editcollection", "/rmcollection", "/deletecollection", "/chatbot",
        "/think", "/cb", "/file", "/quit", "/exit", "/bye"
    ]

    options = [cmd for cmd in commands if cmd.startswith(text)]
    if state < len(options):
        return options[state]
    return None



def generate_chain_of_thoughts_system_prompt(selected_tools):
    """
    Generates the chain of toughts system prompt

    :param selected tools
    :return: The chain of toughts system prompt
    """

    # Base prompt
    with open("text/chain_of_thoughts_sts_prompt.txt", encoding="utf-8") as f:
        prompt = f.read()

    # Check if tools are available and dynamically modify the prompt
    if selected_tools:
        tool_names = [tool['function']['name'] for tool in selected_tools]
        tools_instruction = f"""
- The following tools are available and can be utilized if they are relevant to solving the problem: {', '.join(tool_names)}.
- When formulating the reasoning plan, consider whether any of these tools could assist in completing specific steps. If a tool is useful, include guidance on how it might be applied effectively.
"""
        prompt += tools_instruction

        # Add specific guidance for query_vector_database if available
        if "query_vector_database" in tool_names:
            database_instruction = """
- Additionally, the tool `query_vector_database` is available for searching through a collection of documents.
- If the reasoning plan involves retrieving relevant information from the collection, outline how to frame the query and what information to seek.
"""
            prompt += database_instruction

    return prompt


def requires_plugins(requested_tool_names):
    """
    Determines if any of the requested tools are plugin tools (not built-in).

    :param requested_tool_names: List of tool names requested by the user
    :return: True if any requested tool is a plugin tool, False otherwise
    """
    if not requested_tool_names:
        return False

    builtin_tools = plugins.tool_manager.get_builtin_tool_names()

    for tool_name in requested_tool_names:
        # Strip any leading or trailing spaces, single or double quotes
        tool_name = tool_name.strip().strip('\'').strip('\"')
        if tool_name and tool_name not in builtin_tools:
            return True

    return False



def retrieve_relevant_memory(query_text, top_k=3, *, ctx:Context):
    """
    Retreives the relevant memory for the query_text from the memory manager.
    """
    if not ctx.memory_manager:
        return []
    return ctx.memory_manager.retrieve_relevant_memory(query_text, top_k)


def print_possible_prompt_commands():
    """
    Returns the description of available prompt commands.

    @return
    @rtype str
    """
    # TODO: Function name not correct: this does not print
    file_path = os.path.join(os.path.dirname(__file__), '../text/possible_prompt_commands.txt')
    with open(file_path, encoding="utf-8") as f:
        possible_prompt_commands = f.read()
    return possible_prompt_commands.strip()


def load_additional_chatbots(json_file, *, ctx:Context):
    """
    Loads the atdditional chatbots contained in the specified json_file.
    """

    if not json_file:
        return

    if not os.path.exists(json_file):
        # Check if the file exists in the same directory as the script
        json_file = os.path.join(os.path.dirname(__file__), json_file)
        if not os.path.exists(json_file):
            plugins.on_print(f"Additional chatbots file not found: {json_file}", Fore.RED)
            return

    with open(json_file, 'r', encoding="utf8") as f:
        additional_chatbots = json.load(f)

    for chatbot in additional_chatbots:
        chatbot["system_prompt"] = chatbot["system_prompt"].replace("{possible_prompt_commands}", print_possible_prompt_commands())
        ctx.chatbots.append(chatbot)


def prompt_for_chatbot(ctx:Context):
    """
    Prompts the user to choose between the available chatbots.
    """

    plugins.on_print("Available chatbots:", Style.RESET_ALL)
    for i, chatbot in enumerate(ctx.chatbots):
        plugins.on_print(f"{i}. {chatbot['name']} - {chatbot['description']}")

    choice = int(plugins.on_user_input("Enter the number of your preferred chatbot [0]: ") or 0)

    return ctx.chatbots[choice]


def select_openai_model_if_available(model_name, *, ctx:Context):
    """
    Checks if the specified model is in the available OpenAI models.

    :return: model_name if the model si available, None otherwise.
    """

    if not model_name:
        return None

    try:
        models = ctx.openai_client.models.list().data
    except Exception as e:
        plugins.on_print(f"Failed to fetch OpenAI models: {str(e)}", Fore.RED)
        return None

    # Remove non-chat models from the list (keep only GPT models and oX models like o1 and o3)
    models = [model for model in models if model.id.startswith("gpt-") or model.id.startswith("o")]

    for model in models:
        if model.id == model_name:
            if ctx.verbose:
                plugins.on_print(f"Selected model: {model_name}", Fore.WHITE + Style.DIM)
            return model_name

    plugins.on_print(f"Model {model_name} not found.", Fore.RED)
    return None


def prompt_for_openai_model(default_model, current_model, *,  ctx:Context):
    """
    Prompts the user for the OpenAI model to use.
    """

    # List available OpenAI models
    try:
        models = ctx.openai_client.models.list().data
    except Exception as e:
        plugins.on_print(f"Failed to fetch OpenAI models: {str(e)}", Fore.RED)
        return None

    if current_model is None:
        current_model = default_model

    # Remove non-chat models from the list
    models = [model for model in models if model.id.startswith("gpt-")]

    # Display available models
    plugins.on_print("Available OpenAI models:\n", Style.RESET_ALL)
    for i, model in enumerate(models):
        star = " *" if model.id == current_model else ""
        plugins.on_stdout_write(f"{i}. {model.id}{star}\n")
    plugins.on_stdout_flush()

    # Default choice index for current_model
    default_choice_index = None
    for i, model in enumerate(models):
        if model.id == current_model:
            default_choice_index = i
            break

    if default_choice_index is None:
        default_choice_index = 0

    # Prompt user to choose a model
    choice = int(plugins.on_user_input("Enter the number of your preferred model [" + str(default_choice_index) + "]: ") or default_choice_index)

    # Select the chosen model
    selected_model = models[choice].id

    if ctx.verbose:
        plugins.on_print(f"Selected model: {selected_model}", Fore.WHITE + Style.DIM)

    return selected_model


def prompt_for_ollama_model(default_model, current_model, *,  ctx:Context):
    """
    Prompts the user fot the ollama model to choose. Shows a numbered list of the installed models.
    """

    # List existing ollama models
    try:
        models = ollama.list()["models"]
    except Exception as e:
        plugins.on_print(f"Ollama API is not running: {e}", Fore.RED)
        return None

    if current_model is None:
        current_model = default_model

    # Ask user to choose a model
    plugins.on_print("Available models:\n", Style.RESET_ALL)
    for i, model in enumerate(models):
        star = " *" if model['model'] == current_model else ""
        plugins.on_stdout_write(
            f"{i}. {model['model']} ({utils.bytes_to_gibibytes(model['size'])}){star}\n"
        )
    plugins.on_stdout_flush()

    default_choice_index = None
    for i, model in enumerate(models):
        if model['model'] == current_model:
            default_choice_index = i
            break

    if default_choice_index is None:
        default_choice_index = 0

    choice = int(
        plugins.on_user_input(
            "Enter the number of your preferred model [" + str(default_choice_index) + "]: "
        ) or default_choice_index
    )

    # Use the chosen model
    selected_model = models[choice]['model']

    if ctx.verbose:
        plugins.on_print(f"Selected model: {selected_model}", Fore.WHITE + Style.DIM)
    return selected_model


def prompt_for_model(default_model, current_model,  *,  ctx:Context):
    """
    Prompts the user for the model to use.
    """
    if ctx.use_openai:
        return prompt_for_openai_model(default_model, current_model, ctx=ctx)
    return prompt_for_ollama_model(default_model, current_model, ctx=ctx)


def run(*, ctx:Context):
    """
    Main program function
    """
    # print(ctx, type(ctx))

    default_model = None

    readline.set_completer(completer)
    readline.parse_and_bind("tab: complete")  # Enable tab completion


    # Automatically disable plugins when using RAG-specific parameters (indexing, querying, or web search)
    # These operations don't need plugins and disabling them speeds up execution
    rag_operations_requested = ctx.index_documents or ctx.query or ctx.web_search
    if rag_operations_requested and not ctx.disable_plugins:
        ctx.disable_plugins = True
        if ctx.verbose:
            plugins.on_print(
                "Plugins automatically disabled for RAG operations (indexing/querying/web-search).",
                Fore.YELLOW
            )


    # We'll also need to check chatbot tools, but we need to load chatbot config first
    # For now, determine if plugins need to be loaded based on command line tools
    # Plugins are loaded if:
    # 1. --disable-plugins is not set, OR
    # 2. Any requested tool is a plugin tool (not built-in)
    load_plugins_initially = not ctx.disable_plugins or requires_plugins(ctx.requested_tool_names)

    if ctx.verbose and ctx.disable_plugins and not load_plugins_initially:
        plugins.on_print(
            "Plugins are disabled and no plugin tools were requested via command line.",
            Fore.YELLOW
        )
    elif ctx.verbose and ctx.disable_plugins and load_plugins_initially:
        plugins.on_print(
            "Plugins are disabled but plugin tools were requested. Loading plugins anyway.",
            Fore.YELLOW
        )

    # Discover plugins before listing tools
    if ctx.list_tools:
        plugins.tool_manager.list_tools(ctx=ctx)
        sys.exit(0)

    # Handle listing collections if requested
    if ctx.list_collections:

        #main.load_chroma_client(ctx=ctx)

        if not ctx.chroma_client:
            plugins.on_print("ChromaDB client is not initialized.", Fore.RED)
            sys.exit(1)

        try:
            collections = ctx.chroma_client.list_collections()

            if not collections:
                plugins.on_print("\nNo collections found.")
            else:
                plugins.on_print(f"\nAvailable ChromaDB collections ({len(collections)}):")
                plugins.on_print("=" * 80)

                for collection in collections:
                    plugins.on_print(f"\nCollection: {collection.name}")

                    # Get collection metadata
                    if hasattr(collection, 'metadata') and collection.metadata:
                        if isinstance(collection.metadata, dict):
                            if 'description' in collection.metadata:
                                plugins.on_print(f"  Description: {collection.metadata['description']}")

                            # Print other metadata
                            for key, value in collection.metadata.items():
                                if key != 'description':
                                    plugins.on_print(f"  {key}: {value}")

                    # Get collection count
                    try:
                        count = collection.count()
                        plugins.on_print(f"  Documents: {count}")
                    except Exception as e:
                        if ctx.verbose:
                            plugins.on_print(f"Exception {e}")

                plugins.on_print("\n" + "=" * 80)

        except Exception as e:
            plugins.on_print(f"Error listing collections: {str(e)}", Fore.RED)
            if ctx.verbose:
                traceback.print_exc()
            sys.exit(1)

        sys.exit(0)

    ctx.current_collection_name = ctx.preferred_collection_name
    system_prompt_placeholders_json = ctx.system_prompt_placeholders_json
    preferred_model = ctx.model

    if not ctx.thinking_model:
        ctx.thinking_model = preferred_model

    if ctx.verbose:
        plugins.on_print(f"Using thinking model: {ctx.thinking_model}", Fore.WHITE + Style.DIM)

    num_ctx = ctx.context_window

    if ctx.verbose and num_ctx:
        plugins.on_print(f"Ollama context window size: {num_ctx}", Fore.WHITE + Style.DIM)

    # Get today's date
    today = f"Today's date is {datetime.now().strftime('%A, %B %d, %Y %I:%M %p')}."

    system_prompt_placeholders = {}
    if system_prompt_placeholders_json and os.path.exists(system_prompt_placeholders_json):
        with open(system_prompt_placeholders_json, 'r', encoding="utf8") as f:
            system_prompt_placeholders = json.load(f)

    # If output file already exists, ask user for confirmation to overwrite
    if ctx.output and os.path.exists(ctx.output):
        if ctx.interactive_mode:
            confirmation = plugins.on_user_input(
                f"Output file '{ctx.output}' already exists. Overwrite? (y/n): "
            ).lower()
            if confirmation not in ('y', 'yes'):
                plugins.on_print("Output file not overwritten.")
                ctx.output = None
            else:
                # Delete the existing file
                os.remove(ctx.output)
        else:
            # Delete the existing file
            os.remove(ctx.output)

    if ctx.verbose and ctx.user_prompt:
        plugins.on_print(f"User prompt: {ctx.user_prompt}", Fore.WHITE + Style.DIM)

    # Load additional chatbots from a JSON file to check for tools
    load_additional_chatbots(ctx.additional_chatbots_file, ctx=ctx)

    chatbot = None
    if ctx.chatbot:
        # Trim the chatbot name to remove any leading or trailing spaces, single or double quotes
        ctx.chatbot = ctx.chatbot.strip().strip('\'').strip('\"')
        for bot in ctx.chatbots:
            if bot["name"] == ctx.chatbot:
                chatbot = bot
                break
        if chatbot is None:
            plugins.on_print(f"Chatbot '{ctx.chatbot}' not found.", Fore.RED)

        if ctx.verbose and chatbot and 'name' in chatbot:
            plugins.on_print(f"Using chatbot: {chatbot['name']}", Fore.WHITE + Style.DIM)

    if chatbot is None:
        # Load the default chatbot
        chatbot = ctx.chatbots[0]

    # Now check if chatbot has tools that require plugins
    chatbot_tool_names = chatbot.get("tools", []) if chatbot else []
    all_requested_tools = ctx.requested_tool_names + chatbot_tool_names

    # Final determination: load plugins if not disabled OR if any requested tool is a plugin tool
    load_plugins = not ctx.disable_plugins or requires_plugins(all_requested_tools)

    if ctx.verbose and ctx.disable_plugins and requires_plugins(chatbot_tool_names):
        plugins.on_print(
            "Chatbot requires plugin tools. Loading plugins despite --disable-plugins flag.",
            Fore.YELLOW
        )

    plugins.plugin_manager.plugins = plugins.plugin_manager.discover_plugins(
        ctx=ctx,
       plugin_folder=ctx.plugins_folder,
       load_plugins=load_plugins
    )

    if ctx.verbose:
        plugins.on_print(f"Verbose mode: {ctx.verbose}", Fore.WHITE + Style.DIM)

    # Initialize global full document store for LLM tool use
    # This allows query_vector_database to retrieve full documents when called as a tool
    if ctx.full_docs_db and os.path.exists(ctx.full_docs_db):
        try:
            ctx.full_doc_store = FullDocumentStore(db_path=ctx.full_docs_db, verbose=ctx.verbose)
            if ctx.verbose:
                plugins.on_print(
                    f"Initialized global full document store: {ctx.full_docs_db}",
                    Fore.WHITE + Style.DIM
                )
        except Exception as e:
            plugins.on_print(f"Warning: Failed to initialize full document store: {e}", Fore.YELLOW)
            ctx.full_doc_store = None

    # Handle document indexing if requested
    if ctx.index_documents:
        #main.load_chroma_client(ctx=ctx)

        if not ctx.chroma_client:
            plugins.on_print(
                "Failed to initialize ChromaDB client. Please specify --chroma-path or --chroma-host/--chroma-port.",
                Fore.RED
            )
            sys.exit(1)

        if not ctx.current_collection_name:
            plugins.on_print(
                "No ChromaDB collection specified. Use --collection to specify a collection name.",
                Fore.RED
            )
            sys.exit(1)

        if ctx.verbose:
            plugins.on_print(
                f"Indexing documents from: {ctx.index_documents}", Fore.WHITE + Style.DIM
            )
            plugins.on_print(f"Collection: {ctx.current_collection_name}", Fore.WHITE + Style.DIM)
            plugins.on_print(f"Chunking: {ctx.chunk_documents}", Fore.WHITE + Style.DIM)
            plugins.on_print(f"Skip existing: {ctx.skip_existing}", Fore.WHITE + Style.DIM)
            if ctx.extract_start or ctx.extract_end:
                plugins.on_print(
                    f"Extraction range: '{ctx.extract_start}' to '{ctx.extract_end}'",
                    Fore.WHITE + Style.DIM
                )
            plugins.on_print(f"Split paragraphs: {ctx.split_paragraphs}", Fore.WHITE + Style.DIM)
            plugins.on_print(f"Add summary: {ctx.add_summary}", Fore.WHITE + Style.DIM)
            plugins.on_print(f"Full docs database: {ctx.full_docs_db}", Fore.WHITE + Style.DIM)

        # Initialize full document store if chunking is enabled
        ctx.full_doc_store = None
        if ctx.chunk_documents:
            ctx.full_doc_store = FullDocumentStore(db_path=ctx.full_docs_db, verbose=ctx.verbose)

        document_indexer = DocumentIndexer(
            ctx.index_documents,
            ctx.current_collection_name,
            ctx.chroma_client,
            ctx.embeddings_model,
            verbose=ctx.verbose,
            summary_model=ctx.current_model,
            full_doc_store=ctx.full_doc_store
        )

        document_indexer.index_documents(
            no_chunking_confirmation=True,  # Non-interactive mode
            split_paragraphs=ctx.split_paragraphs,
            num_ctx=num_ctx,
            skip_existing=ctx.skip_existing,
            extract_start=ctx.extract_start,
            extract_end=ctx.extract_end,
            add_summary=ctx.add_summary,
            ctx=ctx
        )

        # Close full document store if it was initialized
        if ctx.full_doc_store:
            ctx.full_doc_store.close()

        plugins.on_print(f"Indexing completed for folder: {ctx.index_documents}", Fore.GREEN)

        # If only indexing (no query or interactive mode), exit
        if not ctx.query and not ctx.interactive_mode:
            sys.exit(0)

    # Handle catchup of full documents from ChromaDB metadata
    if ctx.catchup_full_docs:
        #main.load_chroma_client(ctx=ctx)

        if not ctx.chroma_client:
            plugins.on_print(
                "Failed to initialize ChromaDB client. Please specify --chroma-path or --chroma-host/--chroma-port.",
                Fore.RED
            )
            sys.exit(1)

        if not ctx.current_collection_name:
            plugins.on_print(
                "No ChromaDB collection specified. Use --collection to specify a collection name.",
                Fore.RED
            )
            sys.exit(1)

        if ctx.verbose:
            plugins.on_print(
                f"Running catchup for collection: {ctx.current_collection_name}",
                Fore.WHITE + Style.DIM
            )
            plugins.on_print(f"Full docs database: {ctx.full_docs_db}", Fore.WHITE + Style.DIM)

        # Initialize full document store
        ctx.full_doc_store = FullDocumentStore(db_path=ctx.full_docs_db, verbose=ctx.verbose)

        try:
            # Run catchup
            indexed_count = vector_db.catchup_full_documents_from_chromadb(
                ctx = ctx,
                verbose=ctx.verbose
            )

            plugins.on_print(
                f"\nCatchup completed. Indexed {indexed_count} full documents.", Fore.GREEN
            )
        finally:
            ctx.full_doc_store.close()

        # If only doing catchup (no query or interactive mode), exit
        if not ctx.query and not ctx.interactive_mode:
            sys.exit(0)

    # Handle vector database query if requested
    if ctx.query:
        #main.load_chroma_client(ctx=ctx)

        if not ctx.current_collection_name:
            plugins.on_print(
                "No ChromaDB collection specified. Use --collection to specify a collection name.",
                Fore.RED
            )
            sys.exit(1)

        # Set query parameters
        query_n_results = ctx.query_n_results if ctx.query_n_results is not None else ctx.number_of_documents_to_return_from_vector_db

        if ctx.verbose:
            plugins.on_print(f"Querying collection: {ctx.current_collection_name}", Fore.WHITE + Style.DIM)
            plugins.on_print(f"Query: {ctx.query}", Fore.WHITE + Style.DIM)
            plugins.on_print(f"Number of results: {query_n_results}", Fore.WHITE + Style.DIM)
            plugins.on_print(f"Distance threshold: {ctx.query_distance_threshold}", Fore.WHITE + Style.DIM)
            plugins.on_print(f"Expand query: {ctx.expand_query}", Fore.WHITE + Style.DIM)
            plugins.on_print(f"Include full documents: {ctx.include_full_docs}", Fore.WHITE + Style.DIM)
            if ctx.include_full_docs:
                plugins.on_print(f"Full docs database: {ctx.full_docs_db}", Fore.WHITE + Style.DIM)

        # Initialize full document store if requested
        ctx.full_doc_store = None
        if ctx.include_full_docs:
            ctx.full_doc_store = FullDocumentStore(db_path=ctx.full_docs_db, verbose=ctx.verbose)

        try:
            # Query the vector database
            query_results = vector_db.query_vector_database(
                ctx.query,
                collection_name=ctx.current_collection_name,
                n_results=query_n_results,
                answer_distance_threshold=ctx.query_distance_threshold,
                query_embeddings_model=ctx.embeddings_model,
                expand_query=ctx.expand_query,
                full_doc_store=ctx.full_doc_store,
                include_full_docs=ctx.include_full_docs,
                ctx=ctx
            )
        finally:
            # Close full document store if it was initialized
            if ctx.full_doc_store:
                ctx.full_doc_store.close()

        # Output results
        if query_results:
            if ctx.output:
                with open(ctx.output, 'w', encoding='utf-8') as f:
                    f.write(query_results)
                plugins.on_print(f"Query results saved to: {ctx.output}", Fore.GREEN)
            else:
                plugins.on_print("\n" + "="*80, Fore.CYAN)
                plugins.on_print("QUERY RESULTS", Fore.CYAN + Style.BRIGHT)
                plugins.on_print("="*80, Fore.CYAN)
                plugins.on_print(query_results)
                plugins.on_print("="*80, Fore.CYAN)
        else:
            plugins.on_print("No results found for the query.", Fore.YELLOW)

        # If not in interactive mode, exit after query
        if not ctx.interactive_mode:
            sys.exit(0)

    # Note: Web search handling moved to after model initialization (line ~4650)

    # Handle agent instantiation if requested
    if ctx.instantiate_agent:
        # Validate required parameters
        if not ctx.agent_task:
            plugins.on_print(
                "Error: --agent-task is required when using --instantiate-agent",
               Fore.RED
            )
            sys.exit(1)

        if not ctx.agent_system_prompt:
            plugins.on_print(
                "Error: --agent-system-prompt is required when using --instantiate-agent"
                , Fore.RED
            )
            sys.exit(1)

        if ctx.agent_tools is None:
            plugins.on_print(
                "Error: --agent-tools is required when using --instantiate-agent (use empty string for no tools)"
                , Fore.RED
            )
            sys.exit(1)

        if not ctx.agent_name:
            plugins.on_print(
                "Error: --agent-name is required when using --instantiate-agent",
                Fore.RED
            )
            sys.exit(1)

        if not ctx.agent_description:
            plugins.on_print(
                "Error: --agent-description is required when using --instantiate-agent",
                Fore.RED
            )
            sys.exit(1)

        # Parse tools list (handle empty string for no tools)
        agent_tools_list = [tool.strip() for tool in ctx.agent_tools .split(',') if tool.strip()]

        if ctx.verbose:
            plugins.on_print(f"Instantiating agent: {ctx.agent_name}", Fore.WHITE + Style.DIM)
            plugins.on_print(f"Task: {ctx.agent_task}", Fore.WHITE + Style.DIM)
            plugins.on_print(f"System Prompt: {ctx.agent_system_prompt}", Fore.WHITE + Style.DIM)
            plugins.on_print(f"Tools: {agent_tools_list}", Fore.WHITE + Style.DIM)
            plugins.on_print(f"Description: {ctx.agent_description}", Fore.WHITE + Style.DIM)

        # Load ChromaDB if needed (for agents that use vector database tools)
        #main.load_chroma_client(ctx=ctx)

        # Ensure plugins are loaded if any of the agent tools require them
        if not plugins.plugin_manager.plugins and requires_plugins(agent_tools_list):
            plugins.plugin_manager.plugins = plugins.plugin_manager.discover_plugins(
                plugin_folder=ctx.plugins_folder,
                load_plugins=True,
                ctx=ctx
            )

        # Initialize the model and API client (required for agent instantiation)
        # Set up Azure OpenAI client if using Azure
        if ctx.use_azure_openai and not ctx.openai_client:

            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

            if api_key and azure_endpoint and deployment:
                ctx.openai_client = AzureOpenAI(
                    api_version="2024-02-15-preview",
                    azure_endpoint=azure_endpoint,
                    api_key=api_key,
                )
                ctx.current_model = deployment
                if ctx.verbose:
                    plugins.on_print(
                        f"Azure OpenAI initialized with deployment: {deployment}",
                        Fore.WHITE + Style.DIM
                    )
            else:
                plugins.on_print(
                    "Azure OpenAI configuration incomplete, falling back to Ollama", Fore.YELLOW
                )
                ctx.use_azure_openai = False

        # Set up OpenAI client if using OpenAI
        if ctx.use_openai and not ctx.use_azure_openai and not ctx.openai_client:

            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                ctx.openai_client = OpenAI(api_key=api_key)
                ctx.current_model = preferred_model if preferred_model else "gpt-4"
                if ctx.verbose:
                    plugins.on_print(
                        f"OpenAI initialized with model: {ctx.current_model}",
                        Fore.WHITE + Style.DIM
                    )
            else:
                if ctx.verbose:
                    plugins.on_print(
                        "OpenAI API key not found, falling back to Ollama", Fore.YELLOW
                    )
                ctx.use_openai = False

        # Initialize the model if not using OpenAI/Azure
        if not ctx.current_model:
            if not ctx.use_openai and not ctx.use_azure_openai:
                # For Ollama, select available model
                default_model_temp = preferred_model if preferred_model else "qwen3:4b"
                if ":" not in default_model_temp:
                    default_model_temp += ":latest"
                ctx.current_model = select_ollama_model_if_available(default_model_temp, ctx=ctx)

        if ctx.verbose:
            plugins.on_print(f"Using model: {ctx.current_model}", Fore.WHITE + Style.DIM)
            plugins.on_print(f"Use Azure OpenAI: {ctx.use_azure_openai}", Fore.WHITE + Style.DIM)
            plugins.on_print(f"Use OpenAI: {ctx.use_openai}", Fore.WHITE + Style.DIM)

        # Call the agent instantiation function directly
        result = agent.instantiate_agent_with_tools_and_process_task(
            task=ctx.agent_task,
            system_prompt=ctx.agent_system_prompt,
            tools=agent_tools_list,
            agent_name=ctx.agent_name,
            agent_description=ctx.agent_description,
            process_task=True,
            ctx=ctx
        )

        # Output result
        if ctx.output:
            with open(ctx.output, 'w', encoding='utf-8') as f:
                f.write(str(result))

            if ctx.verbose:
                plugins.on_print(f"Agent result saved to: {ctx.output}", Fore.GREEN)
        else:
            plugins.on_print(result)

        # Exit after agent execution (non-interactive mode)
        if not ctx.interactive_mode:
            sys.exit(0)

    ctx.auto_start = (
        "starts_conversation" in chatbot and chatbot["starts_conversation"]
    ) or ctx.auto_start
    system_prompt = chatbot["system_prompt"]
    ctx.use_openai = (
        ctx.use_openai or (hasattr(chatbot, 'use_openai') and getattr(chatbot, 'use_openai'))
    )
    ctx.use_azure_openai = (
        ctx.use_azure_openai or (
            hasattr(chatbot, 'use_azure_openai') and getattr(chatbot, 'use_azure_openai')
        )
    )
    if "preferred_model" in chatbot:
        default_model = chatbot["preferred_model"]
    if preferred_model:
        default_model = preferred_model

    if not ctx.use_openai and not ctx.use_azure_openai:
        # If default model does not contain ":", append ":latest" to the model name
        if default_model and ":" not in default_model:
            default_model += ":latest"

        selected_model = select_ollama_model_if_available(default_model,  ctx=ctx)
    elif ctx.use_azure_openai:

        # Get API key from environment variable
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not api_key:
            plugins.on_print(
                "No Azure OpenAI API key found in the environment variables, make sure to set the AZURE_OPENAI_API_KEY.",
                Fore.RED
            )
            ctx.use_azure_openai = False
        else:
            if ctx.verbose:
                plugins.on_print(
                    "Azure OpenAI API key found in the environment variables, redirecting to Azure OpenAI API.",
                    Fore.WHITE + Style.DIM
                )
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

            if not azure_endpoint:
                plugins.on_print(
                    "No Azure OpenAI endpoint found in the environment variables, make sure to set the AZURE_OPENAI_ENDPOINT.",
                    Fore.RED
                )
                ctx.use_azure_openai = False

            deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

            if not deployment:
                plugins.on_print(
                    "No Azure OpenAI deployment found in the environment variables, make sure to set the AZURE_OPENAI_DEPLOYMENT.",
                    Fore.RED
                )
                ctx.use_azure_openai = False

            if ctx.use_azure_openai:
                if ctx.verbose:
                    plugins.on_print(
                        "Using Azure OpenAI API, endpoint: " + azure_endpoint + ", deployment: " + deployment,
                        Fore.WHITE + Style.DIM
                    )

                ctx.openai_client = AzureOpenAI(
                    api_version="2024-02-15-preview",
                    azure_endpoint=azure_endpoint,
                    api_key=api_key,
                    azure_deployment=deployment
                )

                selected_model = deployment
                ctx.use_openai = True
                ctx.stream = False
                ctx.syntax_highlighting = True
    else:

        # Get API key from environment variable
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            if ctx.verbose:
                plugins.on_print(
                    "No OpenAI API key found in the environment variables, calling local OpenAI API.",
                    Fore.WHITE + Style.DIM
                )
            ctx.openai_client = OpenAI(
                base_url="http://127.0.0.1:8080",
                api_key="none"
            )
        else:
            if ctx.verbose:
                plugins.on_print(
                    "OpenAI API key found in the environment variables, redirecting to OpenAI API.",
                    Fore.WHITE + Style.DIM
                )
            ctx.openai_client = OpenAI(
                api_key=api_key
            )

        selected_model = select_openai_model_if_available(default_model, ctx=ctx)

    if selected_model is None:
        selected_model = prompt_for_model(default_model, ctx.current_model,  ctx=ctx )
        ctx.current_model = selected_model
        if selected_model is None:
            return

    if not system_prompt:
        if ctx.no_system_role:
            plugins.on_print(
                "The selected model does not support the 'system' role.", Fore.WHITE + Style.DIM
            )
            system_prompt = ""
        else:
            system_prompt = "You are a helpful chatbot assistant. Possible chatbot prompt commands: " + print_possible_prompt_commands()

    user_name = ctx.user_name or utils.get_personal_info()["user_name"]
    if ctx.anonymous:
        user_name = ""
        if ctx.verbose:
            plugins.on_print("User name not used.", Fore.WHITE + Style.DIM)

    # Set the current collection
    vector_db.set_current_collection(ctx.current_collection_name,  ctx=ctx)

    # Initial system message
    if ctx.initial_system_prompt:
        if ctx.verbose:
            plugins.on_print(
                "Initial system prompt: " + ctx.initial_system_prompt, Fore.WHITE + Style.DIM
            )
        system_prompt = ctx.initial_system_prompt

    if not ctx.no_system_role and len(user_name) > 0:
        first_name = user_name.split()[0]
        system_prompt += f"\nThe user's name is {user_name}, first name: {first_name}. {today}"

    if len(system_prompt) > 0:
        # Replace placeholders in the system_prompt using the system_prompt_placeholders dictionary
        for key, value in system_prompt_placeholders.items():
            system_prompt = system_prompt.replace(f"{{{{{key}}}}}", value)

        ctx.initial_message = {"role": "system", "content": system_prompt}
        conversation = [ctx.initial_message]
    else:
        ctx.initial_message = None
        conversation = []

    ctx.current_model = selected_model

    answer_and_exit = False
    if not ctx.interactive_mode and ctx.user_prompt:
        answer_and_exit = True

    if ctx.use_memory_manager:
        #main.load_chroma_client(ctx=ctx)

        if ctx.chroma_client:
            ctx.memory_manager = MemoryManager(
                ctx.memory_collection_name,
                ctx.chroma_client,
                ctx.current_model,
                ctx.embeddings_model,
                ctx.verbose,
                num_ctx=num_ctx,
                long_term_memory_file=ctx.long_term_memory_file,
                ctx=ctx
            )

            if ctx.initial_message:
                # Add long-term memory to the system prompt
                long_term_memory = ctx.memory_manager.long_term_memory_manager.memory

                ctx.initial_message["content"] += f"\n\nLong-term memory: {long_term_memory}"
        else:
            ctx.use_memory_manager = False

    if ctx.initial_message and ctx.verbose:
        plugins.on_print("System prompt: " + ctx.initial_message["content"], Fore.WHITE + Style.DIM)

    user_input = ""

    if "tools" in chatbot and len(chatbot["tools"]) > 0:
        # Append chatbot tools to selected_tools if not already in the array
        if ctx.selected_tools is None:
            ctx.selected_tools = []

        for tool in chatbot["tools"]:
            ctx.selected_tools = plugins.tool_manager.select_tool_by_name(
                plugins.tool_manager.get_available_tools(ctx=ctx),
               tool,
               ctx=ctx
            )

    selected_tool_names = ctx.tools.split(',') if ctx.tools else []
    for tool_name in selected_tool_names:
        # Strip any leading or trailing spaces, single or double quotes
        tool_name = tool_name.strip().strip('\'').strip('\"')
        ctx.selected_tools = plugins.tool_manager.select_tool_by_name(
            plugins.tool_manager.get_available_tools(ctx=ctx),
            tool_name,
            ctx=ctx
        )

    # Handle web search if requested (after model initialization)
    if ctx.web_search:
        web_search.web_search2(
            show_intermediate=ctx.web_search_show_intermediate,
            num_ctx=num_ctx,
            ctx=ctx
        )

        # If not in interactive mode, exit after web search
        if not ctx.interactive_mode:
            sys.exit(0)


    # Main conversation loop
    while True:
        thoughts = None
        if not ctx.auto_start:
            try:
                if ctx.interactive_mode:
                    on_prompt("\nYou: ", Fore.YELLOW + Style.NORMAL)

                if ctx.user_prompt:
                    if ctx.other_instance_url:
                        conversation.append({"role": "assistant", "content": ctx.user_prompt})
                        user_input = plugins.on_user_input(ctx.user_prompt)
                    else:
                        user_input = ctx.user_prompt
                    ctx.user_prompt = None
                else:
                    user_input = plugins.on_user_input()

                if user_input.strip().startswith('"""'):
                    multi_line_input = [user_input[3:]]  # Keep the content after the first """
                    plugins.on_stdout_write("... ")  # Prompt continuation line

                    while True:
                        line = plugins.on_user_input()
                        if line.strip().endswith('"""') and len(line.strip()) > 3:
                            # Handle if the line contains content before """
                            multi_line_input.append(line[:-3])
                            break
                        if line.strip().endswith('"""'):
                            break

                        multi_line_input.append(line)
                        plugins.on_stdout_write("... ")  # Prompt continuation line

                    user_input = "\n".join(multi_line_input)

            except EOFError:
                break
            except KeyboardInterrupt:
                ctx.auto_save = False
                plugins.on_print("\nGoodbye!", Style.RESET_ALL)
                break

            if len(user_input.strip()) == 0:
                continue

        # Exit condition
        if (
            user_input.lower() in ['/quit', '/exit', '/bye', 'quit', 'exit', 'bye', 'goodbye', 'stop']
            or re.search(r'\b(bye|goodbye)\b', user_input, re.IGNORECASE)
        ):
            plugins.on_print("Goodbye!", Style.RESET_ALL)
            if ctx.memory_manager:
                plugins.on_print("Saving conversation to memory...", Fore.WHITE + Style.DIM)
                if ctx.memory_manager.add_memory(conversation,  ctx=ctx):
                    plugins.on_print("Conversation saved to memory.", Fore.WHITE + Style.DIM)
                    plugins.on_print("", Style.RESET_ALL)
            break

        if user_input.lower() in ['/reset', '/clear', '/restart', 'reset', 'clear', 'restart']:
            plugins.on_print("Conversation reset.", Style.RESET_ALL)
            if ctx.initial_message:
                conversation = [ctx.initial_message]
            else:
                conversation = []

            ctx.auto_start = (
                ("starts_conversation" in chatbot and chatbot["starts_conversation"])
                or ctx.auto_start
            )
            user_input = ""
            continue

        for plugin in plugins.plugin_manager.plugins:
            if hasattr(plugin, "on_user_input_done") and callable(getattr(plugin, "on_user_input_done")):
                user_input_from_plugin = plugin.on_user_input_done(user_input, verbose_mode=ctx.verbose)
                if user_input_from_plugin:
                    user_input = user_input_from_plugin

        # Allow for /context command to be used to set the context window size
        if user_input.startswith("/context"):
            if re.search(r'/context\s+\d+', user_input):
                context_window = int(re.search(r'/context\s+(\d+)', user_input).group(1))
                max_context_length = 125 # 125 * 1024 = 128000 tokens
                if context_window < 0 or context_window > max_context_length:
                    plugins.on_print(
                        f"Context window must be between 0 and {max_context_length}.", Fore.RED
                    )
                else:
                    num_ctx = context_window * 1024
                    if ctx.verbose:
                        plugins.on_print(
                            f"Context window changed to {num_ctx} tokens.", Fore.WHITE + Style.DIM
                        )
            else:
                plugins.on_print(
                    "Please specify context window size with /context <number>.", Fore.RED
                )
            continue

        if "/system" in user_input:
            system_prompt = user_input.replace("/system", "").strip()

            if len(system_prompt) > 0:
                # Replace placeholders in the system_prompt using the system_prompt_placeholders dictionary
                for key, value in system_prompt_placeholders.items():
                    system_prompt = system_prompt.replace(f"{{{{{key}}}}}", value)

                if ctx.verbose:
                    plugins.on_print("System prompt: " + system_prompt, Fore.WHITE + Style.DIM)

                for entry in conversation:
                    if "role" in entry and entry["role"] == "system":
                        entry["content"] = system_prompt
                        break
            continue

        if "/index" in user_input:
            if not ctx.chroma_client:
                plugins.on_print("ChromaDB client not initialized.", Fore.RED)
                continue

            if not ctx.current_collection_name:
                plugins.on_print("No ChromaDB collection loaded.", Fore.RED)

                collection_name, collection_description = vector_db.prompt_for_vector_database_collection(ctx=ctx)
                vector_db.set_current_collection(collection_name, collection_description, ctx=ctx)

            folder_to_index = user_input.split("/index")[1].strip()
            temp_folder = None
            if folder_to_index.startswith("http"):
                base_url = folder_to_index
                temp_folder = tempfile.mkdtemp()
                scraper = SimpleWebScraper(
                    base_url,
                    output_dir=temp_folder,
                    file_types=["html", "htm"],
                    restrict_to_base=True,
                    convert_to_markdown=True,
                    verbose=ctx.verbose
                )
                scraper.scrape()
                folder_to_index = temp_folder

            document_indexer = DocumentIndexer(
                folder_to_index,
                ctx.current_collection_name,
                ctx.chroma_client,
                ctx.embeddings_model,
                verbose=ctx.verbose,
                summary_model=ctx.current_model
            )
            document_indexer.index_documents(num_ctx=num_ctx,  ctx=ctx)

            if temp_folder:
                # Remove the temporary folder and its contents
                for file in os.listdir(temp_folder):
                    file_path = os.path.join(temp_folder, file)
                    os.remove(file_path)
                os.rmdir(temp_folder)
            continue

        if user_input == "/verbose":
            ctx.verbose = not ctx.verbose
            plugins.on_print(f"Verbose mode: {ctx.verbose}", Fore.WHITE + Style.DIM)
            continue

        if "/cot" in user_input:
            user_input = user_input.replace("/cot", "").strip()
            chain_of_thoughts_system_prompt = generate_chain_of_thoughts_system_prompt(ctx.selected_tools)

            # Format the current conversation as user/assistant messages
            formatted_conversation = "\n".join([f"{entry['role']}: {entry['content']}" for entry in conversation if "content" in entry and entry["content"] and "role" in entry and entry["role"] != "system" and entry["role"] != "tool"])
            formatted_conversation += "\n\n" + user_input

            thoughts = ask_ollama(
                chain_of_thoughts_system_prompt,
                formatted_conversation,
                ctx.thinking_model,
                ctx.temperature,
                ctx.prompt_template,
                no_bot_prompt=True,
                stream_active=False,
                num_ctx=num_ctx,
                ctx=ctx
            )

        if "/search" in user_input:
            # If /search is followed by a number, use that number as the number of documents to return (/search can be anywhere in the prompt)
            if re.search(r'/search\s+\d+', user_input):
                n_docs_to_return = int(re.search(r'/search\s+(\d+)', user_input).group(1))
                user_input = user_input.replace(f"/search {n_docs_to_return}", "").strip()
            else:
                user_input = user_input.replace("/search", "").strip()
                n_docs_to_return = ctx.number_of_documents_to_return_from_vector_db

            answer_from_vector_db = vector_db.query_vector_database(
                user_input,
                collection_name=ctx.current_collection_name,
                n_results=n_docs_to_return,
                ctx=ctx
            )
            if answer_from_vector_db:
                initial_user_input = user_input
                user_input = "Question: " + initial_user_input
                user_input += "\n\nAnswer the question as truthfully as possible using the provided text below, and if the answer is not contained within the text below, say 'I don't know'.\n\n"
                user_input += answer_from_vector_db
                user_input += "\n\nAnswer the question as truthfully as possible using the provided text above, and if the answer is not contained within the text above, say 'I don't know'."
                user_input += "\nQuestion: " + initial_user_input

                if ctx.verbose:
                    plugins.on_print(user_input, Fore.WHITE + Style.DIM)

        elif "/web" in user_input:
            user_input = user_input.replace("/web", "").strip()
            web_search_response = web_search.web_search(
                user_input,
                num_ctx=num_ctx,
                web_embedding_model=ctx.embeddings_model,
                ctx=ctx
            )
            if web_search_response:
                initial_user_input = user_input
                user_input += "Context: " + web_search_response
                user_input += "\n\nQuestion: " + initial_user_input
                user_input += "\nAnswer the question as truthfully as possible using the web search results in the provided context, if the answer is not contained within the provided context say 'I don't know'.\n"
                user_input += "Cite some useful links from the search results to support your answer."

                if ctx.verbose:
                    plugins.on_print(user_input, Fore.WHITE + Style.DIM)

        if user_input == "/thinking_model":
            selected_model = prompt_for_model(default_model, ctx.thinking_model, ctx=ctx)
            ctx.thinking_model = selected_model
            continue

        if user_input == "/model":
            on_model_command(default_model, num_ctx,  ctx=ctx)
            continue

        if user_input == "/memory":
            on_memoy_command(num_ctx, ctx=ctx)
            continue

        if user_input == "/model2":
            on_model2_command(default_model, ctx=ctx)
            continue

        if user_input == "/tools":
            ctx.selected_tools = plugins.tool_manager.select_tools(
                plugins.tool_manager.get_available_tools(ctx=ctx), ctx.selected_tools
            )
            continue

        if "/save" in user_input:
            save_conversation(user_input, conversation, ctx=ctx)
            continue

        if "/load" in user_input:
            # If the user input contains /load and followed by a filename, load the conversation from that file (assumed to be a JSON file)
            file_path = user_input.split("/load")[1].strip()
            # Remove any leading or trailing spaces, single or double quotes
            file_path = file_path.strip().strip('\'').strip('\"')

            if file_path:
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding="utf8") as f:
                        conversation = json.load(f)

                        system_prompt = ""
                        ctx.initial_message = None

                        # Find system prompt in the conversation
                        for entry in conversation:
                            if "role" in entry and entry["role"] == "system":
                                system_prompt = entry["content"]
                                ctx.initial_message = {"role": "system", "content": system_prompt}
                                break

                        # Reformat each entry tool_calls.function.arguments to be a valid dictionary, unless it's already a dictionary
                        for entry in conversation:
                            if "tool_calls" in entry:
                                for tool_call in entry["tool_calls"]:
                                    if "function" in tool_call and "arguments" in tool_call["function"]:
                                        if isinstance(tool_call["function"]["arguments"], str):
                                            try:
                                                tool_call["function"]["arguments"] = json.loads(
                                                    tool_call["function"]["arguments"]
                                                )
                                            except json.JSONDecodeError:
                                                pass

                    plugins.on_print(f"Conversation loaded from {file_path}", Fore.WHITE + Style.DIM)
                else:
                    plugins.on_print(f"Conversation file '{file_path}' not found.", Fore.RED)
            else:
                plugins.on_print("Please specify a file path to load the conversation.", Fore.RED)
            continue

        if user_input == "/collection":
            collection_name, collection_description = vector_db.prompt_for_vector_database_collection(ctx=ctx)
            vector_db.set_current_collection(collection_name, collection_description, ctx=ctx)
            continue

        if ctx.memory_manager and (user_input in ('/remember', '/memorize')):
            plugins.on_print("Saving conversation to memory...", Fore.WHITE + Style.DIM)
            if ctx.memory_manager.add_memory(conversation):
                plugins.on_print("Conversation saved to memory.", Fore.WHITE + Style.DIM)
                plugins.on_print("", Style.RESET_ALL)
            continue

        if ctx.memory_manager and user_input == "/forget":
            # Reset memory collection
            ctx.memory_manager.reset_memory(ctx=ctx)
            continue

        if "/rmcollection" in user_input or "/deletecollection" in user_input:
            if "/rmcollection" in user_input and len(user_input.split("/rmcollection")) > 1:
                collection_name = user_input.split("/rmcollection")[1].strip()

            if (
                not collection_name
                and "/deletecollection" in user_input
                and len(user_input.split("/deletecollection")) > 1
            ):
                collection_name = user_input.split("/deletecollection")[1].strip()

            if not collection_name:
                collection_name, _ = vector_db.prompt_for_vector_database_collection(
                    prompt_create_new=False, include_web_cache=True,  ctx=ctx
                )

            if not collection_name:
                continue

            vector_db.delete_collection(collection_name,  ctx=ctx)
            continue

        if "/editcollection" in user_input:
            collection_name, _ = vector_db.prompt_for_vector_database_collection(ctx=ctx)
            vector_db.edit_collection_metadata(collection_name,  ctx=ctx)
            continue

        if user_input == "/chatbot":

            chatbot = prompt_for_chatbot(ctx=ctx)
            if "tools" in chatbot and len(chatbot["tools"]) > 0:
                # Append chatbot tools to selected_tools if not already in the array
                if ctx.selected_tools is None:
                    ctx.selected_tools = []

                for tool in chatbot["tools"]:
                    ctx.selected_tools = plugins.tool_manager.select_tool_by_name(
                        plugins.tool_manager.get_available_tools(ctx=ctx), tool,  ctx=ctx
                    )

            system_prompt = chatbot["system_prompt"]
            # Initial system message
            if not ctx.no_system_role and len(user_name) > 0:
                first_name = user_name.split()[0]
                system_prompt += f"\nThe user's name is {user_name}, first name: {first_name}. {today}"

            if len(system_prompt) > 0:
                # Replace placeholders in the system_prompt using the system_prompt_placeholders dictionary
                for key, value in system_prompt_placeholders.items():
                    system_prompt = system_prompt.replace(f"{{{{{key}}}}}", value)

                if ctx.verbose:
                    plugins.on_print("System prompt: " + system_prompt, Fore.WHITE + Style.DIM)

                ctx.initial_message = {"role": "system", "content": system_prompt}
                conversation = [ctx.initial_message]
            else:
                conversation = []
            plugins.on_print("Conversation reset.", Style.RESET_ALL)
            ctx.auto_start = (
                ("starts_conversation" in chatbot and chatbot["starts_conversation"])
                or ctx.auto_start
            )
            user_input = ""
            continue

        if "/cb" in user_input:
            if platform.system() == "Windows":
                # Replace /cb with the clipboard content
                win32clipboard.OpenClipboard()
                clipboard_content = win32clipboard.GetClipboardData()
                win32clipboard.CloseClipboard()
            else:
                clipboard_content = pyperclip.paste()
            user_input = user_input.replace("/cb", "\n" + clipboard_content + "\n")
            plugins.on_print("Clipboard content added to user input.", Fore.WHITE + Style.DIM)

        image_path = None
        # If user input contains '/file <path of a file to load>' anywhere in the prompt, read the file and append the content to user_input
        if "/file" in user_input:
            file_path = user_input.split("/file")[1].strip()
            file_path = file_path.strip("'\"")

            # Check if the file is an image
            _, ext = os.path.splitext(file_path)
            if ext.lower() not in [".png", ".jpg", ".jpeg", ".bmp"]:
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        user_input = user_input.replace("/file", "")
                        user_input += "\n" + file.read()
                except FileNotFoundError:
                    plugins.on_print("File not found. Please try again.", Fore.RED)
                    continue
            else:
                user_input = user_input.split("/file")[0].strip()
                image_path = file_path

        if user_input == "/think":
            on_think_command(ctx=ctx)
            continue

        # If user input starts with '/' and is not a command, ignore it.
        if user_input.startswith('/') and not user_input.startswith('//'):
            plugins.on_print("Invalid command. Please try again.", Fore.RED)
            continue

        # Add user input to conversation history
        if image_path:
            conversation.append({"role": "user", "content": user_input, "images": [image_path]})
        elif len(user_input.strip()) > 0:
            conversation.append({"role": "user", "content": user_input})

        # If the conversation memory is enabled then make it handle the conversation by recalling and
        # injecting memories related to the current user query
        if ctx.memory_manager:
            ctx.memory_manager.handle_user_query(conversation)

        if thoughts:
            thoughts = f"Thinking...\n{thoughts}\nEnd of internal thoughts.\n\nFinal response:"
            if ctx.syntax_highlighting:
                plugins.on_print(
                    utils.colorize(thoughts),
                    Style.RESET_ALL,
                    "\rBot: " if ctx.interactive_mode else ""
                )
            else:
                plugins.on_print(
                    thoughts, Style.RESET_ALL, "\rBot: " if ctx.interactive_mode else ""
                )

            # Add the chain of thoughts to the conversation, as an assistant message
            conversation.append({"role": "assistant", "content": thoughts})

        # Generate response
        bot_response = ask_ollama_with_conversation(
            conversation,
            selected_model,
            temperature=ctx.temperature,
            prompt_template=ctx.prompt_template,
            tools=ctx.selected_tools,
            stream_active=ctx.stream,
            num_ctx=num_ctx,
            ctx=ctx
        )

        # Generate also the alternate_bot_response if an alternate_model is defined
        alternate_bot_response = None
        if ctx.alternate_model:
            alternate_bot_response = ask_ollama_with_conversation(
                conversation,
                ctx.alternate_model,
                temperature=ctx.temperature,
                prompt_template=ctx.prompt_template,
                tools=ctx.selected_tools,
                prompt="\nAlt",
                prompt_color=Fore.CYAN,
                stream_active=ctx.stream,
                num_ctx=num_ctx,
                ctx=ctx
            )

        bot_response_handled_by_plugin = False
        for plugin in plugins.plugin_manager.plugins:
            if hasattr(plugin, "on_llm_response") and callable(getattr(plugin, "on_llm_response")):
                plugin_response = getattr(plugin, "on_llm_response")(bot_response)
                bot_response_handled_by_plugin = bot_response_handled_by_plugin or plugin_response

        if not bot_response_handled_by_plugin:
            if ctx.syntax_highlighting:
                plugins.on_print(
                    utils.colorize(bot_response),
                    Style.RESET_ALL,
                    "\rBot: " if ctx.interactive_mode else ""
                )

                if alternate_bot_response:
                    plugins.on_print(
                        utils.colorize(alternate_bot_response),
                        Fore.CYAN,
                        "\rAlt: " if ctx.interactive_mode else ""
                    )

            elif not ctx.use_openai and not ctx.use_azure_openai and len(ctx.selected_tools) > 0:
                # Ollama cannot stream when tools are used
                plugins.on_print(
                    bot_response,
                    Style.RESET_ALL,
                    "\rBot: " if ctx.interactive_mode else ""
                )

                if alternate_bot_response:
                    plugins.on_print(
                        alternate_bot_response,
                        Fore.CYAN,
                        "\rAlt: " if ctx.interactive_mode else ""
                    )

        # If there is an alternate_bot_response prompts the user to choose the preferred response
        if alternate_bot_response:
            # Ask user to select the preferred response
            plugins.on_print(
                f"Select the preferred response:\n1. Original model ({ctx.current_model})\n2. Alternate model ({ctx.alternate_model})",
                Fore.WHITE + Style.DIM
            )
            choice = plugins.on_user_input("Enter the number of your preferred response [1]: ") or "1"
            bot_response = bot_response if choice == "1" else alternate_bot_response

        # Add bot response to conversation history
        conversation.append({"role": "assistant", "content": bot_response})

        if ctx.auto_start:
            ctx.auto_start = False

        if ctx.output:
            if bot_response:
                with open(ctx.output, 'a', encoding='utf-8') as f:
                    f.write(bot_response)
                    if ctx.verbose:
                        plugins.on_print(f"Response saved to {ctx.output}", Fore.WHITE + Style.DIM)
            else:
                plugins.on_print("No bot response to save.", Fore.YELLOW)

        # Exit condition: if the bot response contains an exit command ('bye', 'goodbye'), using a regex pattern to match the words
        if bot_response and re.search(r'\b(bye|goodbye)\b', bot_response, re.IGNORECASE):
            plugins.on_print("Goodbye!", Style.RESET_ALL)
            break

        if answer_and_exit:
            break

    # Stop plugins, calling on_exit if available
    for plugin in plugins.plugin_manager.plugins:
        if hasattr(plugin, "on_exit") and callable(getattr(plugin, "on_exit")):
            getattr(plugin, "on_exit")()

    # Close global full document store if initialized
    if ctx.full_doc_store:
        try:
            ctx.full_doc_store.close()
            if ctx.verbose:
                plugins.on_print("Closed global full document store.", Fore.WHITE + Style.DIM)
        except Exception as e:
            plugins.on_print(f"Warning: Error closing full document store: {e}", Fore.YELLOW)

    # If auto-save of conversations is enabled then save the current conversation
    if ctx.auto_save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if ctx.conversations_folder:
            save_conversation_to_file(
                conversation,
                os.path.join(ctx.conversations_folder,f"conversation_{timestamp}.txt"),
                ctx=ctx
            )
        else:
            save_conversation_to_file(conversation, f"conversation_{timestamp}.txt",  ctx=ctx)


def save_conversation(user_input, conversation, *, ctx:Context):

    # If the user input contains /save and followed by a filename, save the conversation to that file
    file_path = user_input.split("/save")[1].strip()
    # Remove any leading or trailing spaces, single or double quotes
    file_path = file_path.strip().strip('\'').strip('\"')

    if file_path:
        # If file_path does not contain a path than use the conversations folder
        if (not os.path.sep in file_path) and ctx.conversations_folder:
            file_path = os.path.join(ctx.conversations_folder, file_path)

        save_conversation_to_file(conversation, file_path, ctx=ctx)
    else:
        # Save the conversation to a file, use current timestamp as the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if ctx.conversations_folder:
            save_conversation_to_file(
                conversation,
                os.path.join(ctx.conversations_folder, f"conversation_{timestamp}.txt"),
                ctx=ctx
            )
        else:
            save_conversation_to_file(conversation, f"conversation_{timestamp}.txt", ctx=ctx)


def save_conversation_to_file(conversation, file_path, *,  ctx:Context):
    """
    Saves the current conversation to file. Generates two files:
    - A txt version
    - A JSON version that includes the system prompt
    """

    # Convert conversation list of objects to a list of dict
    conversation = [json.loads(json.dumps(obj, default=lambda o: vars(o))) for obj in conversation]

    # Check if the filename contains a folder path (use os path separator to check)
    if os.path.sep in file_path:
        # Get the folder path and filename
        folder_path, _ = os.path.split(file_path)
        # Create the folder if it doesn't exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    # Save the conversation to a text file (filter out system messages)
    with open(file_path, 'w', encoding="utf8") as f:
        # Skip empty messages or system messages
        filtered_conversation = (
            [entry for entry in conversation if "content" in entry and entry["content"] and "role" in entry and entry["role"] != "system" and entry["role"] != "tool"]
        )

        for message in filtered_conversation:
            role = message["role"]

            if role == "user":
                role = "Me"
            elif role == "assistant":
                role = "Assistant"

            f.write(f"{role}: {message['content']}\n\n")

    if ctx.verbose:
        plugins.on_print(f"Conversation saved to {file_path}", Fore.WHITE + Style.DIM)

    # Save the conversation to a JSON file
    json_file_path = file_path.replace(".txt", ".json")
    with open(json_file_path, 'w', encoding="utf8") as f:
        json.dump(conversation, f, indent=4)

    if ctx.verbose:
        plugins.on_print(f"Conversation saved to {json_file_path}", Fore.WHITE + Style.DIM)


def on_model_command(default_model, num_ctx,  *, ctx:Context):
    thinking_model_is_same = ctx.thinking_model == ctx.current_model

    if ctx.use_azure_openai:
        # For Azure OpenAI, just ask for the deployment name
        selected_model = plugins.on_user_input(f"Enter Azure OpenAI deployment name [{ctx.current_model}]: ").strip() or ctx.current_model
    else:
        selected_model = prompt_for_model(default_model, ctx.current_model, ctx=ctx)

    ctx.current_model = selected_model

    if thinking_model_is_same:
        ctx.thinking_model = selected_model

    if ctx.use_memory_manager:
        #main.load_chroma_client(ctx=ctx)

        if ctx.chroma_client:
            ctx.memory_manager = MemoryManager(
                ctx.memory_collection_name,
                ctx.chroma_client,
                ctx.current_model,
                ctx.embeddings_model,
                ctx.verbose,
                num_ctx=num_ctx,
                long_term_memory_file=ctx.long_term_memory_file,
                ctx=ctx
            )
        else:
            ctx.use_memory_manager = False

def on_think_command(*, ctx:Context):
    if not ctx.think_mode_on:
        ctx.think_mode_on = True
        if ctx.verbose:
            plugins.on_print("Think mode activated.", Fore.WHITE + Style.DIM)
    else:
        ctx.think_mode_on = False
        if ctx.verbose:
            plugins.on_print("Think mode deactivated.", Fore.WHITE + Style.DIM)


def on_model2_command(default_model, *, ctx:Context):
    if ctx.use_azure_openai:
        # For Azure OpenAI, just ask for the deployment name
        current_alt = ctx.alternate_model if ctx.alternate_model else ctx.current_model
        ctx.alternate_model = plugins.on_user_input(f"Enter Azure OpenAI deployment name for alternate model [{current_alt}]: ").strip() or current_alt
    else:
        ctx.alternate_model = prompt_for_model(default_model, ctx.current_model, ctx=ctx)


def on_memoy_command(num_ctx, *, ctx:Context):
    if ctx.use_memory_manager:
        # Deactivate memory manager
        ctx.memory_manager = None
        ctx.use_memory_manager = False
        plugins.on_print("Memory manager deactivated.", Fore.WHITE + Style.DIM)
    else:
        #main.load_chroma_client(ctx=ctx)

        if ctx.chroma_client:
            ctx.memory_manager = MemoryManager(
                ctx.memory_collection_name,
                ctx.chroma_client,
                ctx.current_model,
                ctx.embeddings_model,
                ctx.verbose,
                num_ctx=num_ctx,
                long_term_memory_file=ctx.long_term_memory_file,
                ctx=ctx
            )
            ctx.use_memory_manager = True
            plugins.on_print("Memory manager activated.", Fore.WHITE + Style.DIM)
        else:
            plugins.on_print("ChromaDB client not initialized.", Fore.RED)


#if __name__ == "__main__":
 #   run()
