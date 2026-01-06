from dataclasses import dataclass, field
from typing import List,  TYPE_CHECKING,  Optional

from chromadb.api.client import ClientAPI

# Way to have typr checking for class MemoryManager avoiding circular import
if TYPE_CHECKING:
    from memory_manager import MemoryManager

@dataclass
class Context:
    """
    This object is the unique store for application-wide parameters and settings.
    """
    # TODO! Do a clear separation of the conversation memory and long-term memory: the user shoud can manage each separately. I imagine each coud be set to on,off or read only.
    # TODO! The same collection is used for web cache and conversation memory, consider if it may be better to separate them
    # Memory
    use_memory_manager : bool = False

    # Conversation memory
    memory_collection_name : str = ""

    # Long-term memory
    long_term_memory_file : str = ""

    # Models
    alternate_model : str = None
    current_model : str = None
    embeddings_model : str = None
    model : str = None
    thinking_model : str = None

    verbose: bool = False
    number_of_documents_to_return_from_vector_db : int = 8
    no_system_role:bool = False
    use_openai : bool = False
    use_azure_openai : bool = False
    syntax_highlighting : bool = True
    think_mode_on : bool = False
    interactive_mode : bool = True
    openai_client = None
    plugins_folder : str = None
    preferred_collection_name : str = None
    collection = None
    temperature : float = 0.1
    disable_plugins : bool = False
    prompt_template : str = None
    requested_tool_names : List[str] = field(default_factory=list)
    additional_chatbots_file : str = None
    initial_system_prompt : str = None
    allow_chunks : bool = True
    conversations_folder : str = None
    split_paragraphs : bool = False
    skip_existing : bool = True
    index_documents : str = None
    query : str = None
    extract_start : str = None
    extract_end : str = None
    auto_start : bool = False
    auto_save : bool = False
    prompt : str = None
    stream : bool = True
    output : str = None
    other_instance_url : str = None
    system_prompt_placeholders_json : str = None
    thinking_model_reasoning_pattern : str = None
    listening_port : int = 8000
    user_name : str = None
    anonymous : bool = False
    context_window : int = None
    chatbot : str = None
    full_docs_db : str = 'full_documents.db'
    chunk_documents : bool = True
    add_summary : bool = True
    catchup_full_docs : bool = False
    query_distance_threshold : float = 0.0
    initial_message : str = None
    user_prompt : str = None
    expand_query : bool = True
    include_full_docs : bool = False
    full_doc_store = None
    tools : str = None
    # TODO: These args, that serve just to print something and exit may not be managed in the context, it could be correct also to read args.x in the main code and exti.
    list_tools : bool = False
    list_collections : bool = False
    web_cache_collection_name : str = 'web_cache'
    current_collection_name : str = None
    selected_tools : List[str] = field(default_factory=list)
    session_created_files : List[str] = field(default_factory=list)
    chatbots : List[str] = field(default_factory=list)
    #memory_manager : MemoryManager = None
    memory_manager: Optional["MemoryManager"] = field(default=None)

    instantiate_agent : bool = False
    agent_task : str = None
    agent_system_prompt : str = None
    agent_tools : str = None
    agent_name : str = None
    agent_description : str = None

    web_search : str = None
    web_search_results : int = 5
    web_search_region : str = "wt-wt"
    web_search_show_intermediate : bool = False

    # ChromaDB settings
    #chroma_client_host : str = "localhost"
    #chroma_client_port : int = 8000
    #chroma_db_path : str = None
    chroma_client: ClientAPI = None

    # RAG optimization parameters
    # Minimum number of quality results required from cache before skipping web search
    min_quality_results_threshold : int = 5
    # Minimum average BM25 score required for cache results to skip web search
    # This ensures cached results have lexical/keyword relevance to the query
    min_average_bm25_threshold : float = 0.5
    # Minimum hybrid score required for individual results to be considered "quality"
    min_hybrid_score_threshold : float = 0.1
    # Percentile threshold for adaptive distance filtering (0-100)
    # Results with distance > this percentile will be filtered out
    distance_percentile_threshold : int = 75
    # Weight for semantic similarity vs BM25 in hybrid scoring (0.0 to 1.0)
    # 0.5 = equal weight, higher = more semantic, lower = more lexical
    semantic_weight : float = 0.5
    # Maximum distance multiplier for adaptive threshold
    # Results beyond min_distance * this multiplier are filtered
    adaptive_distance_multiplier : float = 2.5
