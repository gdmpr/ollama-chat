import re
import os
from datetime import datetime
import traceback


from tqdm import tqdm
import chromadb
import ollama
from colorama import Fore, Style
import numpy as np
from rank_bm25 import BM25Okapi

from ollama_chat.core import utils
from ollama_chat.core.context import Context
from ollama_chat.core.ollama import ask_ollama

def query_vector_database(question,  collection_name=None, n_results=None, answer_distance_threshold=0, query_embeddings_model=None, expand_query=True, question_context=None, use_adaptive_filtering=True, return_metadata=False, full_doc_store=None, include_full_docs=False, *,  ctx:Context):

    if collection_name is None:
        collection_name = ctx.current_collection_name

    #global collection

    # If full_doc_store not provided, use global instance
    if full_doc_store is None:
        full_doc_store = globals().get('full_doc_store', None)

    # Auto-enable full documents for OpenAI models (better context handling)
    # Keep disabled for Ollama models (performance reasons)
    if not include_full_docs and full_doc_store:
        if ctx.use_openai or ctx.use_azure_openai:
            include_full_docs = True
            if ctx.verbose:
                utils.on_print("Auto-enabled full document retrieval for OpenAI model", Fore.WHITE + Style.DIM)

    # If question is empty, return empty string
    if not question or len(question) == 0:
        if return_metadata:
            return "", {}
        return ""

    initial_question = question

    # Default valure for n_results
    n_results = ctx.number_of_documents_to_return_from_vector_db if n_results is None else n_results

    # If n_results is 0, return empty string
    if n_results == 0:
        if return_metadata:
            return "", {}
        return ""

    # If n_results is negative, set it to the default value
    if n_results < 0:
        n_results = ctx.number_of_documents_to_return_from_vector_db

    # If answer_distance_threshold is a string, convert it to a float
    if isinstance(answer_distance_threshold, str):
        try:
            answer_distance_threshold = float(answer_distance_threshold)
        except:
            answer_distance_threshold = 0

    # If answer_distance_threshold is negative, set it to 0
    if answer_distance_threshold < 0:
        answer_distance_threshold = 0

    if not query_embeddings_model:
        query_embeddings_model = ctx.embeddings_model

    if not ctx.collection and collection_name:
        set_current_collection(collection_name, create_new_collection_if_not_found=False,  ctx=ctx)

    if not ctx.collection:
        utils.on_print("No ChromaDB collection loaded.", Fore.RED)
        collection_name, _ = prompt_for_vector_database_collection(ctx=ctx)
        if not collection_name:
            if return_metadata:
                return "", {}
            return ""

    if collection_name and collection_name != ctx.current_collection_name:
        set_current_collection(collection_name, create_new_collection_if_not_found=False,  ctx=ctx)

    if expand_query:
        expanded_query = None
        # Expand the query for better retrieval
        system_prompt = "You are an assistant that helps expand and clarify user questions to improve information retrieval. When a user provides a question, your task is to write a short passage that elaborates on the query by adding relevant background information, inferred details, and related concepts that can help with retrieval. The passage should remain concise and focused, without changing the original meaning of the question.\r\nGuidelines:\r\n1. Expand the question briefly by including additional context or background, staying relevant to the user's original intent.\r\n2. Incorporate inferred details or related concepts that help clarify or broaden the query in a way that aids retrieval.\r\n3. Keep the passage short, usually no more than 2-3 sentences, while maintaining clarity and depth.\r\n4. Avoid introducing unrelated or overly specific topics. Keep the expansion concise and to the point."
        if question_context:
            system_prompt += f"\n\nAdditional context about the user query:\n{question_context}"

        if not ctx.thinking_model is None and ctx.thinking_model != ctx.current_model:
            if "deepseek-r1" in ctx.thinking_model:
                 # DeepSeek-R1 model requires an empty system prompt
                prompt = f"""{system_prompt}\n{question}"""
                expanded_query = ask_ollama(
                    "",
                    prompt,
                    selected_model=ctx.thinking_model,
                    no_bot_prompt=True,
                    stream_active=False,
                    ctx=ctx
                )
            else:
                expanded_query = ask_ollama(
                    system_prompt,
                    question,
                    selected_model=ctx.thinking_model,
                    no_bot_prompt=True,
                    stream_active=False,
                    ctx=ctx
                )
        else:
            expanded_query = ask_ollama(
                system_prompt,
                question,
                selected_model=ctx.current_model,
                no_bot_prompt=True,
                stream_active=False,
                ctx=ctx
            )
        if expanded_query:
            question += "\n" + expanded_query
            if ctx.verbose:
                utils.on_print("Expanded query:", Fore.WHITE + Style.DIM)
                utils.on_print(question, Fore.WHITE + Style.DIM)

    if ctx.verbose:
        utils.on_print(f"Using query embeddings model: {query_embeddings_model}", Fore.WHITE + Style.DIM)

    if query_embeddings_model is None:
        result = ctx.collection.query(
            query_texts=[question],
            n_results=25
        )
    else:
        # generate an embedding for the question and retrieve the most relevant doc
        response = ollama.embeddings(
            prompt=question,
            model=query_embeddings_model
        )
        result = ctx.collection.query(
            query_embeddings=[response["embedding"]],
            n_results=25
        )

    documents = result["documents"][0]
    distances = result["distances"][0]

    if len(result["metadatas"]) == 0:
        if return_metadata:
            return "", {}
        return ""

    if len(result["metadatas"][0]) == 0:
        if return_metadata:
            return "", {}
        return ""

    metadatas = result["metadatas"][0]

    # Adaptive filtering and hybrid scoring
    if use_adaptive_filtering and len(distances) > 0:
        # Calculate adaptive distance threshold based on result distribution
        min_distance = min(distances) if distances else 0
        adaptive_threshold = min_distance * ctx.adaptive_distance_multiplier

        # Also use percentile-based filtering
        if len(distances) >= 4:  # Only use percentile if we have enough results
            try:
                percentile_threshold = np.percentile(distances, ctx.distance_percentile_threshold)
                # Use the less restrictive of the two thresholds to keep more results
                # This prevents over-filtering when results are clustered
                effective_threshold = max(adaptive_threshold, percentile_threshold)
            except:
                effective_threshold = adaptive_threshold
        else:
            effective_threshold = adaptive_threshold

        if ctx.verbose:
            utils.on_print(f"Adaptive distance threshold: {effective_threshold:.4f} (min: {min_distance:.4f}, adaptive: {adaptive_threshold:.4f}, percentile: {percentile_threshold if len(distances) >= 4 else 'N/A'})", Fore.WHITE + Style.DIM)
    else:
        effective_threshold = float('inf')  # No filtering

    # Preprocess and re-rank using BM25
    initial_question_preprocessed = preprocess_text(initial_question)
    preprocessed_docs = [preprocess_text(doc) for doc in documents]

    # Apply BM25 re-ranking
    bm25 = BM25Okapi(preprocessed_docs)
    bm25_scores = bm25.get_scores(initial_question_preprocessed)

    # Normalize scores for hybrid fusion
    # Normalize distances (invert so higher is better, and scale to 0-1)
    max_dist = max(distances) if len(distances) > 0 and max(distances) > 0 else 1
    normalized_semantic_scores = [1 - (d / max_dist) for d in distances]

    # Normalize BM25 scores (scale to 0-1)
    # Convert to list to avoid numpy array boolean ambiguity
    bm25_scores_list = list(bm25_scores) if hasattr(bm25_scores, '__iter__') else []
    max_bm25 = max(bm25_scores_list) if len(bm25_scores_list) > 0 and max(bm25_scores_list) > 0 else 1
    normalized_bm25_scores = [score / max_bm25 for score in bm25_scores_list]

    # Combine scores with configurable weighting
    hybrid_scores = [
        ctx.semantic_weight * sem + (1 - ctx.semantic_weight) * lex
        for sem, lex in zip(normalized_semantic_scores, normalized_bm25_scores)
    ]

    # Sort by hybrid score and apply adaptive filtering
    reranked_results = []
    for idx, (metadata, distance, document, bm25_score, hybrid_score) in enumerate(
        zip(metadatas, distances, documents, bm25_scores_list, hybrid_scores)
    ):
        # Apply adaptive distance filtering
        if use_adaptive_filtering and distance > effective_threshold:
            if ctx.verbose:
                utils.on_print(f"Filtered out result with distance {distance:.4f} > {effective_threshold:.4f}", Fore.WHITE + Style.DIM)
            continue

        # Also apply user-specified threshold if provided
        if answer_distance_threshold > 0 and distance > answer_distance_threshold:
            if ctx.verbose:
                utils.on_print(f"Filtered out result with distance {distance:.4f} > {answer_distance_threshold:.4f}", Fore.WHITE + Style.DIM)
            continue

        reranked_results.append((idx, metadata, distance, document, bm25_score, hybrid_score))

    # Sort by hybrid score (descending)
    reranked_results.sort(key=lambda x: x[5], reverse=True)

    # Limit to n_results
    reranked_results = reranked_results[:n_results]

    # Join all possible answers into one string
    answers = []
    metadata_list = []
    full_documents_map = {}  # Track full documents by document_id

    for idx, metadata, distance, document, bm25_score, hybrid_score in reranked_results:
        if ctx.verbose:
            utils.on_print(f"Result - Distance: {distance:.4f}, BM25: {bm25_score:.4f}, Hybrid: {hybrid_score:.4f}", Fore.WHITE + Style.DIM)

        # Format the answer with the title, content, and URL
        title = metadata.get("title", "")
        url = metadata.get("url", "")
        file_path = metadata.get("filePath", "")
        doc_id = metadata.get("id", "")
        chunk_index = metadata.get("chunk_index")

        formatted_answer = document

        # If we have a full document store and this is a chunk, try to retrieve full document
        if include_full_docs and full_doc_store and doc_id:
            # Check if we haven't already fetched this full document
            if doc_id not in full_documents_map:
                full_content = full_doc_store.get_document(doc_id)
                if full_content:
                    full_documents_map[doc_id] = full_content
                    if ctx.verbose:
                        utils.on_print(f"Retrieved full document for: {doc_id}", Fore.WHITE + Style.DIM)

            # If we have the full document, include it
            if doc_id in full_documents_map:
                formatted_answer = f"[Chunk {chunk_index if chunk_index is not None else 'N/A'}]\n{document}\n\n[Full Document]\n{full_documents_map[doc_id]}"

        if title:
            formatted_answer = title + "\n" + formatted_answer
        if url:
            formatted_answer += "\nURL: " + url
        if file_path:
            formatted_answer += "\nFile Path: " + file_path

        answers.append(formatted_answer.strip())

        if return_metadata:
            metadata_list.append({
                'distance': distance,
                'bm25_score': bm25_score,
                'hybrid_score': hybrid_score,
                'metadata': metadata,
                'has_full_document': doc_id in full_documents_map if include_full_docs else False
            })

    result_text = '\n\n'.join(answers)

    if return_metadata:
        # Calculate average scores for quality assessment
        avg_bm25 = sum(x[4] for x in reranked_results) / len(reranked_results) if reranked_results else 0.0
        avg_hybrid = sum(x[5] for x in reranked_results) / len(reranked_results) if reranked_results else 0.0
        avg_distance = sum(x[2] for x in reranked_results) / len(reranked_results) if reranked_results else 0.0

        return result_text, {
            'num_results': len(answers),
            'results': metadata_list,
            'effective_threshold': effective_threshold if use_adaptive_filtering else None,
            'avg_bm25_score': avg_bm25,
            'avg_hybrid_score': avg_hybrid,
            'avg_distance': avg_distance,
            'full_documents_retrieved': len(full_documents_map) if include_full_docs else 0
        }

    return result_text

def preprocess_text(text):
    stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
        "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
        'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its',
        'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
        'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
        'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
        'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above',
        'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
        'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
        'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
        'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't",
        'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't",
        'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn',
        "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
        "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't",
        'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]


    # If text is empty, return empty list
    if not text or len(text) == 0:
        return []

    # Convert text to lowercasectx.retries
    text = text.lower()
    # Replace punctuation with spaces, excepting dots
    text = re.sub(r'[^\w\s.,]', ' ', text)
    # Replace '. ' and ', ' with space
    text = re.sub(r'\. |, ', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Tokenize the text
    words = text.split()
    # Remove dot from the end of words
    words = [word[:-1] if word.endswith('.') else word for word in words]
    # Remove stop words
    words = [word for word in words if word not in stop_words]

    return words



def set_current_collection(collection_name, description=None, create_new_collection_if_not_found=True,  *,  ctx:Context):
    #global collection

    load_chroma_client(ctx=ctx)

    if not collection_name or not ctx.chroma_client:
        ctx.collection = None
        ctx.current_collection_name = None
        return

    # Get or create the target collection
    try:
        if create_new_collection_if_not_found:
            ctx.collection = ctx.chroma_client.get_or_create_collection(
                name=collection_name,
                configuration={
                    "hnsw": {
                        "space": "cosine",
                        "ef_search": 1000,
                        "ef_construction": 1000
                    }
            })
        else:
            ctx.collection = ctx.chroma_client.get_collection(name=ctx.collection_name)

        # Update description metadata if provided
        if description:
            existing_metadata = ctx.collection.metadata or {}

            if description != existing_metadata.get("description"):
                existing_metadata["description"] = description
                ctx.collection.modify(metadata=existing_metadata)
                if ctx.verbose:
                    utils.on_print(f"Updated description for collection {collection_name}.", Fore.WHITE + Style.DIM)

        if ctx.verbose:
            utils.on_print(f"Collection {collection_name} loaded.", Fore.WHITE + Style.DIM)

        ctx.current_collection_name = collection_name
    except:
        raise Exception(f"Collection {collection_name} not found")


def prompt_for_vector_database_collection(prompt_create_new=True, include_web_cache=False,  *,  ctx:Context):

    load_chroma_client(ctx=ctx)

    # List existing collections
    collections = None
    if ctx.chroma_client:
        collections = ctx.chroma_client.list_collections()
    else:
        utils.on_print("ChromaDB is not running.", Fore.RED)

    if not collections:
        utils.on_print("No collections found", Fore.RED)
        new_collection_name = utils.on_user_input("Enter a new collection to create: ")
        new_collection_desc = utils.on_user_input("Enter a description for the new collection: ")
        return new_collection_name, new_collection_desc

    # Filter out collections based on parameters
    filtered_collections = []
    for collection in collections:
        # Always exclude memory collection
        if collection.name == ctx.memory_collection_name:
            continue
        # Exclude web cache collection unless explicitly included
        if collection.name == ctx.web_cache_collection_name and not include_web_cache:
            continue
        filtered_collections.append(collection)

    if not filtered_collections:
        utils.on_print("No collections found", Fore.RED)
        new_collection_name = utils.on_user_input("Enter a new collection to create: ")
        new_collection_desc = utils.on_user_input("Enter a description for the new collection: ")
        return new_collection_name, new_collection_desc

    # Ask user to choose a collection
    utils.on_print("Available collections:", Style.RESET_ALL)
    for i, collection in enumerate(filtered_collections):
        collection_name = collection.name

        if type(collection.metadata) == dict:
            collection_metadata = collection.metadata.get("description", "No description")
        else:
            collection_metadata = "No description"

        # Add indicator for web cache collection
        cache_indicator = " (Web Cache)" if collection_name == ctx.web_cache_collection_name else ""
        utils.on_print(f"{i}. {collection_name}{cache_indicator} - {collection_metadata}")

    if prompt_create_new:
        # Propose to create a new collection
        utils.on_print(f"{len(filtered_collections)}. Create a new collection")

    choice = int(utils.on_user_input("Enter the number of your preferred collection [0]: ") or 0)

    if prompt_create_new and choice == len(filtered_collections):
        new_collection_name = utils.on_user_input("Enter a new collection to create: ")
        new_collection_desc = utils.on_user_input("Enter a description for the new collection: ")
        return new_collection_name, new_collection_desc

    return filtered_collections[choice].name, None  # No new description needed for existing collections


def load_chroma_client(*,  ctx:Context):

    if ctx.chroma_client:
        return

    # Initialize the ChromaDB client
    try:
        if ctx.chroma_db_path:
            # Set environment variable ANONYMIZED_TELEMETRY to disable telemetry
            os.environ["ANONYMIZED_TELEMETRY"] = "0"
            ctx.chroma_client = chromadb.PersistentClient(path=ctx.chroma_db_path)
        elif ctx.chroma_client_host and 0 < ctx.chroma_client_port:
            ctx.chroma_client = chromadb.HttpClient(host=ctx.chroma_client_host, port=ctx.chroma_client_port)
        else:
            raise ValueError("Invalid Chroma client configuration")
    except:
        if ctx.verbose:
            utils.on_print("ChromaDB client could not be initialized. Please check the host and port.", Fore.RED + Style.DIM)
        ctx.chroma_client = None

def edit_collection_metadata(collection_name,  *,  ctx:Context):
    """
    Interactively edits the specified ChromaDB collection description.
    """

    load_chroma_client(ctx=ctx)

    if not collection_name or not ctx.chroma_client:
        utils.on_print("Invalid collection name or ChromaDB client not initialized.", Fore.RED)
        return

    try:
        collection = ctx.chroma_client.get_collection(name=collection_name)
        if isinstance(collection.metadata, dict):
            current_description = collection.metadata.get("description", "No description")
        else:
            current_description = "No description"
        utils.on_print(f"Current description: {current_description}")

        new_description = utils.on_user_input("Enter the new description: ")
        existing_metadata = collection.metadata or {}
        existing_metadata["description"] = new_description
        existing_metadata["updated"] = str(datetime.now())
        collection.modify(metadata=existing_metadata)

        utils.on_print(f"Description updated for collection {collection_name}.", Fore.GREEN)
    except:
        raise Exception(f"Collection {collection_name} not found")

def delete_collection(collection_name,  *,  ctx:Context):
    """
    Interactively deletes the specified collection in the ChromaDB database.
    """

    load_chroma_client(ctx=ctx)

    if not ctx.chroma_client:
        return

    # Ask for user confirmation before deleting
    confirmation = utils.on_user_input(f"Are you sure you want to delete the collection '{collection_name}'? (y/n): ").lower()

    if confirmation not in ('y', 'yes'):
        utils.on_print("Collection deletion canceled.", Fore.YELLOW)
        return

    try:
        ctx.chroma_client.delete_collection(name=collection_name)
        utils.on_print(f"Collection {collection_name} deleted.", Fore.GREEN)
    except Exception as e:
        utils.on_print(f"Collection {collection_name} not found: {e}.", Fore.RED)

def catchup_full_documents_from_chromadb(*, ctx:Context, verbose=False):
    """
    Extract filePath metadata from ChromaDB chunks and index full documents in SQLite.
    This is a catchup operation for documents that were chunked but not fully indexed.

    :param chroma_client: ChromaDB client instance
    :param collection_name: Name of the collection to process
    :param full_doc_store: FullDocumentStore instance
    :param verbose: Enable verbose logging
    """
    if verbose:
        utils.on_print(f"Starting catchup for collection: {ctx.collection_name}", Fore.CYAN)

    try:
        # Get the collection
        collection = ctx.chroma_client.get_collection(name=ctx.collection_name)

        # Get all embeddings from the collection
        # We need to fetch in batches to avoid memory issues
        all_results = collection.get(include=['metadatas'])

        if not all_results or not all_results.get('ids'):
            utils.on_print(f"No documents found in collection {ctx.collection_name}", Fore.YELLOW)
            return 0

        ids = all_results['ids']
        metadatas = all_results['metadatas']

        if verbose:
            utils.on_print(f"Found {len(ids)} embeddings in collection", Fore.WHITE + Style.DIM)

        # Extract unique document IDs and file paths from chunk metadata
        # Chunk IDs follow pattern: {document_id}_{chunk_index}
        document_files = {}  # document_id -> file_path

        for embedding_id, metadata in zip(ids, metadatas):
            if not metadata:
                continue

            # Get the base document ID (remove chunk suffix if present)
            if 'id' in metadata:
                doc_id = metadata['id']
            else:
                # Try to extract from embedding_id
                # Pattern: document_id_chunk_index
                parts = embedding_id.rsplit('_', 1)
                if len(parts) == 2 and parts[1].isdigit():
                    doc_id = parts[0]
                else:
                    doc_id = embedding_id

            # Get file path from metadata
            file_path = metadata.get('filePath')

            if doc_id and file_path:
                # Only process if we haven't seen this document yet
                if doc_id not in document_files:
                    document_files[doc_id] = file_path

        if verbose:
            utils.on_print(f"Found {len(document_files)} unique documents to process", Fore.WHITE + Style.DIM)

        # Process each document
        indexed_count = 0
        skipped_count = 0
        error_count = 0

        progress_bar = None
        if verbose:
            progress_bar = tqdm(total=len(document_files), desc="Indexing full documents", unit="doc", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}")

        for doc_id, file_path in document_files.items():
            if progress_bar:
                progress_bar.update(1)

            try:
                # Skip if already indexed
                if ctx.full_doc_store.document_exists(doc_id):
                    skipped_count += 1
                    if verbose:
                        utils.on_print(f"Skipping already indexed document: {doc_id}", Fore.WHITE + Style.DIM)
                    continue

                # Check if file still exists
                if not os.path.exists(file_path):
                    if verbose:
                        utils.on_print(f"File not found: {file_path}", Fore.YELLOW)
                    error_count += 1
                    continue

                # Read the full document content
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        full_content = f.read()
                except UnicodeDecodeError:
                    # Try with different encoding
                    try:
                        with open(file_path, 'r', encoding='latin-1') as f:
                            full_content = f.read()
                    except Exception as e:
                        if verbose:
                            utils.on_print(f"Error reading file {file_path}: {e}", Fore.RED)
                        error_count += 1
                        continue

                # Store in full document store
                if ctx.full_doc_store.store_document(doc_id, full_content, file_path):
                    indexed_count += 1
                else:
                    error_count += 1

            except Exception as e:
                utils.on_print(f"Error processing document {doc_id}: {e}", Fore.RED)
                error_count += 1
                continue

        if progress_bar:
            progress_bar.close()

        # Print summary
        utils.on_print("\nCatchup completed:", Fore.GREEN)
        utils.on_print(f"  Indexed: {indexed_count} documents", Fore.GREEN)
        utils.on_print(f"  Skipped (already indexed): {skipped_count} documents", Fore.YELLOW)
        utils.on_print(f"  Errors: {error_count} documents", Fore.RED if error_count > 0 else Fore.WHITE)

        return indexed_count

    except Exception as e:
        utils.on_print(f"Error during catchup: {e}", Fore.RED)
        if verbose:
            traceback.print_exc()
        return 0


