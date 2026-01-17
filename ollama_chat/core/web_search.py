import tempfile
import re
import os
import sys

from colorama import Fore, Style
from ddgs import DDGS

#from ollama_chat.core import utils
from ollama_chat.core.ollama import ask_ollama
from ollama_chat.core.document_indexer import DocumentIndexer
from ollama_chat.core.simple_web_crawler import SimpleWebCrawler
from ollama_chat.core import plugins
from ollama_chat.core.context import Context
from ollama_chat.core.query_vector_database import query_vector_database
#from ollama_chat.core.query_vector_database import load_chroma_client


def web_search(
    query=None,
    n_results=5,
    region="wt-wt",
    web_embedding_model=None,
    num_ctx=None,
    return_intermediate=False,
    *,
    ctx:Context
):

    if web_embedding_model is None:
        web_embedding_model = ctx.embeddings_model

    web_cache_collection = ctx.web_cache_collection_name or "web_cache"

    if not query:
        if return_intermediate:
            return "", {}
        return ""

    # Initialize ChromaDB client if not already initialized
    #load_chroma_client(ctx=ctx)

    if not ctx.chroma_client:
        error_msg = "Web search requires ChromaDB to be running. Please start ChromaDB server or configure a persistent database path."
        if return_intermediate:
            return error_msg, {}
        return error_msg

    if web_embedding_model is None or web_embedding_model == "":
        web_embedding_model = ctx.embeddings_model

    # OPTIMIZATION: Check cache first before doing web search
    # This prevents redundant web crawling for similar/same queries
    cache_check_results = ""
    cache_metadata = {}
    skip_web_crawl = False

    try:
        # Try to get results from cache with metadata
        cache_check_results, cache_metadata = query_vector_database(
            query,
            collection_name=web_cache_collection,
            n_results=n_results * 2,  # Request more to check quality
            query_embeddings_model=web_embedding_model,
            use_adaptive_filtering=True,
            return_metadata=True,
            expand_query=False ,  # Disable query expansion for web cache to avoid misinterpretation
            ctx=ctx
        )

        # Determine if cache results are good enough to skip web crawling
        # Check both quantity AND quality (BM25/hybrid scores)
        if cache_metadata and 'num_results' in cache_metadata:
            num_quality_results = cache_metadata['num_results']
            avg_bm25 = cache_metadata.get('avg_bm25_score', 0.0)
            avg_hybrid = cache_metadata.get('avg_hybrid_score', 0.0)

            # Cache hit requires: enough results AND good lexical relevance
            quality_check = (
                num_quality_results >= ctx.min_quality_results_threshold and
                avg_bm25 >= ctx.min_average_bm25_threshold
            )

            if quality_check:
                skip_web_crawl = True
                if ctx.verbose:
                    plugins.on_print(f"Cache hit: Found {num_quality_results} quality results (avg BM25: {avg_bm25:.4f}, avg hybrid: {avg_hybrid:.4f}). Skipping web crawl.", Fore.GREEN + Style.DIM)
            else:
                if ctx.verbose:
                    reason = []
                    if num_quality_results < ctx.min_quality_results_threshold:
                        reason.append(f"only {num_quality_results}/{ctx.min_quality_results_threshold} results")
                    if avg_bm25 < ctx.min_average_bm25_threshold:
                        reason.append(f"low BM25 {avg_bm25:.4f} < {ctx.min_average_bm25_threshold}")
                    plugins.on_print(f"Cache insufficient: {', '.join(reason)}. Performing web crawl.", Fore.YELLOW + Style.DIM)

    except Exception as e:
        if ctx.verbose:
            plugins.on_print(f"Cache check failed: {str(e)}. Proceeding with web crawl.", Fore.YELLOW + Style.DIM)
        skip_web_crawl = False

    # If we have enough quality results from cache, return them
    if skip_web_crawl and cache_check_results:
        if return_intermediate:
            intermediate_data = {
                'cache_hit': True,
                'num_results': cache_metadata.get('num_results', 0),
                'search_results': [],
                'urls': [],
                'articles': [],
                'vector_db_results': cache_check_results
            }
            return cache_check_results, intermediate_data
        return cache_check_results

    # Proceed with web search and crawling
    search = DDGS()
    urls = []
    search_results_list = []
    # Add the search results to the chatbot response
    try:
        search_results = search.text(query, region=region, max_results=n_results)
        if search_results:
            for i, search_result in enumerate(search_results):
                urls.append(search_result['href'])
                search_results_list.append(search_result)
    except:
        # TODO: handle retries in case of duckduckgo_search.exceptions.RatelimitException
        pass

    if ctx.verbose:
        plugins.on_print("Web Search Results:", Fore.WHITE + Style.DIM)
        plugins.on_print(urls, Fore.WHITE + Style.DIM)

    if len(urls) == 0:
        # If no new URLs found, return cache results if available
        if cache_check_results:
            if ctx.verbose:
                plugins.on_print("No new search results found. Returning cache results.", Fore.YELLOW + Style.DIM)
            if return_intermediate:
                intermediate_data = {
                    'cache_hit': True,
                    'fallback': True,
                    'num_results': cache_metadata.get('num_results', 0),
                    'search_results': [],
                    'urls': [],
                    'articles': [],
                    'vector_db_results': cache_check_results
                }
                return cache_check_results, intermediate_data
            return cache_check_results

        if return_intermediate:
            return "No search results found.", {}
        return "No search results found."

    web_crawler = SimpleWebCrawler(urls, llm_enabled=True, system_prompt="You are a web crawler assistant.", selected_model=ctx.current_model, temperature=0.1, verbose=ctx.verbose, plugins=plugins.plugin_manager.plugins, num_ctx=num_ctx)
    # web_crawler.crawl(task=f"Highlight key-points about '{query}', using information provided. Format output as a list of bullet points.")
    web_crawler.crawl(ctx=ctx)
    articles = web_crawler.get_articles()

    # Save articles to temporary files, before indexing them in the vector database
    # Create a random folder to store the temporary files, in the OS temp directory
    temp_folder = tempfile.mkdtemp()
    additional_metadata = {}
    for i, article in enumerate(articles):
        # Compute the file path for the article, using the url as the filename, removing invalid characters
        temp_file_name = re.sub(r'[<>:"/\\|?*]', '', article['url'])

        temp_file_path = os.path.join(temp_folder, f"{temp_file_name}_{i}.txt")
        with open(temp_file_path, 'w', encoding='utf-8') as f:
            f.write(article['text'])
            additional_metadata[temp_file_path] = {'url': article['url']}

    if web_embedding_model is None or web_embedding_model == "":
        web_embedding_model = ctx.embeddings_model

    # Index the articles in the vector database
    document_indexer = DocumentIndexer(temp_folder, web_cache_collection, ctx.chroma_client, web_embedding_model, verbose=ctx.verbose, summary_model=ctx.current_model)
    document_indexer.index_documents(no_chunking_confirmation=True, additional_metadata=additional_metadata,  ctx=ctx)

    # Remove the temporary folder and its contents
    for file in os.listdir(temp_folder):
        file_path = os.path.join(temp_folder, file)
        os.remove(file_path)
    os.rmdir(temp_folder)

    # Search the vector database for the query with adaptive filtering
    results, result_metadata = query_vector_database(
        query,
        collection_name=web_cache_collection,
        n_results=n_results * 2,  # Request more results for better selection
        query_embeddings_model=web_embedding_model,
        use_adaptive_filtering=True,
        return_metadata=True,
        ctx=ctx
    )

    # If no results are found, refined search
    if not results:
        new_query = ask_ollama(
            "",
            f"No relevant information found. Please provide a refined search query: {query}",
            ctx.current_model,
            temperature=0.7,
            no_bot_prompt=True,
            stream_active=False,
            num_ctx=num_ctx,
            ctx=ctx
        )
        if new_query:
            if ctx.verbose:
                plugins.on_print(f"Refined search query: {new_query}", Fore.WHITE + Style.DIM)
            return web_search(
                new_query,
                n_results,
                region,
                web_embedding_model,
                num_ctx,
                return_intermediate,
                ctx=ctx
            )

    # Return intermediate results if requested
    if return_intermediate:
        intermediate_data = {
            'cache_hit': False,
            'search_results': search_results_list,
            'urls': urls,
            'articles': articles,
            'vector_db_results': results,
            'num_results': result_metadata.get('num_results', 0) if result_metadata else 0
        }
        return results, intermediate_data

    return results


# TODO: Understand difference with web_search and rename appropriately
def web_search2( *,  show_intermediate:bool, num_ctx, ctx:Context):

    if ctx.verbose:
        plugins.on_print(f"Performing web search for: {ctx.web_search}", Fore.WHITE + Style.DIM)
        plugins.on_print(f"Number of results: {ctx.web_search_results}", Fore.WHITE + Style.DIM)
        plugins.on_print(f"Region: {ctx.web_search_region}", Fore.WHITE + Style.DIM)
        plugins.on_print(f"Show intermediate results: {show_intermediate}", Fore.WHITE + Style.DIM)

    # Ensure ChromaDB is loaded for web search caching
    #load_chroma_client(ctx=ctx)

    if not ctx.chroma_client:
        plugins.on_print("Web search requires ChromaDB to be running. Please start ChromaDB server or configure a persistent database path.", Fore.RED)
        sys.exit(1)

    # Perform the web search
    if show_intermediate:
        plugins.on_print("\n" + "="*80, Fore.MAGENTA)
        plugins.on_print("SEARCHING THE WEB, QUERY: " + ctx.web_search, Fore.MAGENTA + Style.BRIGHT)
        plugins.on_print("="*80, Fore.MAGENTA)

        web_search_response, intermediate_data = web_search(
            ctx.web_search,
            n_results=ctx.web_search_results,
            region=ctx.web_search_region,
            web_embedding_model=ctx.embeddings_model,
            num_ctx=num_ctx,
            return_intermediate=True,
            ctx=ctx
        )

        # Display intermediate results
        if intermediate_data:
            plugins.on_print("\n" + "="*80, Fore.MAGENTA)
            plugins.on_print("INTERMEDIATE RESULTS", Fore.MAGENTA + Style.BRIGHT)
            plugins.on_print("="*80, Fore.MAGENTA)

            # Show search results
            if 'search_results' in intermediate_data and intermediate_data['search_results']:
                plugins.on_print("\n" + "-"*80, Fore.MAGENTA)
                plugins.on_print("1. SEARCH RESULTS FROM DUCKDUCKGO", Fore.MAGENTA + Style.BRIGHT)
                plugins.on_print("-"*80, Fore.MAGENTA)
                for i, result in enumerate(intermediate_data['search_results'], 1):
                    plugins.on_print(f"\n{i}. {result.get('title', 'N/A')}", Fore.CYAN + Style.BRIGHT)
                    plugins.on_print(f"   URL: {result.get('href', 'N/A')}", Fore.CYAN)
                    plugins.on_print(f"   Snippet: {result.get('body', 'N/A')}", Fore.WHITE)

            # Show URLs being crawled
            if 'urls' in intermediate_data and intermediate_data['urls']:
                plugins.on_print("\n" + "-"*80, Fore.MAGENTA)
                plugins.on_print("2. URLS BEING CRAWLED", Fore.MAGENTA + Style.BRIGHT)
                plugins.on_print("-"*80, Fore.MAGENTA)
                for i, url in enumerate(intermediate_data['urls'], 1):
                    plugins.on_print(f"   {i}. {url}", Fore.CYAN)

            # Show crawled articles
            if 'articles' in intermediate_data and intermediate_data['articles']:
                plugins.on_print("\n" + "-"*80, Fore.MAGENTA)
                plugins.on_print("3. CRAWLED CONTENT", Fore.MAGENTA + Style.BRIGHT)
                plugins.on_print("-"*80, Fore.MAGENTA)
                for i, article in enumerate(intermediate_data['articles'], 1):
                    plugins.on_print(f"\n{i}. URL: {article.get('url', 'N/A')}", Fore.CYAN + Style.BRIGHT)
                    content = article.get('text', '')
                    # Show first 500 characters of each article
                    preview = content[:500] + "..." if len(content) > 500 else content
                    plugins.on_print(f"   Content preview: {preview}", Fore.WHITE)
                    plugins.on_print(f"   Total length: {len(content)} characters", Fore.YELLOW)

            # Show vector DB results
            if 'vector_db_results' in intermediate_data:
                plugins.on_print("\n" + "-"*80, Fore.MAGENTA)
                plugins.on_print("4. VECTOR DATABASE RETRIEVAL RESULTS", Fore.MAGENTA + Style.BRIGHT)
                plugins.on_print("-"*80, Fore.MAGENTA)
                plugins.on_print(intermediate_data['vector_db_results'], Fore.WHITE)

            plugins.on_print("\n" + "="*80, Fore.MAGENTA)
    else:
        web_search_response = web_search(
            ctx.web_search,
            n_results=ctx.web_search_results,
            region=ctx.web_search_region,
            web_embedding_model=ctx.embeddings_model,
            num_ctx=num_ctx,
            return_intermediate=False,
            ctx=ctx
        )

    if web_search_response:
        # Build the prompt with web search context
        web_search_prompt = f"Context: {web_search_response}\n\n"
        web_search_prompt += f"Question: {ctx.web_search}\n"
        web_search_prompt += "Answer the question as truthfully as possible using the provided web search results, and if the answer is not contained within the text above, say 'I don't know'.\n"
        web_search_prompt += "Cite some useful links from the search results to support your answer."

        if ctx.verbose:
            plugins.on_print("\n" + "="*80, Fore.CYAN)
            plugins.on_print("WEB SEARCH CONTEXT", Fore.CYAN + Style.BRIGHT)
            plugins.on_print("="*80, Fore.CYAN)
            plugins.on_print(web_search_response, Fore.WHITE + Style.DIM)
            plugins.on_print("="*80, Fore.CYAN)

        # Use the current model (already initialized)
        if ctx.verbose:
            plugins.on_print(f"Using model: {ctx.current_model}", Fore.WHITE + Style.DIM)

        # Get answer from the model
        plugins.on_print("\n" + "="*80, Fore.GREEN)
        plugins.on_print("ANSWER", Fore.GREEN + Style.BRIGHT)
        plugins.on_print("="*80, Fore.GREEN)

        answer = ask_ollama(
            "",
            web_search_prompt,
            ctx.current_model,
            temperature=ctx.temperature,
            no_bot_prompt=True,
            stream_active=ctx.stream,
            num_ctx=num_ctx,
            ctx=ctx
        )

        if answer:
            if not ctx.stream:
                plugins.on_print(answer)
            plugins.on_print("\n" + "="*80, Fore.GREEN)

            # Save to output file if specified
            if ctx.output:
                with open(ctx.output, 'w', encoding='utf-8') as f:
                    f.write(f"Query: {ctx.web_search}\n\n")
                    f.write(f"Context:\n{web_search_response}\n\n")
                    f.write(f"Answer:\n{answer}\n")
                plugins.on_print(f"\nResults saved to: {ctx.output}", Fore.GREEN)
        else:
            plugins.on_print("No answer generated.", Fore.YELLOW)
    else:
        plugins.on_print("No web search results found.", Fore.YELLOW)
