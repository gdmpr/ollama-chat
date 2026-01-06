import math

from colorama import Fore, Style

from ollama_chat.core import utils
from ollama_chat.core.ollama import ask_ollama
from ollama_chat.core.context import Context

def summarize_chunk(
    text_chunk,
    model,
    max_summary_words,
    previous_summary=None,
    num_ctx=None,
    language='English',
    *,
    ctx:Context
):
    """
    Summarizes a single chunk of text using the provided LLM.

    Args:
        text_chunk (str): The piece of text to summarize.
        model (str): The name of the LLM model to use for summarization.
        max_summary_words (int): The approximate desired word count for the chunk's summary.
        previous_summary (str, optional): The previous summary to include in the prompt. Defaults to None.
        num_ctx (int, optional): The number of context tokens to use for the LLM. Defaults to None.

    Returns:
        str: The summarized text.
    """
    # Instruct the model to produce the summary in the requested language.
    system_prompt = (
        "You are an expert at summarizing text. Your task is to provide a concise summary of the given content, "
        "maintaining context from previous parts. Always produce the summary in the requested language."
    )

    # Add context from the previous summary to the prompt if it exists.
    if previous_summary:
        user_prompt = (
            f"The summary of the previous text chunk (written in {language}) is: \"{previous_summary}\"\n\n"
            f"Based on that context, please summarize the following new text chunk in approximately {max_summary_words} words. "
            f"Make sure the summary is written in {language} and do not include extra commentary:\n\n"
            f"---\n\n{text_chunk}"
        )
    else:
        user_prompt = (
            f"Please summarize the following text in approximately {max_summary_words} words. "
            f"Make sure the summary is written in {language} and do not include extra commentary:\n\n---\n\n{text_chunk}"
        )

    # This function call should interact with your local LLM. Enforce language in the call.
    summary = ask_ollama(
        system_prompt,
        user_prompt,
        model,
        no_bot_prompt=True,
        stream_active=False,
        num_ctx=num_ctx,
        ctx=ctx
    )
    # If the LLM returns an empty or None response, return an empty string to avoid breaking callers
    return summary or ""

def summarize_text_file(file_path,  model=None, chunk_size=400, overlap=50, max_final_words=500, num_ctx=None, language='English',  *, ctx:Context):
    """
    Summarizes a long text by breaking it into chunks, summarizing them,
    and then iteratively summarizing the summaries until the final text is
    under a specified word count.

    Args:
        file_path (str): The complete text file to summarize.
        model (str): The model name to be used for the summarization (e.g., 'llama3').
        chunk_size (int): The number of words in each text chunk.
        overlap (int): The number of words to overlap between consecutive chunks to maintain context.
        max_final_words (int): The maximum number of words desired for the final summary.
        num_ctx (int, optional): The number of context tokens to use for the LLM. Defaults to None.
        language (str): Language in which intermediate and final summaries should be produced. Defaults to 'English'.
        verbose (bool): If True, print detailed information about the summarization process.

    Returns:
        str: The final, concise summary.
    """

    if not model:
        model = ctx.current_model

    # Read the full text from the file
    with open(file_path, 'r', encoding='utf-8') as f:
        full_text = f.read()

    words = full_text.split()
    current_text_words = words

    while len(current_text_words) > max_final_words:
        if ctx.verbose:
            utils.on_print(f"\n>>> Iteration: Processing {len(current_text_words)} words...", Fore.WHITE + Style.DIM)

        # Determine the size of the summary for each chunk in this iteration
        # We want the total summary to be smaller than the current text length
        num_chunks_approx = math.ceil(len(current_text_words) / (chunk_size - overlap))
        # Aim for summaries that are collectively about half the size of the current text
        # but don't make individual summaries smaller than a reasonable minimum.
        per_chunk_summary_words = max(25, (len(current_text_words) // 2) // num_chunks_approx)


        chunks = []
        start = 0
        while start < len(current_text_words):
            end = start + chunk_size
            chunks.append(" ".join(current_text_words[start:end]))
            start += chunk_size - overlap
            if start >= len(current_text_words):
                break

        summaries = []
        previous_summary = None # Keep track of the last summary
        for i, chunk in enumerate(chunks):
            if ctx.verbose:
                utils.on_print(f"Processing chunk {i+1}/{len(chunks)} with {len(chunk.split())} words", Fore.WHITE + Style.DIM)
            summary = summarize_chunk(
                chunk,
                model,
                per_chunk_summary_words,
                previous_summary=previous_summary,
                num_ctx=num_ctx,
                language=language,
                ctx=ctx
            )
            summaries.append(summary)
            previous_summary = summary # Update the previous summary for the next iteration

        # The new text to be summarized is the concatenation of the summaries from this round
        combined_summaries = " ".join(summaries)
        current_text_words = combined_summaries.split()

        if ctx.verbose:
            utils.on_print(f"<<< Iteration Complete: {len(summaries)} summaries created, new word count is {len(current_text_words)}", Fore.WHITE + Style.DIM)
            utils.on_print(f"Current text after summarization: {combined_summaries[:100]}...", Fore.WHITE + Style.DIM)

    final_summary = " ".join(current_text_words)
    return final_summary
