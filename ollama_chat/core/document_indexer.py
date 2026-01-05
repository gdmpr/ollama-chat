import os
import re
from datetime import datetime
from urllib.parse import urljoin
from colorama import Fore, Style
import ollama

from tqdm import tqdm

from ollama_chat.core import on_print
from ollama_chat.core import ask_ollama
from ollama_chat.core import on_user_input
from ollama_chat.core import MarkdownSplitter
from ollama_chat.core import extract_text_from_html

class DocumentIndexer:
    def __init__(self, root_folder, collection_name, chroma_client, embeddings_model, verbose=False, summary_model=None, full_doc_store=None):
        self.root_folder = root_folder
        self.collection_name = collection_name
        self.client = chroma_client
        self.model = embeddings_model  # For embeddings only
        self.summary_model = summary_model
        self.full_doc_store = full_doc_store  # Optional SQLite store for full documents
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        self.verbose = verbose

        if verbose:
            on_print(f"DocumentIndexer initialized with embedding model: {self.model}", Fore.WHITE + Style.DIM)
            if self.summary_model:
                on_print(f"Using summary model: {self.summary_model}", Fore.WHITE + Style.DIM)
            on_print(f"Using collection: {self.collection.name}", Fore.WHITE + Style.DIM)
            on_print(f"Verbose mode is {'on' if self.verbose else 'off'}", Fore.WHITE + Style.DIM)
            on_print(f"Using embeddings model: {self.model}", Fore.WHITE + Style.DIM)
            if self.full_doc_store:
                on_print(f"Full document store enabled at: {self.full_doc_store.db_path}", Fore.WHITE + Style.DIM)

    def _prepare_text_for_embedding(self, text, num_ctx=None):
        """
        Prepare text to send to the embedding model by truncating it to the model/context limit.

        If num_ctx is provided we assume it is the model token/context window. If not provided,
        we fall back to a default of 2048 tokens. When the model max tokens is unknown we use
        a conservative heuristic of 1 token = 4 characters.

        Returns the possibly-truncated text to send to the embedding API. The original text
        must remain untouched for storage in ChromaDB.
        """
        try:
            if num_ctx and isinstance(num_ctx, int) and num_ctx > 0:
                max_tokens = num_ctx
            else:
                # Default context window if not specified
                max_tokens = 2048

            # Heuristic: 1 token ~= 4 characters
            max_chars = max_tokens * 4

            if len(text) > max_chars:
                if self.verbose:
                    on_print(f"Truncating text for embedding: original {len(text)} chars > {max_chars} chars (tokens={max_tokens})", Fore.YELLOW)
                return text[:max_chars]
            return text
        except Exception as e:
            # In case of unexpected errors, fall back to original text (do not modify stored docs)
            if self.verbose:
                on_print(f"Error while preparing text for embedding: {e}. Using original text.", Fore.YELLOW)
            return text

    def get_text_files(self):
        """
        Recursively find all .txt, .md, .tex files in the root folder.
        Also include HTML files without extensions if they start with <!DOCTYPE html> or <html.
        Ignore empty lines at the beginning of the file and check only the first non-empty line.
        """
        text_files = []
        for root, dirs, files in os.walk(self.root_folder):
            for file in files:
                # Check for files with extension
                if file.endswith(".txt") or file.endswith(".md") or file.endswith(".tex"):
                    text_files.append(os.path.join(root, file))
                else:
                    # Check for HTML files without extensions
                    file_path = os.path.join(root, file)
                    if is_html(file_path):
                        text_files.append(file_path)
        return text_files

    def read_file(self, file_path):
        """
        Read the content of a file.
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            try:
                return file.read()
            except:
                return None

    def extract_text_between_strings(self, content, start_string, end_string):
        """
        Extract text between two specified strings.
        
        :param content: The full text content.
        :param start_string: The string marking the start of extraction.
        :param end_string: The string marking the end of extraction.
        :return: The extracted text, or the full content if strings are not found.
        """
        if not start_string or not end_string:
            return content
            
        start_index = content.find(start_string)
        if start_index == -1:
            if self.verbose:
                on_print(f"Start string '{start_string}' not found, using full content", Fore.YELLOW)
            return content
            
        # Move past the start string
        start_index += len(start_string)
        
        end_index = content.find(end_string, start_index)
        if end_index == -1:
            if self.verbose:
                on_print(f"End string '{end_string}' not found after start string, using content from start string to end", Fore.YELLOW)
            return content[start_index:]
            
        extracted_text = content[start_index:end_index]
        
        if self.verbose:
            on_print(f"Extracted {len(extracted_text)} characters between '{start_string}' and '{end_string}'", Fore.WHITE + Style.DIM)
            
        return extracted_text

    def index_documents(self, allow_chunks=True, no_chunking_confirmation=False, split_paragraphs=False, additional_metadata=None, num_ctx=None, skip_existing=True, extract_start=None, extract_end=None, add_summary=True):
        """
        Index all text files in the root folder.
        
        :param allow_chunks: Whether to chunk large documents.
        :param no_chunking_confirmation: Skip confirmation for chunking and extraction prompts.
        :param split_paragraphs: Whether to split markdown content into paragraphs.
        :param additional_metadata: Optional dictionary to pass additional metadata by file name.
        :param skip_existing: Whether to skip indexing if a document/chunk with the same ID already exists.
        :param extract_start: Optional string marking the start of the text to extract for embedding computation.
        :param extract_end: Optional string marking the end of the text to extract for embedding computation.
        :param add_summary: Whether to generate and prepend a summary to each chunk (default: True).
        """
        # Ask the user to confirm if they want to allow chunking of large documents
        if allow_chunks and not no_chunking_confirmation:
            on_print("Large documents will be chunked into smaller pieces for indexing.")
            allow_chunks = on_user_input("Do you want to continue with chunking (if you answer 'no', large documents will be indexed as a whole)? [y/n]: ").lower() in ['y', 'yes']

        # Ask the user for extraction strings if not provided
        # Skip asking if no_chunking_confirmation is True (automated indexing)
        if extract_start is None and extract_end is None and not no_chunking_confirmation:
            on_print("\nOptional: You can extract only a specific part of each document for embedding computation.")
            on_print("This allows you to focus on relevant sections while still storing the full document.")
            use_extraction = on_user_input("Do you want to extract specific text sections for embedding? [y/n]: ").lower() in ['y', 'yes']
            
            if use_extraction:
                extract_start = on_user_input("Enter the start string (text that marks the beginning of the section): ").strip()
                extract_end = on_user_input("Enter the end string (text that marks the end of the section): ").strip()
                
                if not extract_start or not extract_end:
                    on_print("Warning: Empty start or end string provided. Text extraction will be disabled.", Fore.YELLOW)
                    extract_start = None
                    extract_end = None
                else:
                    on_print(f"Text extraction enabled: extracting content between '{extract_start}' and '{extract_end}'", Fore.GREEN)

        if allow_chunks:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

        # Get the list of text files
        text_files = self.get_text_files()

        if allow_chunks:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

        progress_bar = None
        if self.verbose:
            # Progress bar for indexing
            progress_bar = tqdm(total=len(text_files), desc="Indexing files", unit="file", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}")

        for file_path in text_files:
            if progress_bar:
                progress_bar.update(1)

            try:
                document_id = os.path.splitext(os.path.basename(file_path))[0]

                # Check if skipping existing documents and if the document ID exists (for non-chunked case)
                if not allow_chunks and skip_existing:
                    existing_doc = self.collection.get(ids=[document_id])
                    if existing_doc and len(existing_doc.get('ids', [])) > 0:
                        if self.verbose:
                            on_print(f"Skipping existing document: {document_id}", Fore.WHITE + Style.DIM)
                        continue

                content = self.read_file(file_path)

                if not content:
                    on_print(f"An error occurred while reading file: {file_path}", Fore.RED)
                    continue
                
                # Add any additional metadata for the file
                # Extract file name and base file information
                file_name = os.path.basename(file_path)
                file_name_without_ext = os.path.splitext(file_name)[0]
                current_date = datetime.now().isoformat()
                
                # Create a more comprehensive metadata structure
                file_metadata = {
                    'published': current_date,
                    'docSource': os.path.dirname(file_path),
                    'docAuthor': 'Unknown',
                    'description': f"Document from {file_path}",
                    'title': file_name_without_ext,
                    'id': document_id,
                    'filePath': file_path
                }
                
                # Convert the file path to url and add it to the metadata
                file_metadata['url'] = urljoin("file://", file_path)
                
                # If windows, convert the file path to a URI
                if os.name == 'nt':
                    file_metadata['url'] = file_metadata['url'].replace("\\", "/")
                    
                    # Replace the drive letter with "file:///" prefix
                    file_metadata['url'] = file_metadata['url'].replace("file://", "file:///")
                
                if additional_metadata and file_path in additional_metadata:
                    file_metadata.update(additional_metadata[file_path])

                # Extract text for embedding if start and end strings are provided
                embedding_content = content
                if extract_start and extract_end:
                    embedding_content = self.extract_text_between_strings(content, extract_start, extract_end)
                    # Add metadata to indicate partial extraction was used
                    file_metadata['extraction_used'] = True
                    file_metadata['extract_start'] = extract_start
                    file_metadata['extract_end'] = extract_end
                    file_metadata['extracted_length'] = len(embedding_content)
                    file_metadata['original_length'] = len(content)

                if allow_chunks:
                    chunks = []
                    # Use embedding_content for chunking (which may be extracted text)
                    content_to_chunk = embedding_content
                    
                    # Split Markdown files into sections if needed
                    if is_html(file_path):
                        # Convert to Markdown before splitting
                        markdown_splitter = MarkdownSplitter(extract_text_from_html(content_to_chunk), split_paragraphs=split_paragraphs)
                        chunks = markdown_splitter.split()
                    elif is_markdown(file_path):
                        markdown_splitter = MarkdownSplitter(content_to_chunk, split_paragraphs=split_paragraphs)
                        chunks = markdown_splitter.split()
                    else:
                        chunks = text_splitter.split_text(content_to_chunk)
                    
                    # Generate document summary once if add_summary is enabled
                    document_summary = None
                    # Use summary_model for summary generation, fallback to current_model if available
                    summary_model = self.summary_model
                    if summary_model is None:
                        try:
                            summary_model = current_model
                        except NameError:
                            summary_model = None
                    if add_summary and summary_model:
                        if self.verbose:
                            on_print(f"Generating summary for document {document_id} using model: {summary_model}", Fore.WHITE + Style.DIM)
                        summary_prompt = f"""Provide a brief summary (2-5 sentences) of the following document. Focus on the main topic and key points:

{content_to_chunk[:2000]}"""  # Limit to first 2000 chars for summary generation
                        try:
                            ollama_options = {}
                            if num_ctx:
                                ollama_options["num_ctx"] = num_ctx
                            summary_response = ask_ollama(
                                "You are a helpful assistant that creates concise document summaries.",
                                summary_prompt,
                                summary_model,
                                temperature=0.3,
                                no_bot_prompt=True,
                                stream_active=False,
                                num_ctx=num_ctx
                            )
                            document_summary = f"[Document Summary: {summary_response.strip()}]\n\n"
                            if self.verbose:
                                on_print(f"Summary generated: {summary_response.strip()}", Fore.GREEN)
                        except Exception as e:
                            if self.verbose:
                                on_print(f"Failed to generate summary: {e}", Fore.YELLOW)
                            document_summary = None
                    
                    for i, chunk in enumerate(chunks):
                        chunk_id = f"{document_id}_{i}"

                        # Check if skipping existing chunks and if the chunk ID exists
                        if skip_existing:
                            existing_chunk = self.collection.get(ids=[chunk_id])
                            if existing_chunk and len(existing_chunk.get('ids', [])) > 0:
                                if self.verbose:
                                    on_print(f"Skipping existing chunk: {chunk_id}", Fore.WHITE + Style.DIM)
                                continue
                        
                        # Prepend document summary to chunk if available
                        chunk_with_summary = chunk
                        if document_summary:
                            chunk_with_summary = document_summary + chunk
                        
                        # Embed the chunk content (from extracted text) with summary prepended
                        embedding = None
                        if self.model:
                            ollama_options = {}
                            if num_ctx:
                                ollama_options["num_ctx"] = num_ctx
                                
                            if self.verbose:
                                embedding_info = "using extracted text" if extract_start and extract_end else "using full content"
                                summary_info = " with summary" if document_summary else ""
                                on_print(f"Generating embedding for chunk {chunk_id} using {self.model} ({embedding_info}{summary_info})", Fore.WHITE + Style.DIM)
                            # Prepare a potentially truncated string for the embedding call so we don't exceed
                            # the model/context window and risk freezing the Ollama server. The full chunk_with_summary
                            # remains unchanged for storage in ChromaDB.
                            embedding_prompt = self._prepare_text_for_embedding(chunk_with_summary, num_ctx=num_ctx)
                            response = ollama.embeddings(
                                prompt=embedding_prompt,
                                model=self.model,
                                options=ollama_options
                            )
                            embedding = response["embedding"]
                        
                        # Store the chunk with summary prepended
                        chunk_metadata = file_metadata.copy()
                        chunk_metadata['chunk_index'] = i
                        if document_summary:
                            chunk_metadata['has_summary'] = True
                        
                        # Upsert the chunk with summary and embedding
                        if embedding:
                            self.collection.upsert(
                                documents=[chunk_with_summary],  # Store chunk with summary
                                metadatas=[chunk_metadata],
                                ids=[chunk_id],
                                embeddings=[embedding]  # Embedding computed from chunk with summary
                            )
                        else:
                            self.collection.upsert(
                                documents=[chunk_with_summary],
                                metadatas=[chunk_metadata],
                                ids=[chunk_id]
                            )
                    
                    # Store the full original document in the SQLite database if available
                    # This allows retrieval of complete documents when chunks are found
                    if self.full_doc_store and not self.full_doc_store.document_exists(document_id):
                        if self.verbose:
                            on_print(f"Storing full document {document_id} in SQLite", Fore.WHITE + Style.DIM)
                        self.full_doc_store.store_document(document_id, content, file_path)
                else:
                    # Embed the extracted content but store the whole document
                    embedding = None
                    if self.model:
                        ollama_options = {}
                        if num_ctx:
                            ollama_options["num_ctx"] = num_ctx
                            
                        if self.verbose:
                            embedding_info = "using extracted text" if extract_start and extract_end else "using full content"
                            on_print(f"Generating embedding for document {document_id} using {self.model} ({embedding_info})", Fore.WHITE + Style.DIM)

                        # Use extracted content for embedding computation. Truncate input to embedding API if needed
                        # while keeping the full document content unchanged for storage.
                        embedding_prompt = self._prepare_text_for_embedding(embedding_content, num_ctx=num_ctx)
                        response = ollama.embeddings(
                            prompt=embedding_prompt,
                            model=self.model,
                            options=ollama_options
                        )
                        embedding = response["embedding"]

                    # Store the full document content but use embedding from extracted text
                    if embedding:
                        self.collection.upsert(
                            documents=[content],  # Store full document content
                            metadatas=[file_metadata],
                            ids=[document_id],
                            embeddings=[embedding]  # Embedding computed from extracted text
                        )
                    else:
                        self.collection.upsert(
                            documents=[content],  # Store full document content
                            metadatas=[file_metadata],
                            ids=[document_id]
                        )
            except KeyboardInterrupt:
                break
            except Exception as e: # Catch other potential errors during processing
                on_print(f"Error processing file {file_path}: {e}", Fore.RED)
                continue # Continue to the next file


def is_html(file_path):
    """
    Check if the given file is an HTML file, either by its extension or content.
    """
    # Check for .htm and .html extensions
    if file_path.endswith(".htm") or file_path.endswith(".html") or file_path.endswith(".xhtml"):
        return True
    
    # Check for HTML files without extensions
    try:
        with open(file_path, 'r') as f:
            first_line = next((line.strip() for line in f if line.strip()), None)
            return first_line and (first_line.lower().startswith('<!doctype html>') or first_line.lower().startswith('<html'))
    except Exception:
        return False
    
def is_markdown(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    # Automatically consider .md files as Markdown
    if file_path.endswith('.md'):
        return True
    
    # If the file is not .md, but is .txt, proceed with content checking
    if not file_path.endswith('.txt'):
        raise ValueError(f"The file {file_path} is neither .md nor .txt.")

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Check for common Markdown patterns
            if re.match(r'^#{1,6}\s', line):  # Heading (e.g., # Heading)
                return True
    
    # If no Markdown features are found, assume it's a regular text file
    return False
    
