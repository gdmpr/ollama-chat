import json
import os
from datetime import datetime

import ollama
from colorama import Fore, Style

from ollama_chat.core.context import Context
from ollama_chat.core.long_term_memory_manager import LongTermMemoryManager
from ollama_chat.core.utils import on_print
from ollama_chat.core.ollama import ask_ollama

class MemoryManager:
    def __init__(self, collection_name, chroma_client, selected_model, embedding_model_name, verbose=False, num_ctx=None, long_term_memory_file="long_term_memory.json"):
        """
        Initialize the MemoryManager with a specific ChromaDB collection.

        :param collection_name: The name of the ChromaDB collection used to store memory.
        :param chroma_client: The ChromaDB client instance.
        :param selected_model: The model used in ask_ollama for generating responses and embeddings.
        :param embedding_model_name: The name of the embedding model for generating embeddings.
        """
        self.collection_name = collection_name
        self.client = chroma_client
        self.selected_model = selected_model
        self.embedding_model_name = embedding_model_name
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        self.verbose = verbose
        self.num_ctx = num_ctx
        self.long_term_memory_manager = LongTermMemoryManager(selected_model, verbose, num_ctx, memory_file=long_term_memory_file)

    def preprocess_conversation(self, conversation,  ctx:Context):
        """
        Preprocess the conversation to filter out tool or function role entries, and then summarize key points.

        :param conversation: The conversation array (list of role/content dictionaries).
        :return: Summarized key points for the conversation.
        """
        # Convert conversation list of objects to a list of dict
        conversation = [json.loads(json.dumps(obj, default=lambda o: vars(o))) for obj in conversation]

        # Filter out tool/function roles
        filtered_conversation = [entry for entry in conversation if entry['role'] not in ['system', 'tool', 'function']]

        if len(filtered_conversation) == 0:
            return ""

        # Concatenate the filtered conversation into a single input (make sure entries contain 'role' and 'content' keys)
        user_input = "\n".join([f"{entry['role']}: {entry['content']}" for entry in filtered_conversation if 'role' in entry and 'content' in entry])

        # Define an elaborated system prompt for the LLM to generate a high-quality summary
        system_prompt = """
        You are a memory assistant tasked with summarizing conversations for future reference.
        Your goal is to identify the key points, user intents, important questions, decisions made, and any personal information shared by the user.
        Focus on gathering and summarizing:
        - Core ideas and user questions
        - Notable responses from the assistant or system
        - Personal details shared by the user (e.g., family, life, location, occupation, interests)
        - Any decisions, action points, or follow-up tasks

        When capturing personal details, organize them clearly for future context (e.g., 'User mentioned living in X city' or 'User has a family with two children').
        Avoid excessive technical details, irrelevant tool-related content, or repetition.

        Important: ensure the summary is generated in conversation language.

        Conversation:
        """

        # Use the ask_ollama function to summarize key points
        summary = ask_ollama(
            system_prompt,
            user_input,
            self.selected_model,
            temperature=0.1,
            no_bot_prompt=True,
            stream_active=False,
            num_ctx=self.num_ctx,
            ctx = ctx
        )

        return summary

    def generate_embedding(self, text):
        """
        Generate embeddings for a given text using the specified embedding model.

        :param text: The input text to generate embeddings for.
        :return: The embedding vector.
        """
        embedding = None
        if self.embedding_model_name:
            ollama_options = {}
            if self.num_ctx:
                ollama_options["num_ctx"] = self.num_ctx

            response = ollama.embeddings(
                prompt=text,
                model=self.embedding_model_name,
                options=ollama_options
            )
            embedding = response["embedding"]
        return embedding

    def add_memory(self, conversation, metadata=None, *, ctx:Context):
        """
        Preprocess and store a conversation in memory by summarizing it and storing the summary.

        :param conversation: The conversation array (list of role/content dictionaries).
        :param metadata: Additional metadata to store with the memory (e.g., timestamp, user info).
        """
        conversation_id = datetime.now().strftime("%Y%m%d%H%M%S")

        # Preprocess the conversation to summarize the key points
        summarized_conversation = self.preprocess_conversation(conversation,  ctx=ctx)

        if len(summarized_conversation) == 0:
            if self.verbose:
                on_print("Empty conversation. No memory added.", Fore.WHITE + Style.DIM)
            return False

        # Create metadata if none is provided
        if metadata is None:
            # Format the metadata with a timestamp in a human-readable format (July 1, 2022, 12:00 PM)
            timestamp = datetime.now().strftime("%A, %B %d, %Y, %I:%M %p")
            metadata = {'timestamp': timestamp}

        # Generate an embedding for the summarized conversation
        embedding = self.generate_embedding(summarized_conversation)

        # Store the summarized conversation in the ChromaDB collection
        self.collection.upsert(
            documents=[summarized_conversation],
            metadatas=[metadata],
            ids=[conversation_id],
            embeddings=[embedding]
        )

        if self.verbose:
            on_print(f"Memory for conversation {conversation_id} added. Summary: {summarized_conversation}", Fore.WHITE + Style.DIM)

        user_id = "anonymous"
        try:
            user_id = os.getlogin()
        except:
            user_id = os.environ['USER']

        self.long_term_memory_manager.process_conversation(user_id, conversation, ctx=ctx)

        if self.verbose:
            on_print("Long-term memory updated.", Fore.WHITE + Style.DIM)

        return True

    def retrieve_relevant_memory(self, query_text, top_k=3, answer_distance_threshold=200):
        """
        Retrieve the most relevant memories based on the given query.

        :param query_text: The query or question for which relevant memories should be retrieved.
        :param top_k: Number of relevant memories to retrieve.
        :return: A list of the top-k most relevant memories.
        """
        if self.verbose:
            on_print(f"Retrieving relevant memories for query: {query_text}", Fore.WHITE + Style.DIM)

        # Generate an embedding for the query
        query_embedding = self.generate_embedding(query_text)

        if query_embedding is None:
            return [], []

        # Query the memory collection for relevant memories
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        documents = results["documents"][0]
        distances = results["distances"][0]
        metadatas = results["metadatas"][0]

        # Filter the results based on the answer distance threshold
        filtered_results = {
            'documents': [],
            'metadatas': []
        }
        for metadata, answer_distance, document in zip(metadatas, distances, documents):
            if answer_distance_threshold > 0 and answer_distance > answer_distance_threshold:
                if self.verbose:
                    on_print(f"Answer distance: {answer_distance} > {answer_distance_threshold}. Skipping memory.", Fore.WHITE + Style.DIM)
                continue

            if self.verbose:
                on_print(f"Answer distance: {answer_distance}", Fore.WHITE + Style.DIM)
                on_print(f"Memory: {document}", Fore.WHITE + Style.DIM)
                on_print(f"Metadata: {metadata}", Fore.WHITE + Style.DIM)

            filtered_results['documents'].append(document)
            filtered_results['metadatas'].append(metadata)

        return filtered_results['documents'], filtered_results['metadatas']

    def handle_user_query(self, conversation, query=None):
        """
        Handle a user query by updating the 'system' part of the conversation with relevant memories in XML markup.

        :param conversation: The current conversation array (list of role/content dictionaries).
        :return: Updated conversation with a modified system prompt containing memory placeholders in XML format.
        """

        # Find the latest user input from the conversation (role 'user')
        user_input = query
        for entry in reversed(conversation):
            if entry['role'] == 'user':
                user_input = entry['content']
                break

        if not user_input or len(user_input.strip()) == 0:
            return

        # Retrieve relevant memories based on the current user query
        relevant_memories, memory_metadata = self.retrieve_relevant_memory(user_input)

        # Find the existing 'system' prompt in the conversation
        system_prompt_entry = None
        for entry in conversation:
            if entry['role'] == 'system':
                system_prompt_entry = entry
                break

        if system_prompt_entry:
            # Keep the initial system prompt unchanged and remove the old memory section
            original_system_prompt = system_prompt_entry['content']

            # Define the memory section using XML-style tags
            memory_start_tag = "<short-term-memories>"
            memory_end_tag = "</short-term-memories>"

            # Remove any previous memory section if it exists
            if memory_start_tag in original_system_prompt:
                original_system_prompt = original_system_prompt.split(memory_start_tag)[0].strip()

            # Format the new memory content in XML markup, including metadata serialization
            memory_text = ""
            for i, memory in enumerate(relevant_memories):
                metadata_str = json.dumps(memory_metadata[i], indent=2) if i < len(memory_metadata) else "{}"
                memory_text += f"Memory {i+1}:\n{memory}\nMetadata: {metadata_str}\n\n"

            if memory_text:
                memory_section = f"{memory_start_tag}\nIn the past we talked about...\n{memory_text.strip()}\n{memory_end_tag}"

                # Update the system prompt with the new memory section
                system_prompt_entry['content'] = f"{original_system_prompt}\n\n{memory_section}"

                if self.verbose:
                    on_print(f"System prompt updated with relevant memories:\n{system_prompt_entry['content']}", Fore.WHITE + Style.DIM)
            else:
                if self.verbose:
                    on_print("No relevant memories found for the user query.", Fore.WHITE + Style.DIM)
                system_prompt_entry['content'] = original_system_prompt
        else:
            # If no system prompt exists, raise an exception (or create one, depending on desired behavior)
            raise ValueError("No system prompt found in the conversation")
