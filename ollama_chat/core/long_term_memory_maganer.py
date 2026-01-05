from appdirs import AppDirs
import os
import json
from ollama_chat.core import extract_json
from ollama_chat.core import ask_ollama
from ollama_chat.core import on_print

class LongTermMemoryManager:
    def __init__(self, selected_model, verbose=False, num_ctx=None, memory_file="long_term_memory.json"):
        # Initialize app directories using appdirs
        dirs = AppDirs(APP_NAME, APP_AUTHOR, version=APP_VERSION)

        # The user-specific data directory (varies depending on OS)
        prefs_dir = dirs.user_data_dir

        # Ensure the directory exists
        os.makedirs(prefs_dir, exist_ok=True)

        # Path to the preferences file
        self.memory_file = os.path.join(prefs_dir, memory_file)
        self.memory = self._load_memory()
        self.selected_model = selected_model
        self.verbose = verbose
        self.num_ctx = num_ctx

    def _load_memory(self):
        """Loads the long-term memory from the JSON file."""
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'r') as file:
                return json.load(file)
        else:
            return {"users": {}}

    def _save_memory(self):
        """Saves the current memory state to the JSON file."""
        with open(self.memory_file, 'w') as file:
            json.dump(self.memory, file, indent=4)

    def _update_user_memory(self, user_id, new_info):
        """Updates or adds key-value pairs in the user's long-term memory."""
        if user_id not in self.memory["users"]:
            self.memory["users"][user_id] = {}

        if isinstance(new_info, dict):
            # Update the user's memory with new info
            for key, value in new_info.items():
                self.memory["users"][user_id][key] = value

            # Save the updated memory back to the JSON file
            self._save_memory()

    def process_conversation(self, user_id, conversation):
        """
        Processes a conversation and uses GPT to:
        - Extract relevant key-value pairs for long-term memory.
        - Check for contradictions in the memory.
        """

        # Convert conversation list of objects to a list of dict
        conversation = [json.loads(json.dumps(obj, default=lambda o: vars(o))) for obj in conversation]

        filtered_conversation = [entry for entry in conversation if entry['role'] not in ['system', 'tool', 'function']]

        # Convert conversation array into a string for GPT prompt
        conversation_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in filtered_conversation if 'role' in msg and 'content' in msg])

        # Step 1: Extract key-value information
        system_prompt_extract = self._get_extraction_prompt()
        extracted_info = extract_json(ask_ollama(system_prompt_extract, conversation_str, self.selected_model, temperature=0.1, no_bot_prompt=True, stream_active=False, num_ctx=self.num_ctx))

        if self.verbose:
            on_print(f"Extracted information: {extracted_info}", Fore.WHITE + Style.DIM)

        # Step 2: Check for contradictions with existing memory
        existing_memory = self.memory["users"].get(user_id, {})
        system_prompt_conflict = self._get_conflict_check_prompt(existing_memory, conversation_str)
        conflicting_info = extract_json(ask_ollama(system_prompt_conflict, conversation_str, self.selected_model, temperature=0.1, no_bot_prompt=True, stream_active=False, num_ctx=self.num_ctx))

        # Remove conflicting info from memory if flagged by GPT
        if conflicting_info:
            self._remove_conflicting_info(user_id, conflicting_info)

        # Update user's long-term memory with the newly extracted info
        self._update_user_memory(user_id, extracted_info)

    def _get_extraction_prompt(self):
        """
        Returns the system prompt for extracting key-value information from the conversation.
        """
        return """
        You are analyzing a conversation between a user and an assistant. Your task is to extract key pieces of information 
        about the user that could be useful for long-term memory.
        
        The information should be structured as key-value pairs, where the **keys** represent different aspects of the user's life, such as:
        - Relationships (e.g., 'sister', 'friends', 'spouse')
        - Preferences (e.g., 'favorite color', 'preferred music', 'favorite food')
        - Hobbies (e.g., 'hobbies', 'sports')
        - Jobs (e.g., 'job', 'role', 'employer')
        - Interests (e.g., 'interests', 'books', 'movies')

        Focus on extracting personal, long-term information that is explicitly or implicitly mentioned in the conversation. 
        Ignore temporary or context-specific information (e.g., emotions, recent events).

        The format should be a JSON object with key-value pairs. For example:
        {{
            "sister": "Rebecca",
            "friends": ["John", "Alice"],
            "hobbies": ["playing guitar"]
        }}

        If the conversation does not provide relevant information for any of these categories, do not generate that key. Be concise and ensure the values are clear and accurate.
        """

    def _get_conflict_check_prompt(self, existing_memory, conversation_str):
        """
        Returns the system prompt for checking contradictions between existing memory and the new conversation.
        """
        return f"""
        You are analyzing a conversation between a user and an assistant to determine if any part of the user's existing 
        long-term memory is incorrect or outdated.

        The user has the following existing memory, structured as key-value pairs:
        {json.dumps(existing_memory, indent=4)}

        Compare this existing memory with the following conversation:
        {conversation_str}

        Your task is to:
        1. Identify if any key-value pairs from the existing memory are **contradicted** by the information in the conversation.
        2. For each key-value pair that is contradicted, list the **key** that should be removed or updated based on the new conversation.

        Output the list of conflicting keys as a JSON array. For example:
        ```json
        ["sister", "favorite_color"]
        ```

        If no conflicts are found, return an empty JSON array:
        ```json
        []
        ```
        """

    def _remove_conflicting_info(self, user_id, conflicting_keys):
        """Removes conflicting keys from the user's memory."""
        if isinstance(conflicting_keys, dict):
            if user_id in self.memory["users"]:
                for key in conflicting_keys:
                    if key in self.memory["users"][user_id]:
                        del self.memory["users"][user_id][key]
                self._save_memory()
