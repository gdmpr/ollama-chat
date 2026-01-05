import sys
import re
import json
from colorama import Fore, Style

def extract_json(garbage_str):
    global verbose_mode

    if garbage_str is None:
        return []

    # First, try to parse the entire input as JSON directly
    result = try_parse_json(garbage_str, verbose=verbose_mode)
    if result is not None:
        return result
    
    json_str = None

    if "```json" not in garbage_str:
        # Find the first curly brace or square bracket
        start_index = garbage_str.find("[")
        if start_index == -1:
            start_index = garbage_str.find("{")

        last_index = garbage_str.rfind("]")
        if last_index == -1:
            last_index = garbage_str.rfind("}")

        if start_index != -1 and last_index != -1:
            # Extract the JSON content
            json_str = garbage_str[start_index:last_index + 1]

            # If a carriage return is found between the curly braces or square brackets, try to recompute the last index based on the newline character position
            if "\n" in json_str:
                last_index = json_str.rfind("]")
                if last_index == -1:
                    last_index = json_str.rfind("}")
                
                json_str = json_str[:last_index + 1]

    if not json_str:
        # Define a regular expression pattern to match the JSON block
        pattern = r'```json\s*(\[\s*.*?\s*\])\s*```'
        
        # Search for the pattern
        match = re.search(pattern, garbage_str, re.DOTALL)
        
        if match:
            # Extract the JSON content
            json_str = match.group(1)

    if not json_str:
        # JSON may be enclosed between <tool_call> and </tool_call>
        pattern = r'<tool_call>\s*(\[\s*.*?\s*\])\s*</tool_call>'

        # Search for the pattern
        match = re.search(pattern, garbage_str, re.DOTALL)
        
        if match:
            # Extract the JSON content
            json_str = match.group(1)
    
    if json_str:
        json_str = json_str.strip()
        lines = json_str.splitlines()
        stripped_lines = [line.strip() for line in lines if line.strip()]  # Strip blanks and ignore empty lines
        json_str = ''.join(stripped_lines)  # Join lines into a single string
        # Use a regular expression to find missing commas between adjacent }{, "}{" or "" 
        json_str = re.sub(r'"\s*"', '","', json_str)  # Add comma between adjacent quotes
        json_str = re.sub(r'"\s*{', '",{', json_str)  # Add comma between "{
        json_str = re.sub(r'}\s*"', '},"', json_str)  # Add comma between }"

        # Attempt to load the JSON to verify it's correct
        if verbose_mode:
            on_print(f"Extracted JSON: '{json_str}'", Fore.WHITE + Style.DIM)
        result = try_parse_json(json_str, verbose=verbose_mode)
        if result is not None:
            return result
        
        # If parsing fails, try to handle concatenated JSON objects
        if verbose_mode:
            on_print("[DEBUG] Initial JSON parsing failed, attempting to handle concatenated JSON objects", Fore.CYAN + Style.DIM)
        
        # Try to split and merge multiple JSON objects
        merged_result = try_merge_concatenated_json(json_str, verbose=verbose_mode)
        if merged_result is not None:
            return merged_result
        
        if verbose_mode:
            on_print("Extracted string is not a valid JSON.", Fore.RED)
    else:
        if verbose_mode:
            on_print("Extracted string is not a valid JSON.", Fore.RED)
    
    return []

def try_parse_json(json_str, verbose=False):
    """Helper function to attempt JSON parsing and return the result if successful."""
    result = None

    if not json_str or not isinstance(json_str, str):
        return result

    try:
        result = json.loads(json_str)
    except json.JSONDecodeError as e:
        if verbose:
            on_print(f"JSON parsing error: {e}", Fore.RED)
        pass

    return result

def on_print(message, style="", prompt=""):
    function_handled = False
    for plugin in plugins:
        if hasattr(plugin, "on_print") and callable(getattr(plugin, "on_print")):
            plugin_response = getattr(plugin, "on_print")(message)
            function_handled = function_handled or plugin_response

    if not function_handled:
        if style or prompt:
            print(f"{style}{prompt}{message}")
        else:
            print(message)


def try_merge_concatenated_json(json_str, verbose=False):
    """
    Handle concatenated JSON objects (e.g., {"key": "value"}{"key2": "value2"})
    by attempting to extract and merge them intelligently.
    """
    if verbose:
        on_print(f"[DEBUG] Attempting to parse concatenated JSON: {json_str[:100]}...", Fore.CYAN + Style.DIM)
    
    # Try to find and extract individual JSON objects
    json_objects = []
    i = 0
    brace_count = 0
    current_obj = ""
    
    while i < len(json_str):
        char = json_str[i]
        current_obj += char
        
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            
            # When we close a brace and count reaches 0, we have a complete object
            if brace_count == 0 and current_obj.count('{') > 0:
                try:
                    obj = json.loads(current_obj.strip())
                    json_objects.append(obj)
                    if verbose:
                        on_print(f"[DEBUG] Found JSON object: {obj}", Fore.CYAN + Style.DIM)
                    current_obj = ""
                except json.JSONDecodeError:
                    pass
        
        i += 1
    
    if not json_objects:
        if verbose:
            on_print("[DEBUG] No individual JSON objects found in concatenated string", Fore.CYAN + Style.DIM)
        return None
    
    # If we have multiple objects, merge them by taking the last (most recent) one
    # or merge them into a single dict if they're all dicts
    if len(json_objects) > 1:
        if verbose:
            on_print(f"[DEBUG] Found {len(json_objects)} JSON objects, merging...", Fore.CYAN + Style.DIM)
        
        # If all are dicts, merge them
        if all(isinstance(obj, dict) for obj in json_objects):
            merged = {}
            for obj in json_objects:
                merged.update(obj)
            if verbose:
                on_print(f"[DEBUG] Merged objects: {merged}", Fore.CYAN + Style.DIM)
            return merged
        else:
            # If not all dicts, return the last one (most recent)
            if verbose:
                on_print(f"[DEBUG] Not all objects are dicts, returning last one: {json_objects[-1]}", Fore.CYAN + Style.DIM)
            return json_objects[-1]
    elif len(json_objects) == 1:
        if verbose:
            on_print(f"[DEBUG] Single JSON object found: {json_objects[0]}", Fore.CYAN + Style.DIM)
        return json_objects[0]
    
    return None


def print_spinning_wheel(print_char_index):
    # use turning block character as spinner
    spinner =  ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    on_stdout_write(spinner[print_char_index % len(spinner)], Style.RESET_ALL, "\rBot: ")
    on_stdout_flush()

def on_stdout_write(message, style="", prompt=""):
    function_handled = False
    for plugin in plugins:
        if hasattr(plugin, "on_stdout_write") and callable(getattr(plugin, "on_stdout_write")):
            plugin_response = getattr(plugin, "on_stdout_write")(message)
            function_handled = function_handled or plugin_response

    if not function_handled:
        if style or prompt:
            sys.stdout.write(f"{style}{prompt}{message}")
        else:
            sys.stdout.write(message)

def on_stdout_flush():
    function_handled = False
    for plugin in plugins:
        if hasattr(plugin, "on_stdout_flush") and callable(getattr(plugin, "on_stdout_flush")):
            plugin_response = getattr(plugin, "on_stdout_flush")()
            function_handled = function_handled or plugin_response

    if not function_handled:
        sys.stdout.flush()

def on_user_input(input_prompt=None):
    for plugin in plugins:
        if hasattr(plugin, "on_user_input") and callable(getattr(plugin, "on_user_input")):
            plugin_response = getattr(plugin, "on_user_input")(input_prompt)
            if plugin_response:
                return plugin_response

    if input_prompt:
        return input(input_prompt)
    else:
        return input()

def render_tools(tools):
    """Convert tools into a string format suitable for the system prompt."""
    tool_descriptions = []
    for tool in tools:
        tool_info = f"Tool name: {tool['function']['name']}\nDescription: {tool['function']['description']}\n"
        parameters = json.dumps(tool['function']['parameters'], indent=4)
        tool_info += f"Parameters:\n{parameters}\n"
        tool_descriptions.append(tool_info)
    return "\n".join(tool_descriptions)

