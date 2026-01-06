import os
import sys
import re
import json
from colorama import Fore, Style

from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters.terminal256 import Terminal256Formatter
from ollama_chat.core import plugins
from ollama_chat.core.context import Context

#from ollama_chat.core.plugin_manager import plugin_manager

def extract_json(garbage_str, *, ctx:Context):

    if garbage_str is None:
        return []

    # First, try to parse the entire input as JSON directly
    result = try_parse_json(garbage_str, verbose=ctx.verbose)
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
        if ctx.verbose:
            on_print(f"Extracted JSON: '{json_str}'", Fore.WHITE + Style.DIM)
        result = try_parse_json(json_str, verbose=ctx.verbose)
        if result is not None:
            return result

        # If parsing fails, try to handle concatenated JSON objects
        if ctx.verbose:
            on_print("[DEBUG] Initial JSON parsing failed, attempting to handle concatenated JSON objects", Fore.CYAN + Style.DIM)

        # Try to split and merge multiple JSON objects
        merged_result = try_merge_concatenated_json(json_str, verbose=ctx.verbose)
        if merged_result is not None:
            return merged_result

        if ctx.verbose:
            on_print("Extracted string is not a valid JSON.", Fore.RED)
    else:
        if ctx.verbose:
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

    return result

def on_print(message, style="", prompt=""):
    function_handled = False
    for plugin in plugins.plugin_manager.plugins:
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

        # If not all dicts, return the last one (most recent)
        if verbose:
            on_print(f"[DEBUG] Not all objects are dicts, returning last one: {json_objects[-1]}", Fore.CYAN + Style.DIM)
        return json_objects[-1]
    if len(json_objects) == 1:
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
    for plugin in plugins.plugin_manager.plugins:
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
    for plugin in plugins.plugin_manager.plugins:
        if hasattr(plugin, "on_stdout_flush") and callable(getattr(plugin, "on_stdout_flush")):
            plugin_response = getattr(plugin, "on_stdout_flush")()
            function_handled = function_handled or plugin_response

    if not function_handled:
        sys.stdout.flush()

def on_user_input(input_prompt=None):
    for plugin in plugins.plugin_manager.plugins:
        if hasattr(plugin, "on_user_input") and callable(getattr(plugin, "on_user_input")):
            plugin_response = getattr(plugin, "on_user_input")(input_prompt)
            if plugin_response:
                return plugin_response

    if input_prompt:
        return input(input_prompt)

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

def is_docx(file_path):
    """
    Check if the given file is a DOCX file.
    """
    # Check for .docx extension
    if file_path.lower().endswith(".docx"):
        return True

    return False

def is_pptx(file_path):
    """
    Check if the given file is a PPTX file.
    """
    # Check for .pptx extension
    if file_path.lower().endswith(".pptx"):
        return True

    return False

def bytes_to_gibibytes(bytes):
    gigabytes = bytes / (1024 ** 3)
    return f"{gigabytes:.1f} GB"

def colorize(input_text, language='md'):
    try:
        lexer = get_lexer_by_name(language)
    except ValueError:
        return input_text  # Unknown language, return unchanged

    formatter = Terminal256Formatter(style='default')

    if input_text is None:
        return ""

    try:
        output = highlight(input_text, lexer, formatter)
    except:
        return input_text

    return output

def split_numbered_list(input_text):
    lines = input_text.split('\n')
    output = []
    for line in lines:
        if re.match(r'^\d+\.', line):  # Check if the line starts with a number followed by a period
            output.append(line.split('.', 1)[1].strip())  # Remove the leading number and period, then strip any whitespace
    return output

def get_personal_info():
    personal_info = {}
    user_name = os.getenv('USERNAME') or os.getenv('USER') or ""

    # Attempt to read the username from .gitconfig file
    gitconfig_path = os.path.expanduser("~/.gitconfig")
    if os.path.exists(gitconfig_path):
        with open(gitconfig_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.strip().startswith('name'):
                    user_name = line.split('=')[1].strip()
                    break

    personal_info['user_name'] = user_name
    return personal_info
