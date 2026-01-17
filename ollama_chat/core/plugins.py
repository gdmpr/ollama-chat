import os
import importlib.util
import inspect
import sys
#from typing import TYPE_CHECKING

from colorama import Fore, Style

from ollama_chat.core.context import Context
#from ollama_chat.core.simple_web_crawler import SimpleWebCrawler

# Way to have type checking for class SimpleWebCrawler avoiding circular import
#if TYPE_CHECKING:
#    from simple_web_crawler import SimpleWebCrawler

class PluginManager:
    def __init__(self):
        self.plugins = []

#    def load_plugins(ctx:Context,  plugin_folder=None, load_plugins=True):
 #       self.plugins = discover_plugins(ctx=ctx,  plugin_folder=ctx.plugins_folder, load_plugins=load_plugins)

    def discover_plugins(self, *, plugin_folder=None, load_plugins=True, ctx:Context):

        if not load_plugins:
            if ctx.verbose:
                on_print("Plugin loading is disabled.", Fore.YELLOW)
            return []

        if plugin_folder is None:
            # Get the directory of the current script (main program)
            main_dir = os.path.dirname(os.path.abspath(__file__))
            # Default plugin folder named "plugins" in the same directory
            plugin_folder = os.path.join(main_dir, "plugins")

        if not os.path.isdir(plugin_folder):
            if ctx.verbose:
                on_print("Plugin folder does not exist: " + plugin_folder, Fore.RED)
            return []

        plugins = []
        for filename in os.listdir(plugin_folder):
            if filename.endswith(".py") and not filename.startswith("__"):
                module_name = filename[:-3]
                module_path = os.path.join(plugin_folder, filename)
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                for name, obj in inspect.getmembers(module):
                    if inspect.isclass(obj) and "plugin" in name.lower():
                        if ctx.verbose:
                            on_print(f"Discovered class: {name}", Fore.WHITE + Style.DIM)

                        plugin = obj()
                        if hasattr(obj, 'set_web_crawler') and callable(getattr(obj, 'set_web_crawler')):
                            plugin.set_web_crawler(self.__create_crawler())

                        if (
                            ctx.other_instance_url
                            and hasattr(obj, 'set_other_instance_url')
                            and callable(getattr(obj, 'set_other_instance_url'))
                        ):
                            plugin.set_other_instance_url(ctx.other_instance_url)  # URL of the other instance to communicate with

                        if (
                            ctx.listening_port
                            and hasattr(obj, 'set_listening_port')
                            and callable(getattr(obj, 'set_listening_port'))
                        ):
                            plugin.set_listening_port(ctx.listening_port)  # Port for this instance to listen on for communication with the other instance

                        if (
                            ctx.user_prompt
                            and hasattr(obj, 'set_initial_message')
                            and callable(getattr(obj, 'set_initial_message'))
                        ):
                            plugin.set_initial_message(ctx.user_prompt) # Initial message to send to the other instance

                        plugins.append(plugin)
                        if ctx.verbose:
                            on_print(f"Discovered plugin: {name}", Fore.WHITE + Style.DIM)
                        if hasattr(obj, 'get_tool_definition') and callable(getattr(obj, 'get_tool_definition')):
                            ctx.tool_manager.custom_tools.append(obj().get_tool_definition())
                            if ctx.verbose:
                                on_print(f"Discovered tool: {name}", Fore.WHITE + Style.DIM)
        return plugins

    def __create_crawler(self):
        #from simple_web_crawler import SimpleWebCrawler
        from ollama_chat.core.simple_web_crawler import SimpleWebCrawler
        return SimpleWebCrawler

#    def add_plugin(self, name):
#        self.plugins.append(name)

#    def process(self):
#        for p in self.plugins:
#            print(f"Processing {p}")

# istanza "singleton"
plugin_manager = PluginManager()


def on_print(message, style="", prompt=""):
    function_handled = False
    for plugin in plugin_manager.plugins:
        if hasattr(plugin, "on_print") and callable(getattr(plugin, "on_print")):
            plugin_response = getattr(plugin, "on_print")(message)
            function_handled = function_handled or plugin_response

    if not function_handled:
        if style or prompt:
            print(f"{style}{prompt}{message}")
        else:
            print(message)


def on_stdout_write(message, style="", prompt=""):
    function_handled = False
    for plugin in plugin_manager.plugins:
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
    for plugin in plugin_manager.plugins:
        if hasattr(plugin, "on_stdout_flush") and callable(getattr(plugin, "on_stdout_flush")):
            plugin_response = getattr(plugin, "on_stdout_flush")()
            function_handled = function_handled or plugin_response

    if not function_handled:
        sys.stdout.flush()

def on_user_input(input_prompt=None):
    for plugin in plugin_manager.plugins:
        if hasattr(plugin, "on_user_input") and callable(getattr(plugin, "on_user_input")):
            plugin_response = getattr(plugin, "on_user_input")(input_prompt)
            if plugin_response:
                return plugin_response

    if input_prompt:
        return input(input_prompt)

    return input()


class ToolManager:
    def __init__(self):
        self.custom_tools = []

    def get_available_tools(self,  *,  ctx:Context):

        #load_chroma_client(ctx=ctx)

        # List existing collections
        available_collections = []
        available_collections_description = []
        if ctx.chroma_client:
            collections = ctx.chroma_client.list_collections()

            for collection in collections:
                if collection.name in [ctx.web_cache_collection_name, ctx.memory_collection_name]:
                    continue
                available_collections.append(collection.name)

                if isinstance(collection.metadata, dict) and "description" in collection.metadata:
                    available_collections_description.append(
                        f"'{collection.name}': {collection.metadata['description']}"
                    )

        default_tools = [{
            'type': 'function',
            'function': {
                'name': 'web_search',
                'description': 'Perform a web search using DuckDuckGo',
                'parameters': {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        },
                        "region": {
                            "type": "string",
                            "description": "Region for search results, e.g., 'us-en' for United States, 'fr-fr' for France, etc... or 'wt-wt' for No region",
                            "default": "wt-wt"
                        }
                    },
                    "required": [
                        "query"
                    ]
                }
            }
        },
        {
            'type': 'function',
            'function': {
                'name': 'query_vector_database',
                'description': 'Performs a semantic search using a knowledge base collection.',
                'parameters': {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The question to search for, in a human-readable format, e.g., 'What is the capital of France?'"
                        },
                        "collection_name": {
                            "type": "string",
                            "description": f"The name of the collection to search in, which must be one of the available collections: {', '.join(available_collections_description)}",
                            "default": ctx.current_collection_name,
                            "enum": available_collections
                        },
                        "question_context": {
                            "type": "string",
                            "description": "Current discussion context or topic, based on previous exchanges with the user"
                        },
                        "include_full_docs": {
                            "type": "boolean",
                            "description": "If true, includes the full original documents along with the relevant chunks in the response. This provides complete context but may return more text.",
                            "default": False
                        }
                    },
                    "required": [
                        "question",
                        "collection_name",
                        "question_context"
                    ]
                }
            }
        },
        {
            'type': 'function',
            'function': {
                'name': 'retrieve_relevant_memory',
                'description': 'Retrieve relevant memories based on a query',
                'parameters': {
                    "type": "object",
                    "properties": {
                        "query_text": {
                            "type": "string",
                            "description": "The query or question for which relevant memories should be retrieved"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of relevant memories to retrieve",
                            "default": 3
                        }
                    },
                    "required": [
                        "query_text"
                    ]
                }
            }
        },
        {
            'type': 'function',
            'function': {
                "name": "instantiate_agent_with_tools_and_process_task",
                "description": (
                    "✅ PRIMARY AGENT FUNCTION: Creates a specialized agent and IMMEDIATELY executes a task, returning actual results. "
                    "Use this when the user wants an agent to DO something (search, analyze, investigate, research, etc.). "
                    "The agent will break down the task, use the provided tools, and return findings. "
                    "Example: 'Create an agent to search for X' → Use this function with task='search for X'. "
                    "Tools must be chosen from a predefined set."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "The task or problem that the agent needs to solve. Provide a clear and concise description."
                        },
                        "system_prompt": {
                            "type": "string",
                            "description": "The system prompt that defines the agent's behavior, personality, and approach to solving the task."
                        },
                        "tools": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": []
                            },
                            "description": "A list of tools to be used by the agent for solving the task. Must be provided as an array of tool names."
                        },
                        "agent_name": {
                            "type": "string",
                            "description": "A unique name for the agent that will be used for instantiation."
                        },
                        "agent_description": {
                            "type": "string",
                            "description": "A brief description of the agent's purpose and capabilities."
                        }
                    },
                    "required": ["task", "system_prompt", "tools", "agent_name", "agent_description"]
                }
            }
        },
        {
            'type': 'function',
            'function': {
                "name": "create_new_agent_with_tools",
                "description": (
                    "Creates an new agent with a specified name and description, using a provided system prompt and a list of tools. "
                    "The tools must be chosen from a predefined set."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "system_prompt": {
                            "type": "string",
                            "description": "The system prompt that defines the agent's behavior, personality, and approach to solving the task."
                        },
                        "tools": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": []
                            },
                            "description": "A list of tools to be used by the agent for solving the task. Must be provided as an array of tool names."
                        },
                        "agent_name": {
                            "type": "string",
                            "description": "A unique name for the agent that will be used for instantiation."
                        },
                        "agent_description": {
                            "type": "string",
                            "description": "A brief description of the agent's purpose and capabilities."
                        }
                    },
                    "required": ["system_prompt", "tools", "agent_name", "agent_description"]
                }
            }
        },
        {
            'type': 'function',
            'function': {
                'name': 'summarize_text_file',
                'description': 'Summarizes a long text file by breaking it into chunks and summarizing them iteratively.',
                'parameters': {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The long text file to summarize. Provide the full path to the file."
                        },
                        "max_final_words": {
                            "type": "integer",
                            "description": "The maximum number of words desired for the final summary.",
                            "default": 500
                        },
                        "language": {
                            "type": "string",
                            "description": "Language in which intermediate and final summaries should be produced (e.g. 'English', 'French'). Use language specified by the user, or the language of the conversation if known.",
                            "default": "English"
                        }
                    },
                    "required": ["file_path"]
                }
            }
        },
        {
            'type': 'function',
            'function': {
                'name': 'read_file',
                'description': 'Read the contents of a file and return the text',
                'parameters': {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The full path to the file to read"
                        },
                        "encoding": {
                            "type": "string",
                            "description": "The encoding to use when reading the file (e.g., 'utf-8', 'ascii', 'latin-1')",
                            "default": "utf-8"
                        }
                    },
                    "required": ["file_path"]
                }
            }
        },
        {
            'type': 'function',
            'function': {
                'name': 'create_file',
                'description': 'Create a new file with the given content. The file will be tracked in the session for safe deletion.',
                'parameters': {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The full path where the file should be created. Parent directories will be created if they do not exist."
                        },
                        "content": {
                            "type": "string",
                            "description": "The content to write to the file"
                        },
                        "encoding": {
                            "type": "string",
                            "description": "The encoding to use when writing the file (e.g., 'utf-8', 'ascii', 'latin-1')",
                            "default": "utf-8"
                        }
                    },
                    "required": ["file_path", "content"]
                }
            }
        },
        {
            'type': 'function',
            'function': {
                'name': 'delete_file',
                'description': 'Delete a file that was created during this session. Only files created with the create_file tool can be deleted.',
                'parameters': {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The full path to the file to delete. Must be a file that was created during this session."
                        }
                    },
                    "required": ["file_path"]
                }
            }
        }]

        # Find index of instantiate_agent_with_tools_and_process_task function
        index = -1
        for i, tool in enumerate(default_tools):
            if tool['function']['name'] == 'instantiate_agent_with_tools_and_process_task':
                index = i
                break

        default_tools[index]["function"]["parameters"]["properties"]["tools"]["items"]["enum"] = [tool["function"]["name"] for tool in ctx.selected_tools]
        default_tools[index]["function"]["parameters"]["properties"]["tools"]["description"] += f" Available tools: {', '.join([tool['function']['name'] for tool in ctx.selected_tools])}"

        # Find index of create_new_agent_with_tools function
        index = -1
        for i, tool in enumerate(default_tools):
            if tool['function']['name'] == 'create_new_agent_with_tools':
                index = i
                break

        default_tools[index]["function"]["parameters"]["properties"]["tools"]["items"]["enum"] = [tool["function"]["name"] for tool in ctx.selected_tools]
        default_tools[index]["function"]["parameters"]["properties"]["tools"]["description"] += f" Available tools: {', '.join([tool['function']['name'] for tool in ctx.selected_tools])}"

        # Add custom tools from plugins
        available_tools = default_tools + tool_manager.custom_tools
        return available_tools


    def select_tool_by_name(self, available_tools, target_tool_name, *,  ctx: Context):
        for tool in available_tools:
            if tool['function']['name'].lower() == target_tool_name.lower():
                if tool not in ctx.selected_tools:
                    ctx.selected_tools.append(tool)

                    if ctx.verbose:
                        on_print(f"Tool '{target_tool_name}' selected.\n")
                else:
                    on_print(f"Tool '{target_tool_name}' is already selected.\n")
                return ctx.selected_tools

        on_print(f"Tool '{target_tool_name}' not found.\n")
        return ctx.selected_tools


    def select_tools(self,  available_tools, selected_tools):
        def display_tool_options():
            on_print("Available tools:\n", Style.RESET_ALL)
            for i, tool in enumerate(available_tools):
                tool_name = tool['function']['name']

                status = "[ ]"
                # Find current tool name in selected tools
                for selected_tool in selected_tools:
                    if selected_tool['function']['name'] == tool_name:
                        status = "[X]"
                        break

                on_print(f"{i + 1}. {status} {tool_name}: {tool['function']['description']}")

        while True:
            display_tool_options()
            on_print("Select or deselect tools by entering the corresponding number (e.g., 1).\nPress Enter or type 'done' when done.")

            user_input = on_user_input("Your choice: ").strip()

            if len(user_input) == 0 or user_input == 'done':
                break

            try:
                index = int(user_input) - 1
                if 0 <= index < len(available_tools):
                    selected_tool = available_tools[index]
                    if selected_tool in selected_tools:
                        selected_tools.remove(selected_tool)
                        on_print(f"Tool '{selected_tool['function']['name']}' deselected.\n")
                    else:
                        selected_tools.append(selected_tool)
                        on_print(f"Tool '{selected_tool['function']['name']}' selected.\n")
                else:
                    on_print("Invalid selection. Please choose a valid tool number.\n")
            except ValueError:
                on_print("Invalid input. Please enter a number corresponding to a tool or 'done'.\n")

        return selected_tools

    def list_tools(self, *,  ctx:Context):
        # Load plugins first
        plugin_manager.plugins = plugin_manager.discover_plugins(ctx=ctx, plugin_folder=ctx.plugins_folder, load_plugins=True)  # Always load for --list-tools
        if ctx.verbose:
            on_print(f"\nDiscovered {len(plugin_manager.plugins)} plugins")

        tools = tool_manager.get_available_tools(ctx=ctx)
        on_print("\nAvailable tools:")

        # Split tools into built-in and plugin tools
        builtin_tools = [tool for tool in tools if not any(pt['function']['name'] == tool['function']['name'] for p in plugin_manager.plugins for pt in ([p.get_tool_definition()] if hasattr(p, 'get_tool_definition') and callable(getattr(p, 'get_tool_definition')) else []))]
        plugin_tools = [tool for tool in tools if tool not in builtin_tools]

        # Print built-in tools
        if builtin_tools:
            on_print("\nBuilt-in tools:")
            for tool in builtin_tools:
                on_print(f"\n{tool['function']['name']}:")
                on_print(f"  Description: {tool['function']['description']}")
                if 'parameters' in tool['function']:
                    if 'properties' in tool['function']['parameters']:
                        on_print("  Parameters:")
                        for param_name, param_info in tool['function']['parameters']['properties'].items():
                            required = param_name in tool['function']['parameters'].get('required', [])
                            on_print(f"    {param_name}{'*' if required else ''}: {param_info['description']}")

        # Print plugin tools
        if plugin_tools:
            on_print("\nPlugin tools:")
            for tool in plugin_tools:
                on_print(f"\n{tool['function']['name']}:")
                on_print(f"  Description: {tool['function']['description']}")
                if 'parameters' in tool['function']:
                    if 'properties' in tool['function']['parameters']:
                        on_print("  Parameters:")
                        for param_name, param_info in tool['function']['parameters']['properties'].items():
                            required = param_name in tool['function']['parameters'].get('required', [])
                            on_print(f"    {param_name}{'*' if required else ''}: {param_info['description']}")


    def get_builtin_tool_names(self):
        """
        Returns a list of built-in tool names (tools that don't come from plugins).
        """
        builtin_tools = [
            'web_search',
            'query_vector_database',
            'retrieve_relevant_memory',
            'instantiate_agent_with_tools_and_process_task',
            'create_new_agent_with_tools',
            'summarize_text_file'
        ]
        return builtin_tools


# istanza "singleton"
tool_manager = ToolManager()
