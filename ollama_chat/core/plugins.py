import os
import importlib.util
import inspect
from colorama import Fore, Style

from ollama_chat.core import utils
from ollama_chat.core.context import Context

from ollama_chat.core.simple_web_crawler import SimpleWebCrawler
#from ollama_chat.core.utils import on_print
from ollama_chat.core import toolman

class PluginManager:
    def __init__(self):
        self.plugins = []

#    def load_plugins(ctx:Context,  plugin_folder=None, load_plugins=True):
 #       self.plugins = discover_plugins(ctx=ctx,  plugin_folder=ctx.plugins_folder, load_plugins=load_plugins)

    def discover_plugins(self, *, plugin_folder=None, load_plugins=True, ctx:Context):

        if not load_plugins:
            if ctx.verbose:
                utils.on_print("Plugin loading is disabled.", Fore.YELLOW)
            return []

        if plugin_folder is None:
            # Get the directory of the current script (main program)
            main_dir = os.path.dirname(os.path.abspath(__file__))
            # Default plugin folder named "plugins" in the same directory
            plugin_folder = os.path.join(main_dir, "plugins")

        if not os.path.isdir(plugin_folder):
            if ctx.verbose:
                utils.on_print("Plugin folder does not exist: " + plugin_folder, Fore.RED)
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
                            utils.on_print(f"Discovered class: {name}", Fore.WHITE + Style.DIM)

                        plugin = obj()
                        if hasattr(obj, 'set_web_crawler') and callable(getattr(obj, 'set_web_crawler')):
                            plugin.set_web_crawler(SimpleWebCrawler)

                        if ctx.other_instance_url and hasattr(obj, 'set_other_instance_url') and callable(getattr(obj, 'set_other_instance_url')):
                            plugin.set_other_instance_url(ctx.other_instance_url)  # URL of the other instance to communicate with

                        if ctx.listening_port and hasattr(obj, 'set_listening_port') and callable(getattr(obj, 'set_listening_port')):
                            plugin.set_listening_port(ctx.listening_port)  # Port for this instance to listen on for communication with the other instance

                        if ctx.user_prompt and hasattr(obj, 'set_initial_message') and callable(getattr(obj, 'set_initial_message')):
                            plugin.set_initial_message(ctx.user_prompt) # Initial message to send to the other instance

                        plugins.append(plugin)
                        if ctx.verbose:
                            utils.on_print(f"Discovered plugin: {name}", Fore.WHITE + Style.DIM)
                        if hasattr(obj, 'get_tool_definition') and callable(getattr(obj, 'get_tool_definition')):
                            toolman.tool_manager.custom_tools.append(obj().get_tool_definition())
                            if ctx.verbose:
                                utils.on_print(f"Discovered tool: {name}", Fore.WHITE + Style.DIM)
        return plugins


#    def add_plugin(self, name):
#        self.plugins.append(name)

#    def process(self):
#        for p in self.plugins:
#            print(f"Processing {p}")

# istanza "singleton"
plugin_manager = PluginManager()
