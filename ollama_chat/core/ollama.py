import sys
import json
import ollama

from colorama import Fore, Style

from ollama_chat.core import utils
from ollama_chat.core import plugins
from ollama_chat.core.context import Context

def ask_ollama(
    system_prompt:str,
    user_input,
    selected_model,
    temperature=0.1,
    prompt_template=None,
    tools=None,
    no_bot_prompt=False,
    stream_active=True,
    num_ctx=None,
    use_think_mode=False,
    *,
    ctx:Context
):
    """
    Submits to the specified model a system prompt and user input and returns the model's answer

    @param system_prompt The system prompt for the model
    @type TYPE
    @param user_input The user imput to submit to the model
    @type TYPE
    @param selected_model The model to use
    @type TYPE
    @param temperature The temperature for the model generation (defaults to 0.1)
    @type TYPE (optional)
    @param prompt_template DESCRIPTION (defaults to None)
    @type TYPE (optional)
    @param tools Array of tools the model can use. (defaults to None)
    @type TYPE (optional)
    @param no_bot_prompt DESCRIPTION (defaults to False)
    @type TYPE (optional)
    @param stream_active DESCRIPTION (defaults to True)
    @type TYPE (optional)
    @param num_ctx Context size of the model (defaults to None)
    @type TYPE (optional)
    @param use_think_mode DESCRIPTION (defaults to False)
    @type TYPE (optional)
    @keyparam ctx The application context object
    @type Context
    @return Returns the model response
    @rtype str
    """

    if tools is None:
        tools = []

    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]
    answer = ask_ollama_with_conversation(
        conversation,
        selected_model,
        temperature,
        prompt_template,
        tools,
        no_bot_prompt,
        stream_active,
        num_ctx=num_ctx,
        use_think_mode=use_think_mode,
        ctx=ctx
    )

    return answer

def ask_ollama_with_conversation(
    conversation,
    model,
    temperature=0.1,
    prompt_template=None,
    tools=None,
    no_bot_prompt=False,
    stream_active=True,
    prompt="Bot",
    prompt_color=None,
    num_ctx=None,
    use_think_mode=False,
    *,
    ctx:Context
):

    if ctx.verbose:
        plugins.on_print(f"ask_ollama_with_conversation using model {model}.", Fore.WHITE + Style.DIM)

    # Default value for tools
    if tools is None:
        tools = []

    # Some models do not support the "system" role, merge the system message with the first user message
    if (
        ctx.no_system_role
        and len(conversation) > 1
        and conversation[0]["role"] == "system"
        and not conversation[0]["content"] is None
        and not conversation[1]["content"] is None
    ):
        conversation[1]["content"] = conversation[0]["content"] + "\n" + conversation[1]["content"]
        conversation = conversation[1:]

    model_is_an_ollama_model = is_model_an_ollama_model(model,  ctx=ctx)

    if (ctx.use_openai or ctx.use_azure_openai) and not model_is_an_ollama_model:
        if ctx.verbose:
            plugins.on_print("Using OpenAI API for conversation generation.", Fore.WHITE + Style.DIM)

    if not ctx.syntax_highlighting:
        if ctx.interactive_mode and not no_bot_prompt:
            if prompt_color:
                on_prompt(f"{prompt}: ", prompt_color)
            else:
                on_prompt(f"{prompt}: ", Style.RESET_ALL)
        else:
            if prompt_color:
                plugins.on_stdout_write("", prompt_color)
            else:
                plugins.on_stdout_write("", Style.RESET_ALL)
        plugins.on_stdout_flush()

    model_support_tools = True

    if (ctx.use_openai or ctx.use_azure_openai) and not model_is_an_ollama_model:

        completion_done = False
        while not completion_done:
            bot_response, bot_response_is_tool_calls, completion_done = ask_openai_with_conversation(
                conversation,
                model,
                temperature,
                prompt_template,
                stream_active,
                tools,
                ctx=ctx
            )
            if bot_response and bot_response_is_tool_calls:
                # Convert bot_response list of objects to a list of dict
                bot_response = [json.loads(json.dumps(obj, default=lambda o: vars(o))) for obj in bot_response]

                if ctx.verbose:
                    plugins.on_print(f"Bot response: {bot_response}", Fore.WHITE + Style.DIM)

                bot_response = handle_tool_response(
                    bot_response,
                    model_support_tools,
                    conversation,
                    model,
                    temperature,
                    prompt_template,
                    tools,
                    stream_active,
                    num_ctx=num_ctx,
                    ctx=ctx
                )

                # Consider completion done
                completion_done = True
        if not bot_response is None:
            return bot_response.strip()

        return None

    bot_response = ""
    bot_thinking_response = ""
    bot_response_is_tool_calls = False
    ollama_options = {"temperature": temperature}
    if num_ctx:
        ollama_options["num_ctx"] = num_ctx

    think = use_think_mode or ctx.think_mode_on

    if ctx.verbose and think:
        plugins.on_print("Thinking...", Fore.WHITE + Style.DIM)

    try:
        stream = ollama.chat(
            model=model,
            messages=conversation,
            # If tools are selected, deactivate the stream to get the full response (Ollama API limitation)
            stream=False if len(tools) > 0 else stream_active,
            options=ollama_options,
            tools=tools,
            think=think
        )
    except ollama.ResponseError as e:
        if "does not support tools" in str(e):
            tool_response = generate_tool_response(
                find_latest_user_message(conversation),
                tools,
                model,
                temperature,
                prompt_template,
                num_ctx=num_ctx,
                ctx=ctx
            )

            if not tool_response is None and len(tool_response) > 0:
                bot_response = tool_response
                bot_response_is_tool_calls = True
                model_support_tools = False
            else:
                return ""
        else:
            plugins.on_print(f"An error occurred during the conversation: {e}", Fore.RED)
            return ""

    if not bot_response_is_tool_calls:
        try:
            if stream_active and len(tools) == 0:
                if ctx.alternate_model:
                    plugins.on_print(f"Response from model: {model}\n")
                chunk_count = 0
                for chunk in stream:
                    continue_response_generation = True
                    for plugin in plugins.plugin_manager.plugins:
                        if hasattr(plugin, "stop_generation") and callable(getattr(plugin, "stop_generation")):
                            plugin_response = getattr(plugin, "stop_generation")()
                            if plugin_response:
                                continue_response_generation = False
                                break

                    if not continue_response_generation:
                        stream.close()
                        break

                    chunk_count += 1

                    thinking_delta = ""
                    if think:
                        thinking_delta = chunk['message'].get('thinking', '')

                        if thinking_delta is None:
                            thinking_delta = ""
                        else:
                            bot_thinking_response += thinking_delta

                    delta = chunk['message'].get('content', '')

                    if len(bot_response) == 0 and len(thinking_delta) == 0:
                        delta = delta.strip()

                        if len(delta) == 0:
                            continue

                    bot_response += delta

                    if ctx.syntax_highlighting and ctx.interactive_mode:
                        utils.print_spinning_wheel(chunk_count)
                    else:
                        if think and len(thinking_delta) > 0:
                            on_llm_thinking_token_response(thinking_delta, Fore.WHITE + Style.DIM)
                        else:
                            on_llm_token_response(delta, Fore.WHITE + Style.NORMAL)
                        plugins.on_stdout_flush()

                on_llm_token_response("\n")
                plugins.on_stdout_flush()
            else:
                tool_calls = stream['message'].get('tool_calls', [])
                if tool_calls is None:
                    tool_calls = []

                if len(tool_calls) > 0:
                    conversation.append(stream['message'])

                    if ctx.verbose:
                        plugins.on_print(f"Tool calls: {tool_calls}", Fore.WHITE + Style.DIM)
                    bot_response = tool_calls
                    bot_response_is_tool_calls = True
                else:
                    if think:
                        bot_thinking_response = stream['message'].get('thinking', '')
                    bot_response = stream['message']['content']
        except KeyboardInterrupt:
            stream.close()
        except ollama.ResponseError as e:
            plugins.on_print(f"An error occurred during the conversation: {e}", Fore.RED)
            return ""

    # Check if the bot response is a tool call object
    if (
        not bot_response_is_tool_calls
        and bot_response
        and len(bot_response.strip()) > 0
        and bot_response.strip()[0] == "{"
        and bot_response.strip()[-1] == "}"
    ):
        bot_response = [utils.extract_json(bot_response.strip(), ctx=ctx)]
        bot_response_is_tool_calls = True

    # Check if the bot response is a list of tool calls
    if (
        not bot_response_is_tool_calls
        and bot_response
        and len(bot_response.strip()) > 0
        and bot_response.strip()[0] == "["
        and bot_response.strip()[-1] == "]"
    ):
        bot_response = utils.extract_json(bot_response.strip(), ctx=ctx)
        bot_response_is_tool_calls = True

    # Check if the bot response starts with <tool_call>
    if (
        not bot_response_is_tool_calls
        and bot_response
        and len(bot_response.strip()) > 0
        and bot_response.startswith("<tool_call>")
    ):
        bot_response = utils.extract_json(bot_response.strip(), ctx=ctx)
        bot_response_is_tool_calls = True

    if bot_response and bot_response_is_tool_calls:
        bot_response = handle_tool_response(
            bot_response,
            model_support_tools,
            conversation,
            model,
            temperature,
            prompt_template,
            tools,
            stream_active,
            num_ctx=num_ctx,
            ctx=ctx
        )

    if isinstance(bot_response, str):
        return bot_response.strip()

    return None

def is_model_an_ollama_model(model_name:str, *, ctx:Context):
    """
    Returns true if the model_name in in the Ollama list of available models.

    @param model_name The name of the model to check
    @type str
    """
    try:
        models = ollama.list()["models"]
    except:
        if ctx.verbose:
            plugins.on_print("Error while retriving the Ollama models list.", Fore.RED + Style.DIM)
        return False

    for model in models:
        if model["model"] == model_name:
            return True

    return False


def on_prompt(prompt, style=""):
    function_handled = False
    for plugin in plugins.plugin_manager.plugins:
        if hasattr(plugin, "on_prompt") and callable(getattr(plugin, "on_prompt")):
            plugin_response = getattr(plugin, "on_prompt")(prompt)
            function_handled = function_handled or plugin_response

    if not function_handled:
        if style:
            sys.stdout.write(f"{style}{prompt}")
        else:
            sys.stdout.write(prompt)


def handle_tool_response(
    bot_response,
    model_support_tools,
    conversation,
    model,
    temperature,
    prompt_template,
    tools,
    stream_active,
    num_ctx=None,
    *,
    ctx:Context
):
    if ctx.verbose:
        plugins.on_print(f"Call to {handle_tool_response.__name__}", Fore.WHITE + Style.DIM)

    # Iterate over each function call in the bot response
    tool_found = False
    for tool_call in bot_response:
        if not 'function' in tool_call:
            tool_call = { 'function': tool_call }
            if not 'name' in tool_call['function']:
                continue

        tool_name = tool_call['function']['name']
        # Iterate over the available tools
        for tool in tools:
            if 'type' in tool and tool['type'] == 'function' and 'function' in tool and 'name' in tool['function'] and tool['function']['name'] == tool_name:
                # Test if tool_call['function'] as arguments
                if 'arguments' in tool_call:
                    # Extract parameters for the tool function
                    parameters = tool_call.get('arguments', {})  # Update: get parameters from the 'arguments' key
                else:
                    # Call the tool function with the parameters
                    parameters = tool_call['function'].get('arguments', {})

                tool_response = None

                # Debug logging for parameters
                if ctx.verbose:
                    plugins.on_print(f"[DEBUG] Initial parameters: {parameters}", Fore.CYAN + Style.DIM)
                    plugins.on_print(f"[DEBUG] Parameters type: {type(parameters)}", Fore.CYAN + Style.DIM)

                # if parameters is a string, convert it to a dictionary
                if isinstance(parameters, str):
                    if ctx.verbose:
                        plugins.on_print("[DEBUG] Converting string parameters to dict", Fore.CYAN + Style.DIM)
                    try:
                        parameters = utils.extract_json(parameters,  ctx=ctx)
                        if ctx.verbose:
                            plugins.on_print(f"[DEBUG] After extract_json: {parameters} (type: {type(parameters)})", Fore.CYAN + Style.DIM)
                    except Exception as e:
                        if ctx.verbose:
                            plugins.on_print(f"[DEBUG] extract_json failed: {e}, using empty dict", Fore.CYAN + Style.DIM)
                        parameters = {}

                # Ensure parameters is always a dict
                # If it's a list, try to convert it based on the tool's parameter definition
                if isinstance(parameters, list):
                    if ctx.verbose:
                        plugins.on_print("[DEBUG] Parameters is a list, attempting to convert to dict", Fore.CYAN + Style.DIM)
                    # Try to map list items to parameter names from the tool definition
                    if 'parameters' in tool.get('function', {}) and 'properties' in tool['function']['parameters']:
                        param_names = list(tool['function']['parameters']['properties'].keys())
                        if ctx.verbose:
                            plugins.on_print(f"[DEBUG] Parameter names from tool definition: {param_names}", Fore.CYAN + Style.DIM)
                            plugins.on_print(f"[DEBUG] List values: {parameters}", Fore.CYAN + Style.DIM)
                        if len(param_names) > 0 and len(parameters) > 0:
                            # Create dict mapping parameter names to list values
                            parameters = {name: value for name, value in zip(param_names, parameters)}
                            if ctx.verbose:
                                plugins.on_print(f"[DEBUG] Converted list to dict: {parameters}", Fore.CYAN + Style.DIM)
                        else:
                            parameters = {}
                    else:
                        if ctx.verbose:
                            plugins.on_print("[DEBUG] No parameter definition found in tool, using empty dict", Fore.CYAN + Style.DIM)
                        parameters = {}
                elif not isinstance(parameters, dict):
                    # If it's neither string, list, nor dict, convert to empty dict
                    if ctx.verbose:
                        plugins.on_print(f"[DEBUG] Parameters is {type(parameters)}, converting to empty dict", Fore.CYAN + Style.DIM)
                    parameters = {}

                if ctx.verbose:
                    plugins.on_print(f"[DEBUG] Final parameters before tool call: {parameters} (type: {type(parameters)})", Fore.CYAN + Style.DIM)

                # Filter parameters to only include those accepted by the function
                # First, try to get accepted parameters from tool definition
                accepted_params = set()
                if 'parameters' in tool.get('function', {}) and 'properties' in tool['function']['parameters']:
                    accepted_params = set(tool['function']['parameters']['properties'].keys())

                if ctx.verbose and accepted_params:
                    plugins.on_print(f"[DEBUG] Accepted parameters from tool definition: {accepted_params}", Fore.CYAN + Style.DIM)

                # Filter the parameters to only include accepted ones
                if accepted_params and isinstance(parameters, dict):
                    original_params = parameters.copy()
                    parameters = {k: v for k, v in parameters.items() if k in accepted_params}

                    if ctx.verbose and original_params != parameters:
                        plugins.on_print(f"[DEBUG] Filtered parameters: removed {set(original_params.keys()) - set(parameters.keys())}", Fore.CYAN + Style.DIM)
                        plugins.on_print(f"[DEBUG] Parameters after filtering: {parameters}", Fore.CYAN + Style.DIM)

                # Check if the tool is a globally defined function
                if tool_name in globals():
                    if ctx.verbose:
                        plugins.on_print(f"Calling tool function: {tool_name} with parameters: {parameters}", Fore.WHITE + Style.DIM)
                    try:
                        # Call the global function with extracted parameters
                        tool_response = globals()[tool_name](**parameters)
                        if ctx.verbose:
                            plugins.on_print(f"Tool response: {tool_response}", Fore.WHITE + Style.DIM)
                        tool_found = True
                    except Exception as e:
                        plugins.on_print(f"Error calling tool function: {tool_name} - {e}", Fore.RED + Style.NORMAL)
                else:
                    if ctx.verbose:
                        plugins.on_print(f"Trying to find plugin with function '{tool_name}'...", Fore.WHITE + Style.DIM)
                    # Search for the tool function in plugins
                    for plugin in plugins.plugin_manager.plugins:
                        if hasattr(plugin, tool_name) and callable(getattr(plugin, tool_name)):
                            tool_found = True
                            if ctx.verbose:
                                plugins.on_print(f"Calling tool function: {tool_name} from plugin: {plugin.__class__.__name__} with arguments {parameters}", Fore.WHITE + Style.DIM)

                            try:
                                # Call the plugin's tool function with parameters
                                tool_response = getattr(plugin, tool_name)(**parameters)
                                if ctx.verbose:
                                    plugins.on_print(f"Tool response: {tool_response}", Fore.WHITE + Style.DIM)
                                break
                            except Exception as e:
                                plugins.on_print(f"Error calling tool function: {tool_name} - {e}", Fore.RED + Style.NORMAL)

                if not tool_response is None:
                    # If the tool response is a string, append it to the conversation
                    tool_role = "tool"
                    tool_call_id = tool_call.get('id', 0)

                    if not model_support_tools:
                        tool_role = "user"
                    if isinstance(tool_response, str):
                        if not model_support_tools:
                            latest_user_message = find_latest_user_message(conversation)
                            if latest_user_message:
                                tool_response += "\n" + latest_user_message
                        conversation.append({"role": tool_role, "content": tool_response, "tool_call_id": tool_call_id})
                    else:
                        # Convert the tool response to a string
                        tool_response_str = json.dumps(tool_response, indent=4)
                        if not model_support_tools:
                            latest_user_message = find_latest_user_message(conversation)
                            if latest_user_message:
                                tool_response_str += "\n" + latest_user_message
                        conversation.append({"role": tool_role, "content": tool_response_str, "tool_call_id": tool_call_id})
    if tool_found:
        # Pass the tools back so the model can make follow-up tool calls if needed
        bot_response = ask_ollama_with_conversation(conversation, model, temperature, prompt_template, tools=tools, no_bot_prompt=True, stream_active=stream_active, num_ctx=num_ctx,  ctx=ctx)
    else:
        plugins.on_print("Tools not found", Fore.RED)
        return None

    return bot_response


def ask_openai_with_conversation(
    conversation,
    selected_model=None,
    temperature=0.1,
    prompt_template=None,
    stream_active=True,
    tools=None,
    *,
    ctx:Context
):

    if tools is None:
        tools = []

    if prompt_template == "ChatML":
        # Modify conversation to match prompt template: ChatML
        # See https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-code-ft-GGUF for the ChatML prompt template
        #        '''
        #        <|im_start|>system
        #        {system_message}<|im_end|>
        #        <|im_start|>user
        #        {prompt}<|im_end|>
        #        <|im_start|>assistant
        #        '''

        for i, message in enumerate(conversation):
            if message["role"] == "system":
                conversation[i]["content"] = "<|im_start|>system\n" + message["content"] + "<|im_end|>"
            elif message["role"] == "user":
                conversation[i]["content"] = "<|im_start|>user\n" + message["content"] + "<|im_end|>"
            elif message["role"] == "assistant":
                conversation[i]["content"] = "<|im_start|>assistant\n" + message["content"] + "<|im_end|>"

        # Add assistant message to the end of the conversation
        conversation.append({"role": "assistant", "content": "<|im_start|>assistant\n"})

    if prompt_template == "Alpaca":
        # Modify conversation to match prompt template: Alpaca
        # See https://github.com/tatsu-lab/stanford_alpaca for the Alpaca prompt template
        for i, message in enumerate(conversation):
            if message["role"] == "system":
                conversation[i]["content"] = "### Instruction:\n" + message["content"]
            elif message["role"] == "user":
                conversation[i]["content"] = "### Input:\n" + message["content"]

        # Add assistant message to the end of the conversation
        conversation.append({"role": "assistant", "content": "### Response:\n"})

    if len(tools) == 0:
        tools = None

    completion_done = False
    completion = None
    try:
        completion = ctx.openai_client.chat.completions.create(
            messages=conversation,
            model=selected_model,
            stream=stream_active,
            temperature=temperature,
            tools=tools
        )
    except Exception as e:
        plugins.on_print(f"Error during OpenAI completion: {e}", Fore.RED)
        return "", False, True

    bot_response_is_tool_calls = False
    tool_calls = []

    if hasattr(completion, 'choices') and len(completion.choices) > 0 and hasattr(completion.choices[0], 'message') and hasattr(completion.choices[0].message, 'tool_calls'):
        tool_calls = completion.choices[0].message.tool_calls

        # Test if tool_calls is a list
        if not isinstance(tool_calls, list):
            tool_calls = []

    if len(tool_calls) > 0:
        conversation.append(completion.choices[0].message)

        if ctx.verbose:
            plugins.on_print(f"Tool calls: {tool_calls}", Fore.WHITE + Style.DIM)
        bot_response = tool_calls
        bot_response_is_tool_calls = True

    else:
        if not stream_active:
            bot_response = completion.choices[0].message.content

            if ctx.verbose:
                plugins.on_print(f"Bot response: {bot_response}", Fore.WHITE + Style.DIM)

            # Check if the completion is done based on the finish reason
            if completion.choices[0].finish_reason in ('stop', 'function_call', 'content_filter', 'tool_calls'):
                completion_done = True
        else:
            bot_response = ""
            try:
                chunk_count = 0
                for chunk in completion:
                    delta = chunk.choices[0].delta.content

                    if not delta is None:
                        if ctx.syntax_highlighting and ctx.interactive_mode:
                            utils.print_spinning_wheel(chunk_count)
                        else:
                            on_llm_token_response(delta, Style.RESET_ALL)
                            plugins.on_stdout_flush()
                        bot_response += delta
                    elif isinstance(chunk.choices[0].delta.tool_calls, list) and len(chunk.choices[0].delta.tool_calls) > 0:
                        if isinstance(bot_response, str) and not bot_response_is_tool_calls:
                            bot_response = chunk.choices[0].delta.tool_calls
                            bot_response_is_tool_calls = True
                        elif isinstance(bot_response, list) and bot_response_is_tool_calls:
                            for tool_call, tool_call_index in zip(chunk.choices[0].delta.tool_calls, range(len(chunk.choices[0].delta.tool_calls))):
                                bot_response[tool_call_index].function.arguments += tool_call.function.arguments

                    # Check if the completion is done based on the finish reason
                    if chunk.choices[0].finish_reason in ('stop', 'function_call', 'content_filter', 'tool_calls'):
                        completion_done = True
                        break

                    chunk_count += 1

                if bot_response_is_tool_calls:
                    conversation.append({"role": "assistant", "tool_calls": bot_response})

            except KeyboardInterrupt:
                completion.close()
            except Exception as e:
                plugins.on_print(f"Error during streaming completion: {e}", Fore.RED)
                bot_response = ""
                bot_response_is_tool_calls = False

    if not completion_done and not bot_response_is_tool_calls:
        conversation.append({"role": "assistant", "content": bot_response})

    return bot_response, bot_response_is_tool_calls, completion_done


def on_llm_token_response(token, style="", prompt=""):
    function_handled = False
    for plugin in plugins.plugin_manager.plugins:
        if hasattr(plugin, "on_llm_token_response") and callable(getattr(plugin, "on_llm_token_response")):
            plugin_response = getattr(plugin, "on_llm_token_response")(token)
            function_handled = function_handled or plugin_response

    if not function_handled:
        if style or prompt:
            sys.stdout.write(f"{style}{prompt}{token}")
        else:
            sys.stdout.write(token)


def generate_tool_response(
    user_input,
    tools,
    selected_model,
    temperature=0.1,
    prompt_template=None,
    num_ctx=None,
    *,
    ctx:Context
) -> str:
    """Generate a response using Ollama that suggests function calls based on the user input."""

    # Create the system prompt with the provided tools
    rendered_tools = render_tools(tools)
    system_prompt = f"""You are an assistant that has access to the following set of tools.
Here are the names and descriptions for each tool:

{rendered_tools}

Given the user input, return your response as a JSON array of objects, each representing a different function call. Each object should have the following structure:
{{"function": {{
"name": A string representing the function's name.
"arguments": An object containing key-value pairs representing the arguments to be passed to the function. }}}}

If no tool is relevant to answer, simply return an empty array: [].
"""

    # Call the existing ask_ollama function
    tool_response = ask_ollama(
        system_prompt,
        user_input,
        selected_model,
        temperature,
        prompt_template,
        no_bot_prompt=True,
        stream_active=False,
        num_ctx=num_ctx,
        ctx = ctx
    )

    if ctx.verbose:
        plugins.on_print(f"Tool response: {tool_response}", Fore.WHITE + Style.DIM)

    # The response should be in JSON format already if the function is correct.
    return utils.extract_json(tool_response,  ctx=ctx)



def on_llm_thinking_token_response(token, style="", prompt=""):
    function_handled = False
    for plugin in plugins.plugin_manager.plugins:
        if hasattr(plugin, "on_llm_thinking_token_response") and callable(getattr(plugin, "on_llm_thinking_token_response")):
            plugin_response = getattr(plugin, "on_llm_thinking_token_response")(token)
            function_handled = function_handled or plugin_response

    if not function_handled:
        if style or prompt:
            sys.stdout.write(f"{style}{prompt}{token}")
        else:
            sys.stdout.write(token)

def find_latest_user_message(conversation):
    # Iterate through the conversation list in reverse order
    for message in reversed(conversation):
        if message["role"] == "user":
            return message["content"]
    return None  # If no user message is found


def select_ollama_model_if_available(model_name, *,  ctx:Context):
    """
    Checks if the specified model is in the available ollama models.

    :return: model_name if the model si available, None otherwise.
    """

    if not model_name:
        return None

    try:
        models = ollama.list()["models"]
    except Exception as e:
        plugins.on_print(f"Ollama API is not running: {e}", Fore.RED)
        return None

    for model in models:
        if model["model"] == model_name:
            if ctx.verbose:
                plugins.on_print(f"Selected model: {model_name}", Fore.WHITE + Style.DIM)
            return model_name

    plugins.on_print(f"Model {model_name} not found.", Fore.RED)
    return None


def render_tools(tools):
    """Convert tools into a string format suitable for the system prompt."""
    tool_descriptions = []
    for tool in tools:
        tool_info = f"Tool name: {tool['function']['name']}\nDescription: {tool['function']['description']}\n"
        parameters = json.dumps(tool['function']['parameters'], indent=4)
        tool_info += f"Parameters:\n{parameters}\n"
        tool_descriptions.append(tool_info)
    return "\n".join(tool_descriptions)

