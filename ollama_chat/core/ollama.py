from ollama_chat.core import on_print
from ollama_chat.core import print_spinning_wheel


def ask_ollama(system_prompt, user_input, selected_model, temperature=0.1, prompt_template=None, tools=[], no_bot_prompt=False, stream_active=True, num_ctx=None, use_think_mode=False):
    conversation = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_input}]
    return ask_ollama_with_conversation(conversation, selected_model, temperature, prompt_template, tools, no_bot_prompt, stream_active, num_ctx=num_ctx, use_think_mode=use_think_mode)

def ask_ollama_with_conversation(conversation, model, temperature=0.1, prompt_template=None, tools=[], no_bot_prompt=False, stream_active=True, prompt="Bot", prompt_color=None, num_ctx=None, use_think_mode=False):
    global no_system_role
    global syntax_highlighting
    global interactive_mode
    global verbose_mode
    global plugins
    global alternate_model
    global use_openai
    global use_azure_openai
    global think_mode_on

    # Some models do not support the "system" role, merge the system message with the first user message
    if no_system_role and len(conversation) > 1 and conversation[0]["role"] == "system" and not conversation[0]["content"] is None and not conversation[1]["content"] is None:
        conversation[1]["content"] = conversation[0]["content"] + "\n" + conversation[1]["content"]
        conversation = conversation[1:]

    model_is_an_ollama_model = is_model_an_ollama_model(model)

    if (use_openai or use_azure_openai) and not model_is_an_ollama_model:
        if verbose_mode:
            on_print("Using OpenAI API for conversation generation.", Fore.WHITE + Style.DIM)

    if not syntax_highlighting:
        if interactive_mode and not no_bot_prompt:
            if prompt_color:
                on_prompt(f"{prompt}: ", prompt_color)
            else:
                on_prompt(f"{prompt}: ", Style.RESET_ALL)
        else:
            if prompt_color:
                on_stdout_write("", prompt_color)
            else:
                on_stdout_write("", Style.RESET_ALL)
        on_stdout_flush()

    model_support_tools = True

    if (use_openai or use_azure_openai) and not model_is_an_ollama_model:
        completion_done = False

        while not completion_done:
            bot_response, bot_response_is_tool_calls, completion_done = ask_openai_with_conversation(conversation, model, temperature, prompt_template, stream_active, tools)
            if bot_response and bot_response_is_tool_calls:
                # Convert bot_response list of objects to a list of dict
                bot_response = [json.loads(json.dumps(obj, default=lambda o: vars(o))) for obj in bot_response]

                if verbose_mode:
                    on_print(f"Bot response: {bot_response}", Fore.WHITE + Style.DIM)

                bot_response = handle_tool_response(bot_response, model_support_tools, conversation, model, temperature, prompt_template, tools, stream_active, num_ctx=num_ctx)

                # Consider completion done
                completion_done = True
        if not bot_response is None:
            return bot_response.strip()
        else:
            return None

    bot_response = ""
    bot_thinking_response = ""
    bot_response_is_tool_calls = False
    ollama_options = {"temperature": temperature}
    if num_ctx:
        ollama_options["num_ctx"] = num_ctx

    think = use_think_mode or think_mode_on
    
    if verbose_mode and think:
        on_print("Thinking...", Fore.WHITE + Style.DIM)

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
            tool_response = generate_tool_response(find_latest_user_message(conversation), tools, model, temperature, prompt_template, num_ctx=num_ctx)
            
            if not tool_response is None and len(tool_response) > 0:
                bot_response = tool_response
                bot_response_is_tool_calls = True
                model_support_tools = False
            else:
                return ""
        else:
            on_print(f"An error occurred during the conversation: {e}", Fore.RED)
            return ""

    if not bot_response_is_tool_calls:
        try:
            if stream_active and len(tools) == 0:
                if alternate_model:
                    on_print(f"Response from model: {model}\n")
                chunk_count = 0
                for chunk in stream:
                    continue_response_generation = True
                    for plugin in plugins:
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
                    
                    if syntax_highlighting and interactive_mode:
                        print_spinning_wheel(chunk_count)
                    else:
                        if think and len(thinking_delta) > 0:
                            on_llm_thinking_token_response(thinking_delta, Fore.WHITE + Style.DIM)
                        else:
                            on_llm_token_response(delta, Fore.WHITE + Style.NORMAL)
                        on_stdout_flush()

                on_llm_token_response("\n")
                on_stdout_flush()
            else:
                tool_calls = stream['message'].get('tool_calls', [])
                if tool_calls is None:
                    tool_calls = []

                if len(tool_calls) > 0:
                    conversation.append(stream['message'])

                    if verbose_mode:
                        on_print(f"Tool calls: {tool_calls}", Fore.WHITE + Style.DIM)
                    bot_response = tool_calls
                    bot_response_is_tool_calls = True
                else:
                    if think:
                        bot_thinking_response = stream['message'].get('thinking', '')
                    bot_response = stream['message']['content']
        except KeyboardInterrupt:
            stream.close()
        except ollama.ResponseError as e:
            on_print(f"An error occurred during the conversation: {e}", Fore.RED)
            return ""

    # Check if the bot response is a tool call object
    if not bot_response_is_tool_calls and bot_response and len(bot_response.strip()) > 0 and bot_response.strip()[0] == "{" and bot_response.strip()[-1] == "}":
        bot_response = [extract_json(bot_response.strip())]
        bot_response_is_tool_calls = True

    # Check if the bot response is a list of tool calls
    if not bot_response_is_tool_calls and bot_response and len(bot_response.strip()) > 0 and bot_response.strip()[0] == "[" and bot_response.strip()[-1] == "]":
        bot_response = extract_json(bot_response.strip())
        bot_response_is_tool_calls = True

    # Check if the bot response starts with <tool_call>
    if not bot_response_is_tool_calls and bot_response and len(bot_response.strip()) > 0 and bot_response.startswith("<tool_call>"):
        bot_response = extract_json(bot_response.strip())
        bot_response_is_tool_calls = True

    if bot_response and bot_response_is_tool_calls:
        bot_response = handle_tool_response(bot_response, model_support_tools, conversation, model, temperature, prompt_template, tools, stream_active, num_ctx=num_ctx)

    if isinstance(bot_response, str):
        return bot_response.strip()
    else:
        return None

def is_model_an_ollama_model(model_name):
    global ollama

    try:
        models = ollama.list()["models"]
    except:
        return False

    for model in models:
        if model["model"] == model_name:
            return True

    return False

def on_prompt(prompt, style=""):
    function_handled = False
    for plugin in plugins:
        if hasattr(plugin, "on_prompt") and callable(getattr(plugin, "on_prompt")):
            plugin_response = getattr(plugin, "on_prompt")(prompt)
            function_handled = function_handled or plugin_response

    if not function_handled:
        if style:
            sys.stdout.write(f"{style}{prompt}")
        else:
            sys.stdout.write(prompt)

