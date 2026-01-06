import re
from datetime import datetime
import json

from colorama import Fore, Style

from ollama_chat.core import utils
from ollama_chat.core import toolman
from ollama_chat.core.context import Context
from ollama_chat.core.utils import on_print
from ollama_chat.core.ollama import ask_ollama
from ollama_chat.core.utils import render_tools
from ollama_chat.core.query_vector_database import load_chroma_client


class Agent:
    # Static registry to store all agents
    agent_registry = {}

    def __init__(
        self,
        name,
        description,
        model,
        thinking_model=None,
        system_prompt=None,
        temperature=0.7,
        max_iterations=15,
        tools=None,
        verbose=False,
        num_ctx=None,
        thinking_model_reasoning_pattern=None
    ):
        """
        Initialize the Agent with a name, system prompt, tools, and other parameters.
        """
        self.name = name
        self.description = description
        self.system_prompt = system_prompt or "You are a helpful assistant capable of handling complex tasks."
        self.model = model
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.tools = tools or {}
        self.verbose = verbose
        self.num_ctx = num_ctx
        self.thinking_model = thinking_model or model
        self.thinking_model_reasoning_pattern = thinking_model_reasoning_pattern

        # State management variables for the TODO list
        self.todo_list = []
        self.completed_tasks = []
        self.task_results = {}

        # Register this agent in the global agent registry
        Agent.agent_registry[name] = self

    @staticmethod
    def get_agent(agent_name):
        """
        Retrieve an agent instance by name from the registry.
        """
        return Agent.agent_registry.get(agent_name)

    def query_llm(self, prompt, system_prompt=None, tools=[], model=None, *, ctx:Context):
        """
        Query the Ollama API with the given prompt and return the response.
        """
        if system_prompt is None:
            system_prompt = self.system_prompt

        if model is None:
            model = self.model

        if self.verbose:
            on_print(f"System prompt:\n{system_prompt}", Fore.WHITE + Style.DIM)
            on_print(f"User prompt:\n{prompt}", Fore.WHITE + Style.DIM)
            on_print(f"Model: {model}", Fore.WHITE + Style.DIM)

        llm_response = ask_ollama(
            system_prompt,
            prompt,
            model,
            temperature=self.temperature,
            no_bot_prompt=True,
            stream_active=False,
            tools=tools,
            num_ctx=self.num_ctx,
            ctx=ctx
        )

        if self.verbose:
            on_print(f"Response:\n{llm_response}", Fore.WHITE + Style.DIM)

        return llm_response

    def decompose_task(self, task,  *,  ctx:Context):
        """
        Decompose a task into subtasks using the system prompt for guidance.
        """
        tools_description = render_tools(self.tools)
        prompt = f"""Instructions: Break down the following task into smaller, manageable subtasks:
    {task}

    ## Available tools to assist with subtasks:
    {tools_description or 'No tools available.'}

    ## Constraints:
    - Maximum number of subtasks: {self.max_iterations}
    - Generate between 2 and {self.max_iterations} subtasks (do not exceed {self.max_iterations}).

    ## Output requirements:
    - Output each subtask on a single line.
    - Each subtask MUST begin with either a dash and a space ("- ") or a numbered prefix like "1. " (either format is acceptable). Do NOT use other bullet characters.
    - Do not include any additional text, explanations, headings, or conclusions. Output only the subtasks.
    - Do not include blank lines between subtasks. If a subtask naturally contains multiple sentences or lines, join them into one line by replacing internal newlines with a single space.
    - If an empty line would separate ideas, treat that empty line as the end of the current subtask and start the next subtask on a new line with the required prefix.
    - Avoid trailing colons or ambiguous punctuation that would break simple parsing.

    ## Output format example:
    - Define the goal
    - Research background information
    - Draft an outline
    - Write the first draft
    - Review and revise

    Produce the subtasks now:"""
        thinking_model_is_different = self.thinking_model != self.model
        response = self.query_llm(prompt, system_prompt=self.system_prompt, model=self.thinking_model,  ctx=ctx)

        if thinking_model_is_different:
            _, reasoning_response = split_reasoning_and_final_response(response, self.thinking_model_reasoning_pattern)
            if reasoning_response:
                reasoning = reasoning_response
            prompt = f"""Break down the following task into smaller, manageable subtasks:
{task}

## Available tools to assist with subtasks:
{tools_description or 'No tools available.'}

## Constraints:
- Maximum number of subtasks: {self.max_iterations}
- Generate between 2 and {self.max_iterations} subtasks (do not exceed {self.max_iterations}).

If I were to break down the task '{task}' into subtasks, I would do it as follows:
{reasoning}

You can follow a similar approach or provide a different response based on your own reasoning and understanding of the task.

## Output format:
Output each subtask on a new line, nothing more.
"""
            response = self.query_llm(prompt, system_prompt=self.system_prompt, model=self.model,  ctx=ctx)

        if self.verbose:
            on_print(f"Decomposed subtasks:\n{response}", Fore.WHITE + Style.DIM)
        subtasks = [subtask.strip() for subtask in response.split("\n") if subtask.strip()]
        contains_list = any(re.match(r'^\d+\.\s', subtask) or re.match(r'^[\*\-]\s', subtask) for subtask in subtasks)
        if contains_list:
            subtasks = [subtask for subtask in subtasks if re.match(r'^\d+\.\s', subtask) or re.match(r'^[\*\-]\s', subtask)]
        subtasks = [subtask for subtask in subtasks if not re.search(r':$', subtask) and not re.search(r'\*\*$', subtask)]
        subtasks = [re.sub(r'^\d+\.\s', '', subtask) for subtask in subtasks]
        subtasks = [re.sub(r'^[\*\-]\s', '', subtask) for subtask in subtasks]
        return subtasks

    def execute_subtask(self, main_task, subtask, *, ctx:Context):
        """
        Executes a subtask using available tools and full context from the agent's state.

        Parameters:
        - main_task: The main task being solved.
        - subtask: The subtask to be executed.

        Returns:
        - The result of the subtask execution.
        """
        # Build a richer context for the prompt using the agent's state
        completed_tasks_summary = "\n".join([f"- {t}: {self.task_results.get(t, 'Done.')}" for t in self.completed_tasks])
        remaining_tasks_summary = "\n".join([f"- {t}" for t in self.todo_list])

        prompt = f"""You are executing a plan to solve the main task: '{main_task}'.

## Completed Tasks & Results:
```markdown
{completed_tasks_summary or 'No tasks completed yet.'}
```

## Remaining Tasks (TODO List):
```markdown
{remaining_tasks_summary or 'This is the last task.'}
```

## Current Task:
Your current objective is to execute only this subtask: '{subtask}'

Based on the context of the completed tasks and the remaining plan, provide a response for the current task. Keep the response focused on this single subtask without additional introductions or conclusions.
"""

        if self.verbose:
            on_print(f"\nExecuting subtask: '{subtask}'", Fore.WHITE + Style.DIM)

        # Execute the subtask with available tools
        result = self.query_llm(prompt, system_prompt=self.system_prompt, tools=self.tools, ctx=ctx)

        return result

    def process_task(self, task, return_intermediate_results=False, *, ctx:Context):
        """
        Process the task by decomposing it into subtasks and executing each one,
        while maintaining a TODO list to track progress.
        """
        try:
            # Reset state for each new main task
            self.todo_list = self.decompose_task(task,  ctx=ctx)
            self.completed_tasks = []
            self.task_results = {}

            if self.verbose:
                on_print(f"Initial TODO list: {self.todo_list}", Fore.WHITE + Style.DIM)

            if not self.todo_list:
                return "No subtasks identified. Unable to process the task."

            # Use a while loop to process the dynamic TODO list
            iteration_count = 0
            while self.todo_list and iteration_count < self.max_iterations:
                # Get the next subtask to execute
                current_subtask = self.todo_list.pop(0)

                # Prevent re-doing work
                if current_subtask in self.completed_tasks:
                    if self.verbose:
                        on_print(f"Skipping already completed subtask: '{current_subtask}'", Fore.WHITE + Style.DIM)
                    continue

                # Execute the subtask using the new context-aware method
                result = self.execute_subtask(task, current_subtask,  ctx=ctx)

                if result:
                    # Mark as complete and store the result for future context
                    self.completed_tasks.append(current_subtask)
                    self.task_results[current_subtask] = result

                iteration_count += 1
                if self.verbose:
                    on_print(f"Finished iteration {iteration_count}. Remaining tasks: {len(self.todo_list)}", Fore.WHITE + Style.DIM)


            # Consolidate final response from all stored results
            final_response = "\n\n".join(self.task_results.values())

            if return_intermediate_results:
                # The concept of "intermediate versions" changes slightly.
                # Here we return just the final consolidated result in a list.
                return [final_response]

            return final_response

        except Exception as e:
            return f"Error during task processing: {str(e)}"



def split_reasoning_and_final_response(response, thinking_model_reasoning_pattern):
    """
    Split the reasoning and final response from the thinking model's response.
    """
    if not thinking_model_reasoning_pattern:
        return None, response

    reasoning = None
    final_response = response

    match = re.search(thinking_model_reasoning_pattern, response, re.DOTALL)
    if match and len(match.groups()) > 0:
        reasoning = match.group(1)
        final_response = response.replace(reasoning, "").strip()

    return reasoning, final_response


def create_new_agent_with_tools(system_prompt: str, tools: list[str], agent_name: str, agent_description: str, ctx:Context,  task: str = None):

    # Make sure tools are unique
    tools = list(set(tools))

    if ctx.verbose:
        utils.on_print("Agent Creation Parameters:", Fore.WHITE + Style.DIM)
        utils.on_print(f"System Prompt: {system_prompt}", Fore.WHITE + Style.DIM)
        utils.on_print(f"Tools: {tools}", Fore.WHITE + Style.DIM)
        utils.on_print(f"Agent Name: {agent_name}", Fore.WHITE + Style.DIM)
        utils.on_print(f"Agent Description: {agent_description}", Fore.WHITE + Style.DIM)
        if task:
            utils.on_print(f"Task: {task}", Fore.WHITE + Style.DIM)

    # Validate inputs
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError("System prompt must be a non-empty string.")
    if not isinstance(tools, list) or not all(isinstance(tool, str) for tool in tools):
        raise ValueError("Tools must be a list of strings.")
    if not isinstance(agent_name, str) or not agent_name.strip():
        raise ValueError("Agent name must be a non-empty string.")

    agent_tools = []
    available_tools = toolman.tool_manager.get_available_tools(ctx=ctx)
    for tool in tools:
        # If tool name starts with 'functions.', remove it
        if tool.startswith("functions."):
            tool = tool.split(".", 1)[1]

        for available_tool in available_tools:
            if tool.lower() == available_tool['function']['name'].lower() and tool.lower() != "instantiate_agent_with_tools_and_process_task" and tool.lower() != "create_new_agent_with_tools":
                agent_tools.append(available_tool)
                break

    if len(agent_tools) == 0:
        agent_tools.clear()

        # Some models are confused between collections and tools, so we need to check for this case
        load_chroma_client(ctx=ctx)

        # List existing collections
        collections = None
        if ctx.chroma_client:
            collections = ctx.chroma_client.list_collections()

        if collections:
            all_tools_are_collections = all(tool in [collection.name for collection in collections] for tool in tools)
            if all_tools_are_collections:
                # If tool query_vector_database is available, add it to the agent tools
                query_vector_database_tool = next((tool for tool in available_tools if tool['function']['name'] == 'query_vector_database'), None)
                if query_vector_database_tool:
                    agent_tools.append(query_vector_database_tool)

    # Always include today's date to the system prompt, for context
    system_prompt = f"{system_prompt}\nToday's date is {datetime.now().strftime('%A, %B %d, %Y %I:%M %p')}."

    # Instantiate the Agent with the provided parameters
    agent = Agent(
        name=agent_name,
        description=agent_description,
        model=ctx.current_model,
        thinking_model=ctx.thinking_model,
        system_prompt=system_prompt,
        temperature=0.7,
        tools=agent_tools,
        verbose=ctx.verbose,
        thinking_model_reasoning_pattern=ctx.thinking_model_reasoning_pattern
    )

    # If a task is provided, execute it synchronously and return the result
    if task and isinstance(task, str) and task.strip():
        try:
            result = agent.process_task(task, return_intermediate_results=True,  ctx=ctx)
            # Return the actual result from the agent's task processing
            return result if result else f"Agent '{agent_name}' completed the task but produced no output."
        except Exception as e:
            return f"Error during task processing by agent '{agent_name}': {e}"

    # If no task provided, just return a success message about agent creation
    return f"Agent '{agent_name}' has been successfully created with {len(agent_tools)} tool(s): {', '.join([tool['function']['name'] for tool in agent_tools]) if agent_tools else 'none'}. The agent is registered and ready to be used."

def instantiate_agent_with_tools_and_process_task(task: str, system_prompt: str, tools: list[str], agent_name: str, ctx:Context,  agent_description: str = None, process_task=True) -> str|Agent:
    """
    Instantiate an Agent with a given name, system prompt, a list of tools, and solve a given task.

    Parameters:
    - task (str): The task or problem that the agent will solve.
    - system_prompt (str): The system prompt to guide the agent's behavior and approach.
    - tools (list[str]): A list of tools (from a predefined set) that the agent can use.
    - agent_name (str): A unique name for the agent.
    - agent_description (str): A description of the agent's capabilities and purpose.
    - process_task (bool): Whether to process the task immediately after instantiation.

    Returns:
    - str: The final result after the agent processes the task.
    """
    if ctx.verbose:
        utils.on_print("Agent Instantiation Parameters:", Fore.WHITE + Style.DIM)
        utils.on_print(f"Task: {task}", Fore.WHITE + Style.DIM)
        utils.on_print(f"System Prompt: {system_prompt}", Fore.WHITE + Style.DIM)
        utils.on_print(f"Tools: {tools}", Fore.WHITE + Style.DIM)
        utils.on_print(f"Agent Name: {agent_name}", Fore.WHITE + Style.DIM)
        utils.on_print(f"Agent Description: {agent_description}", Fore.WHITE + Style.DIM)


    # If tools is a string, it's probably a JSON string, so parse it
    if isinstance(tools, str):
        try:
            tools = json.loads(tools)
        except json.JSONDecodeError:
            return "Error: Tools must be a list of strings."
    elif not isinstance(tools, list) or not all(isinstance(tool, str) for tool in tools):
        return "Error: Tools must be a list of strings."

    # Validate inputs
    if process_task and not isinstance(task, str) or not task.strip():
        return "Error: Task must be a non-empty string describing the problem or goal."
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        return "Error: System prompt must be a non-empty string."
    if not isinstance(agent_name, str) or not agent_name.strip():
        return "Error: Agent name must be a non-empty string."

    if not agent_description:
        agent_description = f"An AI assistant named {agent_name} with system role: '{system_prompt}'."

    # Make sure tools are unique
    tools = list(set(tools))

    agent_tools = []
    available_tools = toolman.tool_manager.get_available_tools(ctx=ctx)
    for tool in tools:
        # If tool name starts with 'functions.', remove it
        if tool.startswith("functions."):
            tool = tool.split(".", 1)[1]

        for available_tool in available_tools:
            if tool.lower() == available_tool['function']['name'].lower() and tool.lower() != "instantiate_agent_with_tools_and_process_task" and tool.lower() != "create_new_agent_with_tools":
                agent_tools.append(available_tool)
                break

    if len(agent_tools) == 0:
        agent_tools.clear()

        # Some models are confused between collections and tools, so we need to check for this case
        load_chroma_client(ctx=ctx)

        # List existing collections
        collections = None
        if ctx.chroma_client:
            collections = ctx.chroma_client.list_collections()

        if collections and len(collections) > 0 and len(tools) > 0:
            all_tools_are_collections = all(tool in [collection.name for collection in collections] for tool in tools)
            if all_tools_are_collections:
                # If tool query_vector_database is available, add it to the agent tools
                query_vector_database_tool = next((tool for tool in available_tools if tool['function']['name'] == 'query_vector_database'), None)
                if query_vector_database_tool:
                    agent_tools.append(query_vector_database_tool)

    # Always include today's date to the system prompt, for context
    system_prompt = f"{system_prompt}\nToday's date is {datetime.now().strftime('%A, %B %d, %Y %I:%M %p')}."

    # Instantiate the Agent with the provided parameters
    agent = Agent(
        name=agent_name,
        description=agent_description,
        model=ctx.current_model,
        thinking_model=ctx.thinking_model,
        system_prompt=system_prompt,
        temperature=0.7,
        tools=agent_tools,
        verbose=ctx.verbose,
        thinking_model_reasoning_pattern=ctx.thinking_model_reasoning_pattern
    )

    if process_task:
        # Process the task using the agent
        try:
            result = agent.process_task(task, return_intermediate_results=True,  ctx=ctx)
        except Exception as e:
            return f"Error during task processing: {e}"

        return result

    return agent
