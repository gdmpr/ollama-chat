import chardet
import requests
from colorama import Fore, Style

from ollama_chat.core import utils
from ollama_chat.core.ollama import ask_ollama
from ollama_chat.core.extract_text import extract_text_from_pdf
from ollama_chat.core.extract_text import extract_text_from_html
from ollama_chat.core.context import Context


class SimpleWebCrawler:
    def __init__(
        self,
        urls,
        llm_enabled=False,
        system_prompt='',
        selected_model='',
        temperature=0.1,
        verbose=False,
        plugins=None,
        num_ctx=None
    ):

        if plugins is None:
            plugins = []

        self.urls = urls
        self.articles = []
        self.llm_enabled = llm_enabled
        self.system_prompt = system_prompt
        self.selected_model = selected_model
        self.temperature = temperature
        self.verbose = verbose
        self.plugins = plugins
        self.num_ctx = num_ctx

    def fetch_page(self, url):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Raise an exception for HTTP errors
            return response.content  # Return raw bytes instead of text for PDF support
        except requests.exceptions.RequestException as e:
            if self.verbose:
                utils.on_print(f"Error fetching URL {url}: {e}", Fore.RED)
            return None

    def ask_llm(self, content, user_input, *, ctx:Context):
        # Use the provided ask_ollama function to interact with the LLM
        user_input = content + "\n\n" + user_input
        return ask_ollama(system_prompt=self.system_prompt,
                            user_input=user_input,
                            selected_model=self.selected_model,
                            temperature=self.temperature,
                            prompt_template=None,
                            tools=[],
                            no_bot_prompt=True,
                            stream_active=self.verbose,
                            num_ctx=self.num_ctx,
                            ctx=ctx
                        )

    def decode_content(self, content):
        # Detect encoding
        detected_encoding = chardet.detect(content)['encoding']
        if self.verbose:
            utils.on_print(f"Detected encoding: {detected_encoding}", Fore.WHITE + Style.DIM)

        # Decode content
        try:
            return content.decode(detected_encoding)
        except (UnicodeDecodeError, TypeError):
            if self.verbose:
                utils.on_print(f"Error decoding content with {detected_encoding}, using ISO-8859-1 as fallback.", Fore.RED)
            return content.decode('ISO-8859-1')

    def crawl(self, task=None, *, ctx:Context):
        for url in self.urls:
            continue_response_generation = True
            for plugin in self.plugins:
                if hasattr(plugin, "stop_generation") and callable(getattr(plugin, "stop_generation")):
                    plugin_response = getattr(plugin, "stop_generation")()
                    if plugin_response:
                        continue_response_generation = False
                        break

            if not continue_response_generation:
                break

            if self.verbose:
                utils.on_print(f"Fetching URL: {url}", Fore.WHITE + Style.DIM)
            content = self.fetch_page(url)
            if content:
                # Check if the URL points to a PDF
                if url.lower().endswith('.pdf'):
                    if self.verbose:
                        utils.on_print(f"Extracting text from PDF: {url}", Fore.WHITE + Style.DIM)
                    extracted_text = extract_text_from_pdf(content)
                else:
                    if self.verbose:
                        utils.on_print(f"Extracting text from HTML: {url}", Fore.WHITE + Style.DIM)
                    decoded_content = self.decode_content(content)
                    extracted_text = extract_text_from_html(decoded_content)

                article = {'url': url, 'text': extracted_text}

                if self.llm_enabled and task:
                    if self.verbose:
                        utils.on_print(Fore.WHITE + Style.DIM + f"Using LLM to process the content. Task: {task}")
                    llm_result = self.ask_llm(content=extracted_text, user_input=task,  ctx=ctx)
                    article['llm_result'] = llm_result

                self.articles.append(article)

    def get_articles(self):
        return self.articles
