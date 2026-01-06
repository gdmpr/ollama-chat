import os
import base64
import getpass
from urllib.parse import urljoin, urlparse
from colorama import Fore

import requests
from bs4 import BeautifulSoup

from ollama_chat.core import plugins
from ollama_chat.core.extract_text import extract_text_from_html

class SimpleWebScraper:
    def __init__(self, base_url, output_dir="downloaded_site", file_types=None, restrict_to_base=True, convert_to_markdown=False, verbose=False):
        self.base_url = base_url.rstrip('/')
        self.output_dir = output_dir
        self.file_types = file_types if file_types else ["html", "jpg", "jpeg", "png", "gif", "css", "js"]
        self.restrict_to_base = restrict_to_base
        self.convert_to_markdown = convert_to_markdown
        self.visited = set()
        self.verbose = verbose
        self.username = None
        self.password = None

    def scrape(self, url=None, depth=0, max_depth=50):
        if url is None:
            url = self.base_url

        # Prevent deep recursion
        if depth > max_depth and self.verbose:
            plugins.on_print(f"Max depth reached for {url}")
            return

        # Normalize the URL to avoid duplicates
        url = self._normalize_url(url)

        # Avoid revisiting URLs
        if url in self.visited:
            return
        self.visited.add(url)

        if self.verbose:
            plugins.on_print(f"Scraping: {url}")
        response = self._fetch(url)
        if not response:
            return

        content_type = response.headers.get("Content-Type", "")
        if "text/html" in content_type or not self._has_extension(url):
            if self.convert_to_markdown:
                self._save_markdown(url, response.text)
            else:
                self._save_html(url, response.text)
            self._parse_and_scrape_links(response.text, url, depth + 1)
        else:
            if self._is_allowed_file_type(url):
                self._save_file(url, response.content)

    def _fetch(self, url):
        headers = {}
        if self.username and self.password:
            credentials = f"{self.username}:{self.password}"
            encoded_credentials = base64.b64encode(credentials.encode('utf-8')).decode('utf-8')
            headers['Authorization'] = f"Basic {encoded_credentials}"

        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 401:
                plugins.on_print(f"Unauthorized access to {url}. Please enter your credentials.", Fore.RED)
                self.username = input("Username: ")
                self.password = getpass.getpass("Password: ")
                credentials = f"{self.username}:{self.password}"
                encoded_credentials = base64.b64encode(credentials.encode('utf-8')).decode('utf-8')
                headers['Authorization'] = f"Basic {encoded_credentials}"
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
            return response
        except requests.RequestException as e:
            plugins.on_print(f"Failed to fetch {url}: {e}", Fore.RED)
            return None

    def _save_html(self, url, html):
        local_path = self._get_local_path(url)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "w", encoding="utf-8") as file:
            file.write(html)

    def _save_markdown(self, url, html):
        local_path = self._get_local_path(url, markdown=True)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        markdown_content = extract_text_from_html(html)
        with open(local_path, "w", encoding="utf-8") as file:
            file.write(markdown_content)

    def _save_file(self, url, content):
        local_path = self._get_local_path(url)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as file:
            file.write(content)

    def _get_local_path(self, url, markdown=False):
        parsed_url = urlparse(url)
        local_path = os.path.join(self.output_dir, parsed_url.netloc, parsed_url.path.lstrip('/'))
        if local_path.endswith('/') or not os.path.splitext(parsed_url.path)[1]:
            local_path = os.path.join(local_path, "index.md" if markdown else "index.html")
        elif markdown:
            local_path = os.path.splitext(local_path)[0] + ".md"
        return local_path

    def _normalize_url(self, url):
        # Remove fragments and normalize trailing slashes
        parsed = urlparse(url)
        normalized = parsed._replace(fragment="").geturl()
        return normalized

    def _parse_and_scrape_links(self, html, base_url, depth):
        soup = BeautifulSoup(html, "html.parser")

        for tag, attr in [("a", "href"), ("img", "src"), ("link", "href"), ("script", "src")]:
            for element in soup.find_all(tag):
                link = element.get(attr)
                if link:
                    abs_link = urljoin(base_url, link)
                    abs_link = self._normalize_url(abs_link)
                    if self.restrict_to_base and not self._is_same_domain(abs_link):
                        continue
                    if not self._is_allowed_file_type(abs_link) and self._has_extension(abs_link):
                        continue
                    if abs_link not in self.visited:
                        self.scrape(abs_link, depth=depth)

    def _is_same_domain(self, url):
        base_domain = urlparse(self.base_url).netloc
        target_domain = urlparse(url).netloc
        return base_domain == target_domain

    def _is_allowed_file_type(self, url):
        path = urlparse(url).path
        file_extension = os.path.splitext(path)[1].lstrip('.').lower()
        return file_extension in self.file_types

    def _has_extension(self, url):
        path = urlparse(url).path
        return bool(os.path.splitext(path)[1])
