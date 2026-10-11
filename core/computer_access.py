import base64
import html
import html.parser
import hashlib
import ipaddress
import json
import platform
import re
import shutil
import socket
import subprocess
import urllib.error
import urllib.request
import urllib.parse
from pathlib import Path

from core.persistence import write_json_atomically


READABLE_SUFFIXES = {
    ".c", ".cc", ".cpp", ".css", ".csv", ".go", ".h", ".hpp", ".html",
    ".java", ".js", ".json", ".jsx", ".md", ".py", ".rb", ".rs", ".rst",
    ".sh", ".sql", ".swift", ".tex", ".toml", ".ts", ".tsx", ".txt",
    ".xml", ".yaml", ".yml",
}
MAX_FILE_BYTES = 128 * 1024
MAX_DOCUMENTS = 50
MAX_TOTAL_TEXT_BYTES = 1024 * 1024
SENSITIVE_NAME = re.compile(
    r"(^|[._-])(secret|secrets|credential|credentials|password|passwd|token|"
    r"private|private[_-]?key|api[_-]?key|access[_-]?key|id_rsa|id_ed25519|"
    r"keychain|cookie|cookies)([._-]|$)",
    re.IGNORECASE,
)
ALLOWED_MAC_APPS = {
    "calculator": "Calculator",
    "calendar": "Calendar",
    "finder": "Finder",
    "notes": "Notes",
    "safari": "Safari",
    "system settings": "System Settings",
}
MAX_WEB_RESPONSE_BYTES = 1024 * 1024
MAX_WEB_TEXT_CHARS = 12000


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        return None


class _WebTextParser(html.parser.HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.suppressed = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript", "svg"):
            self.suppressed += 1
        elif tag in ("p", "div", "li", "br", "tr", "h1", "h2", "h3"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript", "svg") and self.suppressed:
            self.suppressed -= 1
        elif tag in ("p", "div", "li", "br", "tr", "h1", "h2", "h3"):
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.suppressed:
            text = " ".join(data.split())
            if text:
                self.parts.append(text)


class _SearchResultsParser(html.parser.HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results = []
        self.current_link = None
        self.current_text = []
        self.in_bing_result = False
        self.in_heading = False

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        classes = set(attributes.get("class", "").split())
        if tag == "li" and "b_algo" in classes:
            self.in_bing_result = True
        if tag == "h2" and self.in_bing_result:
            self.in_heading = True
        if tag == "a" and ("result__a" in classes or self.in_heading):
            self.current_link = attributes.get("href")
            self.current_text = []

    def handle_endtag(self, tag):
        if tag == "a" and self.current_link:
            self.results.append(
                (self.current_link, " ".join(" ".join(self.current_text).split()))
            )
            self.current_link = None
            self.current_text = []
        if tag == "h2":
            self.in_heading = False
        if tag == "li":
            self.in_bing_result = False

    def handle_data(self, data):
        if self.current_link:
            self.current_text.append(data)


class ComputerAccess:
    def __init__(self, root_directory=None, library_path=None):
        self.root = None
        if root_directory:
            root = Path(root_directory).expanduser().resolve(strict=True)
            if not root.is_dir():
                raise ValueError(f"LUCY_LEARN_DIR must be a directory: {root}")
            self.root = root

        self.library_path = Path(library_path).expanduser() if library_path else None
        self.documents = self._load_library()

    def _load_library(self):
        if self.library_path is None:
            return {}
        try:
            with self.library_path.open(encoding="utf-8") as file:
                saved = json.load(file)
        except FileNotFoundError:
            return {}
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError(
                "Learned-file library contains invalid JSON and was left unchanged: "
                f"{self.library_path}"
            ) from error
        except OSError as error:
            raise OSError(
                f"Could not read Lucy's learned-file library {self.library_path}: {error}"
            ) from error
        if not isinstance(saved, dict):
            raise ValueError(f"Learned-file library must be a JSON object: {self.library_path}")

        documents = {}
        total_bytes = 0
        for relative_path, entry in saved.items():
            if (
                not isinstance(relative_path, str)
                or not isinstance(entry, dict)
                or not isinstance(entry.get("text"), str)
            ):
                continue
            text = entry["text"]
            text_bytes = len(text.encode("utf-8"))
            if text_bytes > MAX_FILE_BYTES or len(documents) >= MAX_DOCUMENTS:
                continue
            if total_bytes + text_bytes > MAX_TOTAL_TEXT_BYTES:
                continue
            documents[relative_path] = {
                "text": text,
                "digest": str(entry.get("digest", "")),
            }
            total_bytes += text_bytes
        return documents

    def _save_library(self):
        if self.library_path is None:
            raise RuntimeError("The learned-file library is not configured.")
        write_json_atomically(self.library_path, self.documents)

    def _resolve_directory(self, relative_directory=""):
        if self.root is None:
            raise RuntimeError(
                "File learning is disabled. Set LUCY_LEARN_DIR to one folder you want Lucy to access."
            )
        relative_directory = relative_directory.replace("\\", "/").strip()
        if relative_directory in ("", "."):
            return self.root
        relative = Path(relative_directory)
        if relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts):
            raise ValueError("Use a relative folder path inside LUCY_LEARN_DIR.")

        candidate = self.root
        for part in relative.parts:
            if part.startswith("."):
                raise ValueError("Hidden folders are not accessible.")
            candidate = candidate / part
            if candidate.is_symlink():
                raise ValueError("Symbolic links are not accessible.")
        resolved = candidate.resolve(strict=True)
        try:
            resolved.relative_to(self.root)
        except ValueError as error:
            raise ValueError("That folder is outside LUCY_LEARN_DIR.") from error
        if not resolved.is_dir():
            raise ValueError("That path is not a folder.")
        return resolved

    def _resolve_file(self, relative_file):
        if self.root is None:
            raise RuntimeError(
                "File learning is disabled. Set LUCY_LEARN_DIR to one folder you want Lucy to access."
            )
        normalized = relative_file.replace("\\", "/").strip()
        relative = Path(normalized)
        if not normalized or relative.is_absolute() or any(
            part in ("", ".", "..") for part in relative.parts
        ):
            raise ValueError("Use a relative file path inside LUCY_LEARN_DIR.")
        if any(part.startswith(".") for part in relative.parts):
            raise ValueError("Hidden files and folders are not accessible.")
        if SENSITIVE_NAME.search(relative.name):
            raise ValueError("Files with names that suggest secrets or credentials are not accessible.")
        if relative.suffix.lower() not in READABLE_SUFFIXES:
            raise ValueError("That file type is not supported for learning.")

        candidate = self.root
        for part in relative.parts:
            candidate = candidate / part
            if candidate.is_symlink():
                raise ValueError("Symbolic links are not accessible.")
        resolved = candidate.resolve(strict=True)
        try:
            resolved.relative_to(self.root)
        except ValueError as error:
            raise ValueError("That file is outside LUCY_LEARN_DIR.") from error
        if not resolved.is_file():
            raise ValueError("That path is not a regular file.")
        if resolved.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f"Files must be {MAX_FILE_BYTES // 1024} KB or smaller.")
        return resolved

    def list_files(self, relative_directory=""):
        folder = self._resolve_directory(relative_directory)
        visible_files = []
        for path in folder.iterdir():
            if path.name.startswith(".") or path.is_symlink():
                continue
            if not path.is_file() or path.suffix.lower() not in READABLE_SUFFIXES:
                continue
            if SENSITIVE_NAME.search(path.name) or path.stat().st_size > MAX_FILE_BYTES:
                continue
            visible_files.append(path.relative_to(self.root).as_posix())
        return sorted(visible_files)[:200]

    def learn_file(self, relative_file):
        path = self._resolve_file(relative_file)
        with path.open("rb") as file:
            contents = file.read(MAX_FILE_BYTES + 1)
        if len(contents) > MAX_FILE_BYTES:
            raise ValueError(f"Files must be {MAX_FILE_BYTES // 1024} KB or smaller.")
        try:
            text = contents.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("Only UTF-8 text files can be learned.") from error
        if not text.strip():
            raise ValueError("That file is empty.")
        relative_path = path.relative_to(self.root).as_posix()
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        previous = self.documents.get(relative_path)
        if previous and previous["digest"] == digest:
            return f"I already know the current contents of {relative_path}."
        if relative_path not in self.documents and len(self.documents) >= MAX_DOCUMENTS:
            raise ValueError(f"Lucy can learn at most {MAX_DOCUMENTS} files.")
        stored_bytes = sum(
            len(document["text"].encode("utf-8"))
            for key, document in self.documents.items()
            if key != relative_path
        )
        if stored_bytes + len(text.encode("utf-8")) > MAX_TOTAL_TEXT_BYTES:
            raise ValueError("The local learned-file library is full.")

        previous_document = self.documents.get(relative_path)
        self.documents[relative_path] = {"text": text, "digest": digest}
        try:
            self._save_library()
        except OSError:
            if previous_document is None:
                del self.documents[relative_path]
            else:
                self.documents[relative_path] = previous_document
            raise
        return f"I learned {relative_path} ({len(text)} characters)."

    def forget_file(self, relative_file):
        normalized = relative_file.replace("\\", "/").strip()
        if normalized not in self.documents:
            return f"I have not learned {normalized!r}."
        document = self.documents.pop(normalized)
        try:
            self._save_library()
        except OSError:
            self.documents[normalized] = document
            raise
        return f"I forgot the saved contents of {normalized}."

    def search_learned(self, query, limit=3):
        terms = {
            word.lower()
            for word in re.findall(r"[a-z0-9]+", query)
            if len(word) > 1
        }
        if not terms:
            return []
        results = []
        for path, document in self.documents.items():
            text = document["text"]
            title_terms = {
                word.lower() for word in re.findall(r"[a-z0-9]+", path) if len(word) > 1
            }
            content_terms = {
                word.lower() for word in re.findall(r"[a-z0-9]+", text) if len(word) > 1
            }
            score = len(terms & title_terms) * 2 + len(terms & content_terms)
            if score:
                lowered = text.lower()
                offsets = [
                    lowered.find(term)
                    for term in terms
                    if lowered.find(term) >= 0
                ]
                start = max(0, min(offsets) - 250) if offsets else 0
                excerpt = text[start:start + 1600].strip()
                results.append((score, path, excerpt))
        results.sort(key=lambda item: item[0], reverse=True)
        return [(path, excerpt) for _, path, excerpt in results[:limit]]

    @staticmethod
    def _validate_public_url(url, resolve_host=True):
        parsed = urllib.parse.urlsplit(url.strip())
        if (
            parsed.scheme.lower() not in ("http", "https")
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise ValueError("Only public http and https URLs without embedded credentials are supported.")
        host = parsed.hostname.rstrip(".").lower()
        if host in ("localhost", "localhost.localdomain") or host.endswith(
            (".localhost", ".local", ".internal", ".home", ".lan", ".test")
        ):
            raise ValueError("Local and private-network hosts are not accessible.")
        try:
            port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
        except ValueError as error:
            raise ValueError("The URL contains an invalid port.") from error
        if port not in (80, 443):
            raise ValueError("Only standard public web ports 80 and 443 are supported.")

        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            address = None
        if address is not None:
            if not address.is_global:
                raise ValueError("Private and reserved IP addresses are not accessible.")
        elif resolve_host:
            try:
                addresses = {
                    ipaddress.ip_address(info[4][0].split("%", 1)[0])
                    for info in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
                }
            except (OSError, ValueError) as error:
                raise RuntimeError(f"Could not resolve public web host {host!r}: {error}") from error
            if not addresses or any(not address.is_global for address in addresses):
                raise ValueError("The URL host resolves to a private or reserved network.")
        return parsed

    @classmethod
    def _fetch_public_url(cls, url, accept):
        current_url = url
        opener = urllib.request.build_opener(_NoRedirectHandler)
        for _ in range(5):
            cls._validate_public_url(current_url)
            request = urllib.request.Request(
                current_url,
                headers={
                    "Accept": accept,
                    "User-Agent": "LucyAssistant/1.0 (public web lookup)",
                },
            )
            try:
                with opener.open(request, timeout=15) as response:
                    content_type = response.headers.get_content_type()
                    body = response.read(MAX_WEB_RESPONSE_BYTES + 1)
                    if len(body) > MAX_WEB_RESPONSE_BYTES:
                        raise ValueError("The web response exceeded Lucy's 1 MB size limit.")
                    return current_url, content_type, body
            except urllib.error.HTTPError as error:
                location = error.headers.get("Location")
                if error.code in (301, 302, 303, 307, 308) and location:
                    current_url = urllib.parse.urljoin(current_url, location)
                    error.close()
                    continue
                raise RuntimeError(
                    f"Website returned HTTP {error.code}: {error.reason}"
                ) from error
            except (urllib.error.URLError, TimeoutError, socket.timeout) as error:
                raise RuntimeError(f"Could not access the public website: {error}") from error
        raise RuntimeError("The website redirected too many times.")

    @classmethod
    def search_web(cls, query, limit=5):
        query = query.strip()
        if not query:
            return []
        search_url = "https://www.bing.com/search?q=" + urllib.parse.quote(query)
        _, content_type, body = cls._fetch_public_url(
            search_url, "text/html,application/xhtml+xml"
        )
        if content_type not in ("text/html", "application/xhtml+xml"):
            raise RuntimeError("The web search provider returned an unsupported response.")
        page = body.decode("utf-8", errors="replace")
        parser = _SearchResultsParser()
        parser.feed(page)
        results = []
        for href, title in parser.results:
            if not href or not title:
                continue
            absolute_url = urllib.parse.urljoin("https://www.bing.com", html.unescape(href))
            parsed = urllib.parse.urlsplit(absolute_url)
            if parsed.hostname == "www.bing.com" and parsed.path == "/ck/a":
                encoded_url = urllib.parse.parse_qs(parsed.query).get("u", [])
                if encoded_url and encoded_url[0].startswith("a1"):
                    try:
                        token = encoded_url[0][2:]
                        token += "=" * (-len(token) % 4)
                        absolute_url = base64.urlsafe_b64decode(token).decode("utf-8")
                    except (ValueError, UnicodeDecodeError):
                        continue
            try:
                cls._validate_public_url(absolute_url, resolve_host=False)
            except (RuntimeError, ValueError):
                continue
            results.append({"title": html.unescape(title), "url": absolute_url})
            if len(results) >= limit:
                break
        return results

    @classmethod
    def read_web_page(cls, url):
        final_url, content_type, body = cls._fetch_public_url(url, "text/html,text/plain")
        if content_type == "text/plain":
            text = body.decode("utf-8", errors="replace")
        elif content_type in ("text/html", "application/xhtml+xml"):
            parser = _WebTextParser()
            parser.feed(body.decode("utf-8", errors="replace"))
            text = "\n".join(part for part in parser.parts if part.strip())
        else:
            raise ValueError(
                f"Lucy only reads text and HTML pages, not {content_type!r}."
            )
        text = re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", text).strip()
        if not text:
            return f"The page at {final_url} did not contain readable text."
        return f"Page: {final_url}\n\n{text[:MAX_WEB_TEXT_CHARS]}"

    @staticmethod
    def open_url(url, approval_callback=None):
        ComputerAccess._validate_public_url(url, resolve_host=False)
        if platform.system() != "Darwin":
            raise RuntimeError("Opening URLs is currently supported only on macOS.")
        if approval_callback is None or not approval_callback(f"Open this URL in your browser?\n{url}"):
            return "I did not open the URL because approval was not granted."
        opener = shutil.which("open")
        if not opener:
            raise RuntimeError("macOS 'open' command was not found.")
        try:
            subprocess.run([opener, url], check=True, capture_output=True, text=True)
        except (OSError, subprocess.CalledProcessError) as error:
            raise RuntimeError(f"Could not open the URL: {error}") from error
        return f"Opened {url} in your browser."

    @staticmethod
    def open_app(app_name, approval_callback=None):
        normalized = app_name.strip().lower()
        app = ALLOWED_MAC_APPS.get(normalized)
        if app is None:
            return "That app is not on Lucy's approved app list."
        if platform.system() != "Darwin":
            raise RuntimeError("Opening apps is currently supported only on macOS.")
        if approval_callback is None or not approval_callback(
            f"Open the macOS app {app!r}?"
        ):
            return "I did not open the app because approval was not granted."
        opener = shutil.which("open")
        if not opener:
            raise RuntimeError("macOS 'open' command was not found.")
        try:
            subprocess.run(
                [opener, "-a", app],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise RuntimeError(f"Could not open {app}: {error}") from error
        return f"Opened {app}."
