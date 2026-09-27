"""Xu ly ten file, URL upload va validate noi dung tai lieu dau vao."""

import ipaddress
import re
import socket
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

from botai.core.config import MAX_UPLOAD_BYTES, MAX_URL_BYTES

__all__ = ["_ReadableHTMLParser", "_safe_filename", "_url_to_safe_filename", "_is_blocked_url", "_download_url_as_text", "_validate_upload_bytes"]

class _ReadableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.skip_stack: list[str] = []
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "svg", "nav", "footer"}:
            self.skip_stack.append(tag)
        if tag == "title":
            self._in_title = True
        if tag in {"p", "br", "div", "section", "article", "h1", "h2", "h3", "li", "tr"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self.skip_stack and self.skip_stack[-1] == tag:
            self.skip_stack.pop()
        if tag == "title":
            self._in_title = False
        if tag in {"p", "div", "section", "article", "h1", "h2", "h3", "li", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        text = re.sub(r"\s+", " ", data or "").strip()
        if not text or self.skip_stack:
            return
        if self._in_title:
            self.title = f"{self.title} {text}".strip()
            return
        self.parts.append(text)

    def readable_text(self) -> str:
        return re.sub(r"\n{3,}", "\n\n", "\n".join(self.parts)).strip()



def _safe_filename(filename: str) -> str:
    name = Path(filename or "document").name
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).stem).strip("._") or "document"
    suffix = Path(name).suffix.lower()
    return f"{stem}{suffix}"


def _url_to_safe_filename(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    host = re.sub(r"[^A-Za-z0-9._-]+", "_", parsed.netloc).strip("._") or "website"
    path = re.sub(r"[^A-Za-z0-9._-]+", "_", parsed.path.strip("/")).strip("._")
    stem = "_".join(part for part in (host, path) if part)[:120] or "website"
    return f"{stem}.txt"


def _is_blocked_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return True
    hostname = parsed.hostname.lower()
    if hostname == "localhost" or hostname.endswith(".local"):
        return True
    try:
        addresses = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return True
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast:
            return True
    return False


def _download_url_as_text(url: str) -> tuple[str, str]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "BotAI-Document-Ingest/1.0",
            "Accept": "text/html, text/plain;q=0.9, */*;q=0.5",
        },
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        content_type = response.headers.get("Content-Type", "")
        raw = response.read(MAX_URL_BYTES + 1)
    if len(raw) > MAX_URL_BYTES:
        raise RuntimeError(f"Noi dung URL vuot qua gioi han {MAX_URL_BYTES // (1024 * 1024)} MB.")
    charset_match = re.search(r"charset=([\w.-]+)", content_type, flags=re.I)
    charset = charset_match.group(1) if charset_match else "utf-8"
    html = raw.decode(charset, errors="ignore")
    if "html" in content_type.lower() or "<html" in html[:500].lower():
        parser = _ReadableHTMLParser()
        parser.feed(html)
        title = parser.title or urllib.parse.urlparse(url).netloc
        text = parser.readable_text()
    else:
        title = urllib.parse.urlparse(url).netloc
        text = re.sub(r"\s+", " ", html).strip()
    if len(text) < 80:
        raise RuntimeError("Khong trich xuat duoc noi dung du dai tu URL.")
    return title, text


def _validate_upload_bytes(filename: str, content: bytes) -> str | None:
    suffix = Path(filename).suffix.lower()
    if len(content) > MAX_UPLOAD_BYTES:
        return f"File vượt quá giới hạn {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
    if suffix == ".pdf" and not content.startswith(b"%PDF"):
        return "File PDF không đúng định dạng."
    if suffix in {".docx", ".xlsx", ".pptx"} and not content.startswith(b"PK"):
        return "File Office không đúng định dạng ZIP OOXML."
    if suffix == ".txt" and b"\x00" in content[:4096]:
        return "File TXT có dấu hiệu là file nhị phân."
    return None
