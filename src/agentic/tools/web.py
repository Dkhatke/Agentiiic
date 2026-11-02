"""Web tools for the researcher: search and fetch. Network risk, untrusted output, SSRF-guarded, cached."""
from __future__ import annotations

import hashlib
import ipaddress
import re
import socket
import time
from html import unescape
from html.parser import HTMLParser
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import httpx

from ..config import STATE_DIR
from .builtin import _S, _clip, _obj
from .registry import Registry, Risk, Tool, ToolContext, ToolError

MAX_BYTES = 1_000_000
MAX_REDIRECTS = 3
CACHE_TTL = 24 * 3600
HEADERS = {"User-Agent": "agentic-research/0.1 (personal research agent)"}


def _resolve(host: str) -> list[str]:
    return [i[4][0] for i in socket.getaddrinfo(host, None)]


def check_url(url: str, allow: frozenset | set = frozenset()) -> None:
    """Allow only public http(s) hosts; block loopback, private and link-local targets."""
    try:
        u = urlparse(url)
        host = u.hostname
    except ValueError as e:
        raise ToolError(f"not a valid URL: {url[:80]!r}") from e
    if u.scheme not in {"http", "https"} or not host:
        raise ToolError("only http(s) URLs are allowed")
    if u.hostname in allow:  # explicitly trusted by the user's config (e.g. a local SearXNG)
        return
    try:
        addrs = _resolve(u.hostname)
    except (OSError, UnicodeError) as e:  # UnicodeError: hostnames the IDNA codec rejects
        raise ToolError(f"cannot resolve {u.hostname}") from e
    _require_public(u.hostname, addrs)


def _require_public(host: str, addrs: list[str]) -> None:
    for a in addrs:
        ip = ipaddress.ip_address(a)
        if not ip.is_global or ip.is_multicast:  # also covers 100.64/10 (carrier NAT) and other special-use ranges
            raise ToolError(f"blocked non-public address for {host}")


class PinnedTransport(httpx.BaseTransport):
    """Closes the DNS-rebinding gap: the hostname is resolved once, every address is checked, and the connection goes to that
    exact address (Host header and TLS name stay the original host). A second, different DNS answer is never consulted."""

    def __init__(self, inner: httpx.BaseTransport | None = None, allow: frozenset | set = frozenset()):
        self.inner, self.allow = inner or httpx.HTTPTransport(), allow

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if host not in self.allow:
            try:
                addrs = [host] if _is_ip(host) else _resolve(host)
            except (OSError, UnicodeError) as e:
                raise ToolError(f"cannot resolve {host}") from e
            _require_public(host, addrs)
            if not _is_ip(host):
                request.extensions = {**request.extensions, "sni_hostname": host}  # certificate is still checked against the hostname
                request.url = request.url.copy_with(host=addrs[0])
        return self.inner.handle_request(request)

    def close(self) -> None:
        self.inner.close()


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


class _Text(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "head"}

    def __init__(self):
        super().__init__()
        self.parts, self._skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        if tag in {"p", "div", "li", "br", "h1", "h2", "h3", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    p = _Text()
    p.feed(html)
    return re.sub(r"\n\s*\n+", "\n\n", re.sub(r"[ \t]+", " ", "".join(p.parts))).strip()


TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)


def page_text(html: str) -> str:
    """Readable text with the <title> on top (the title lives in <head>, which html_to_text skips)."""
    body = html_to_text(html)
    m = TITLE_RE.search(html)
    title = re.sub(r"\s+", " ", unescape(m.group(1))).strip() if m else ""
    return f"title: {title}\n\n{body}" if title and title not in body[:300] else body


def _cache_path(key: str):
    return STATE_DIR / "cache" / (hashlib.sha256(key.encode()).hexdigest() + ".txt")


def _http(ctx: ToolContext, allow: frozenset = frozenset()) -> httpx.Client:
    return ctx.http or httpx.Client(timeout=20, follow_redirects=False, headers=HEADERS,
                                    transport=PinnedTransport(allow=ctx.allow_hosts | allow))


def _get(ctx: ToolContext, url: str, params: dict | None = None, allow: frozenset = frozenset()) -> httpx.Response:
    client = _http(ctx, allow)
    for _ in range(MAX_REDIRECTS + 1):
        check_url(url, ctx.allow_hosts | allow)
        try:
            r = client.get(url, params=params)
        except httpx.HTTPError as e:
            raise ToolError(f"request failed: {type(e).__name__}") from e
        if r.is_redirect and "location" in r.headers:
            url, params = urljoin(url, r.headers["location"]), None
            continue
        if r.status_code >= 400:
            raise ToolError(f"HTTP {r.status_code}")
        return r
    raise ToolError("too many redirects")


PAGE_CHARS = 6000
CHUNK = 700


def _relevance(chunk: str, words: list[str]) -> float:
    """Keyword hits, discounted for menu-like text (many short lines) and boosted for passages with numbers."""
    low = chunk.lower()
    hits = sum(low.count(w) for w in words)
    lines = [l for l in chunk.splitlines() if l.strip()]
    avg_line = sum(len(l) for l in lines) / max(len(lines), 1)
    prose = min(1.0, avg_line / 60)
    numbers = len(re.findall(r"\d[\d,.]*", chunk))
    return hits * prose * (1.5 if numbers >= 3 else 1.0)


def select_text(text: str, focus: str = "", offset: int = 0) -> str:
    """Page window: the passages best matching `focus` (document order), or a plain slice from `offset`."""
    total = len(text)
    words = [w for w in re.findall(r"\w+", focus.lower()) if len(w) > 2]
    if words and not offset:
        chunks = [(i, text[i:i + CHUNK]) for i in range(0, total, CHUNK)]
        scored = sorted(chunks, key=lambda c: -_relevance(c[1], words))
        picked, used = [], 0
        for i, c in scored:
            if used + len(c) > PAGE_CHARS - 1000 or not any(w in c.lower() for w in words):
                continue
            picked.append((i, c))
            used += len(c)
        if picked:
            body = text[:800] + "\n[...]\n" + "\n[...]\n".join(c for _, c in sorted(picked))
            return f"{body}\n\n[page has {total} chars; showing the passages matching {focus!r}]"
    piece = text[offset:offset + PAGE_CHARS]
    more = f"\n\n[page has {total} chars; showing {offset}-{offset + len(piece)}. Use focus='...' or offset={offset + len(piece)} for more]" \
        if total > offset + len(piece) else ""
    return piece + more


def fetch_url(ctx: ToolContext, url: str, focus: str = "", offset: int = 0) -> str:
    try:
        offset = max(int(offset), 0)
    except (TypeError, ValueError):
        offset = 0
    cp = _cache_path(url)
    cached = urlparse(url).hostname not in ctx.allow_hosts
    if cached and cp.exists() and time.time() - cp.stat().st_mtime < CACHE_TTL:
        text = cp.read_text()
    else:
        r = _get(ctx, url)
        body = r.content[:MAX_BYTES].decode(r.encoding or "utf-8", errors="replace")
        ctype = r.headers.get("content-type", "")
        text = page_text(body) if "html" in ctype or body.lstrip().startswith("<") else body
        if cached:
            cp.parent.mkdir(parents=True, exist_ok=True)
            cp.write_text(text)
    ctx.fetched.add(url)
    return f"source: {url}\n{select_text(text, focus, offset)}"


RESULT_RE = re.compile(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.S)


def _searxng(ctx: ToolContext, query: str) -> str:
    base = ctx.searxng_url.rstrip("/")
    own = frozenset({urlparse(base).hostname})  # the user configured this host; it may be local, but only the search call may use it
    try:
        results = _get(ctx, f"{base}/search", {"q": query, "format": "json"}, own).json().get("results", [])
    except ValueError as e:
        raise ToolError("SearXNG did not return JSON (enable the json format in its settings.yml)") from e
    out = [f"- {x.get('title', '')}\n  {x.get('url', '')}" + (f"\n  {x['content'][:200]}" if x.get("content") else "")
           for x in results[:8]]
    return "\n".join(out) or "no results"


def web_search(ctx: ToolContext, query: str) -> str:
    if ctx.searxng_url:
        return _searxng(ctx, query)
    r = _get(ctx, "https://html.duckduckgo.com/html/", {"q": query})
    out = []
    for href, title in RESULT_RE.findall(r.text)[:8]:
        if "uddg=" in href:  # DuckDuckGo wraps the real URL
            href = unquote(parse_qs(urlparse(href).query).get("uddg", [href])[0])
        out.append(f"- {html_to_text(title)}\n  {href}")
    return "\n".join(out) or "no results (search page layout may have changed)"


def register(reg: Registry) -> None:
    reg.register(Tool("web_search", "Search the web; returns titles and URLs.",
                      _obj({"query": _S}, ["query"]), Risk.NETWORK, web_search, untrusted=True))
    reg.register(Tool("fetch_url", "Fetch a public web page as text. Long pages are windowed: pass focus (key words you are "
                      "looking for) to get the matching passages from the whole page, or offset to read further.",
                      _obj({"url": _S, "focus": _S, "offset": {"type": "integer"}}, ["url"]),
                      Risk.NETWORK, fetch_url, untrusted=True))
