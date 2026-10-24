"""Section-aware chunking of the Kubernetes docs into chunks.jsonl."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import frontmatter
from transformers import AutoTokenizer

DOCS_ROOT = Path("data/raw/k8s-website/content/en/docs")
OUT_PATH = Path("data/processed/chunks.jsonl")
BASE_URL = "https://kubernetes.io/docs"
MAX_TOKENS = 400   # tunable: you'll ablate this later
MIN_DOC_TOKENS = 40

tok = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
tok.model_max_length = 10**6  # silence the long-text warning; we only count


def n_tokens(text: str) -> int:
    return len(tok.encode(text, add_special_tokens=False))


# --- cleaning -------------------------------------------------------------
GLOSSARY = re.compile(r'\{\{<\s*glossary_tooltip[^>]*?text="([^"]+)"[^>]*>\}\}')
SHORTCODE = re.compile(r"\{\{[<%].*?[>%]\}\}", re.DOTALL)
COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


def clean(md: str) -> str:
    md = COMMENT.sub("", md)       # drop Hugo markers like <!-- overview -->
    md = GLOSSARY.sub(r"\1", md)   # keep the visible term, e.g. "Pod"
    return SHORTCODE.sub("", md)   # drop other Hugo shortcodes


# --- splitting ------------------------------------------------------------
HEADING = re.compile(r"^(#{1,4})\s+(.*?)\s*$")


def split_sections(md: str) -> list[tuple[list[str], str]]:
    """Return (heading_path, body) per section. Ignores '#' inside code fences."""
    sections, path, buf, in_fence = [], [], [], False

    def flush():
        body = "\n".join(buf).strip()
        if body:
            sections.append((list(path), body))
        buf.clear()

    for line in md.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        m = None if in_fence else HEADING.match(line)
        if m:
            flush()
            level = len(m.group(1))
            path[:] = path[: level - 1] + [m.group(2)]
        else:
            buf.append(line)
    flush()
    return sections


def split_blocks(body: str) -> list[str]:
    """Paragraph-level blocks; a code fence is never split here."""
    blocks, cur, in_fence = [], [], False
    for line in body.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        cur.append(line)
        if not in_fence and not line.strip():
            blocks.append("\n".join(cur).strip())
            cur = []
    if cur:
        blocks.append("\n".join(cur).strip())
    return [b for b in blocks if b]


def split_oversized(block: str, budget: int) -> list[str]:
    """Split a too-big block on line boundaries; re-fence code pieces."""
    lines = block.splitlines()
    open_fence = lines[0] if lines[0].lstrip().startswith("```") else None
    if open_fence:
        body = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
    else:
        body = lines

    pieces, cur, cur_t = [], [], 0
    for line in body:
        t = n_tokens(line) + 1
        if cur and cur_t + t > budget:
            pieces.append(cur)
            cur, cur_t = [], 0
        cur.append(line)
        cur_t += t
    if cur:
        pieces.append(cur)

    if open_fence:
        return ["\n".join([open_fence, *p, "```"]) for p in pieces]
    return ["\n".join(p) for p in pieces]


def pack(blocks: list[str], budget: int) -> list[str]:
    chunks, cur, cur_t = [], [], 0
    for b in blocks:
        t = n_tokens(b)
        if t > budget:                      # single block too big: split it
            if cur:
                chunks.append("\n\n".join(cur))
                cur, cur_t = [], 0
            chunks.extend(split_oversized(b, budget))
            continue
        if cur and cur_t + t > budget:
            chunks.append("\n\n".join(cur))
            cur, cur_t = [], 0
        cur.append(b)
        cur_t += t
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


# --- ids / metadata ---------------------------------------------------------
def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-") or "intro"


def doc_slug(path: Path) -> str:
    parts = list(path.relative_to(DOCS_ROOT).with_suffix("").parts)
    if parts[-1] in ("_index", "index"):
        parts = parts[:-1]
    return "/".join(parts)


@dataclass
class Chunk:
    chunk_id: str
    doc_slug: str
    title: str
    heading_path: list[str]
    source_url: str
    text: str
    n_tokens: int


def chunk_file(path: Path) -> list[Chunk]:
    post = frontmatter.load(path)
    if post.get("draft"):
        return []
    slug = doc_slug(path)
    title = str(post.get("title") or slug.split("/")[-1])
    md = clean(post.content)
    if n_tokens(md) < MIN_DOC_TOKENS:
        return []

    url = f"{BASE_URL}/{slug}/"
    counters: dict[str, int] = defaultdict(int)
    out: list[Chunk] = []
    for heading_path, body in split_sections(md):
        header = f"{title} | {' > '.join(heading_path)}"
        budget = max(MAX_TOKENS - n_tokens(header), 100)
        sec = slugify(heading_path[-1]) if heading_path else "intro"
        for text in pack(split_blocks(body), budget):
            n = counters[sec]
            counters[sec] += 1
            out.append(Chunk(
                chunk_id=f"{slug}#{sec}#{n}",
                doc_slug=slug,
                title=title,
                heading_path=heading_path,
                source_url=url,
                text=text,
                n_tokens=n_tokens(header) + n_tokens(text),
            ))
    return out


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(DOCS_ROOT.rglob("*.md"))
    chunks, failed = [], []
    for f in files:
        try:
            chunks.extend(chunk_file(f))
        except Exception as e:  # bad front matter etc.
            failed.append((f, e))

    with OUT_PATH.open("w") as fh:
        for c in chunks:
            fh.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

    toks = [c.n_tokens for c in chunks]
    print(f"files: {len(files)}  chunks: {len(chunks)}  failed: {len(failed)}")
    print(f"tokens/chunk  avg: {sum(toks)/len(toks):.0f}  max: {max(toks)}  "
          f">{MAX_TOKENS}: {sum(t > MAX_TOKENS for t in toks)}")
    assert len({c.chunk_id for c in chunks}) == len(chunks), "duplicate chunk IDs"
    for f, e in failed[:5]:
        print("FAILED:", f, e)


if __name__ == "__main__":
    main()