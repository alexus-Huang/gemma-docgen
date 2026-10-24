#!/usr/bin/env python3
"""gemma-docgen: multimodal developer CLI documentation assistant.

Scans a project directory (source, config files, architecture diagrams),
sends it to Google's Gemma 4 model via the Gemini API, and writes structured
Markdown documentation.

Reliability features: small default payloads, automatic payload shrinking on
server overload (HTTP 500/503/429), and automatic Gemma 4 model fallback.
"""
from __future__ import annotations

import io
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import typer
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from PIL import Image
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

# Load GEMINI_API_KEY from .env (next to this script first, then the current dir).
load_dotenv(Path(__file__).resolve().parent / ".env")
load_dotenv()

console = Console()
app = typer.Typer(
    add_completion=False,
    help="gemma-docgen: generate Markdown docs from code + diagrams using Gemma 4.",
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# Gemma 4 models on the Gemini API. The 26B-A4B mixture-of-experts model only
# activates ~4B parameters per token, so it is the fast default; the 31B dense
# model is the fallback (or pick it with --model for maximum quality).
GEMMA_MODELS = ["gemma-4-26b-a4b-it", "gemma-4-31b-it"]
# Optional extra fallbacks, only used with --with-gemini.
FAST_MODELS = ["gemini-flash-latest", "gemini-3.5-flash", "gemini-2.5-flash"]

# Smaller payloads = fewer 500/503 overload errors.
DEFAULT_MAX_CHARS = 24_000      # total text sent (~6k tokens)
DEFAULT_MAX_FILE_KB = 8         # per-file cap
DEFAULT_MAX_IMAGES = 2
IMAGE_MAX_DIM = 768             # px, longest side
DEFAULT_MAX_OUTPUT_TOKENS = 1000   # short output = fast response

IGNORE_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "venv", ".venv", "env", "__pycache__",
    ".idea", ".vscode", "dist", "build", "target", "out", ".next", ".nuxt",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", ".gradle", "coverage",
    "site-packages", "vendor", ".terraform",
}
ALLOWED_HIDDEN_DIRS = {".github"}

IGNORE_FILES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock",
    "pipfile.lock", "uv.lock", "cargo.lock", "composer.lock", "gemfile.lock",
    ".ds_store", "thumbs.db", "license", "license.md", "license.txt",
}
SENSITIVE_HINTS = ("secret", "credential", "id_rsa", "id_ed25519")
SENSITIVE_SUFFIXES = (".pem", ".key", ".pfx", ".p12", ".keystore", ".jks")
NOISE_SUFFIXES = (".min.js", ".min.css", ".map", ".lock")

TEXT_EXT = {
    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".java", ".kt",
    ".kts", ".go", ".rs", ".rb", ".php", ".c", ".h", ".cpp", ".hpp", ".cc", ".cs",
    ".swift", ".scala", ".sh", ".bash", ".zsh", ".ps1", ".bat", ".cmd", ".sql",
    ".html", ".css", ".scss", ".vue", ".svelte", ".md", ".rst", ".txt", ".json",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".xml", ".gradle",
    ".properties", ".tf", ".proto", ".graphql", ".mmd", ".puml", ".dot",
    ".dockerfile",
}
TEXT_NAMES = {
    "dockerfile", "makefile", "procfile", "jenkinsfile", "gemfile", "rakefile",
    "vagrantfile",
}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

PRIORITY_NAMES = {
    "readme.md", "pyproject.toml", "package.json", "requirements.txt", "setup.py",
    "setup.cfg", "cargo.toml", "go.mod", "pom.xml", "build.gradle", "dockerfile",
    "docker-compose.yml", "docker-compose.yaml", "makefile", ".env.example",
    "tsconfig.json", "composer.json", "gemfile",
}
ENTRY_STEMS = {"main", "app", "index", "cli", "server", "__main__", "manage", "run"}

SECRET_PATTERNS = [
    (re.compile(r"AIza[0-9A-Za-z_\-]{35}"), "[REDACTED_API_KEY]"),
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}"), "[REDACTED_API_KEY]"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}"), "[REDACTED_TOKEN]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[REDACTED_AWS_KEY]"),
    (
        re.compile(
            r"(?i)((?:api[_-]?key|secret|token|passwd|password)\s*[=:]\s*)(['\"]?)([^\s'\"]{8,})(\2)"
        ),
        r"\1\2[REDACTED]\4",
    ),
]


class DocType(str, Enum):
    readme = "readme"
    docs = "docs"


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #

@dataclass
class TextFile:
    rel: str
    content: str
    truncated: bool


@dataclass
class ImageFile:
    rel: str
    data: bytes
    mime: str
    width: int
    height: int


@dataclass
class ScanResult:
    texts: List[TextFile] = field(default_factory=list)
    images: List[ImageFile] = field(default_factory=list)
    all_paths: List[str] = field(default_factory=list)
    skipped: List[Tuple[str, str]] = field(default_factory=list)
    total_chars: int = 0


# --------------------------------------------------------------------------- #
# Scanner
# --------------------------------------------------------------------------- #

def redact_secrets(text: str) -> str:
    for pattern, repl in SECRET_PATTERNS:
        text = pattern.sub(repl, text)
    return text


def classify(path: Path) -> Optional[str]:
    """Return 'text', 'image' or None (ignore)."""
    name = path.name
    lower = name.lower()
    if lower in IGNORE_FILES or lower.endswith(NOISE_SUFFIXES):
        return None
    if any(h in lower for h in SENSITIVE_HINTS) or lower.endswith(SENSITIVE_SUFFIXES):
        return None
    if lower.startswith(".env"):
        # Only templates are safe; real .env files are never read.
        return "text" if lower.endswith((".example", ".sample", ".template")) else None
    ext = path.suffix.lower()
    if ext in IMAGE_EXT:
        return "image"
    if ext in TEXT_EXT or lower in TEXT_NAMES:
        return "text"
    return None


def read_text(path: Path, limit: int) -> Optional[Tuple[str, bool]]:
    try:
        with open(path, "rb") as fh:
            raw = fh.read(limit + 1)
    except OSError:
        return None
    if b"\x00" in raw[:2048]:
        return None  # binary
    truncated = len(raw) > limit
    text = raw[:limit].decode("utf-8", errors="replace")
    return redact_secrets(text), truncated


def load_image(path: Path, max_dim: int = IMAGE_MAX_DIM) -> Optional[ImageFile]:
    try:
        with Image.open(path) as im:
            if getattr(im, "is_animated", False):
                im.seek(0)
            im.load()
            frame = im.copy()
        if max(frame.size) > max_dim:
            frame.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
        rgba = frame.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        buf = io.BytesIO()
        background.save(buf, format="PNG", optimize=True)
        return ImageFile("", buf.getvalue(), "image/png", background.width, background.height)
    except Exception:
        return None


def file_priority(rel: str) -> int:
    p = Path(rel)
    name = p.name.lower()
    if name in PRIORITY_NAMES:
        return 0
    if p.stem.lower() in ENTRY_STEMS:
        return 1
    if p.suffix.lower() in {".md", ".rst", ".txt"}:
        return 3
    return 2


def scan_project(
    root: Path,
    output_path: Path,
    max_chars: int,
    max_file_bytes: int,
    max_images: int,
    include_images: bool,
) -> ScanResult:
    result = ScanResult()
    candidates: List[Tuple[int, int, int, Path, str, str]] = []

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames
            if d not in IGNORE_DIRS
            and not d.endswith(".egg-info")
            and (not d.startswith(".") or d in ALLOWED_HIDDEN_DIRS)
        )
        for fname in sorted(filenames):
            fpath = Path(dirpath) / fname
            try:
                if fpath.resolve() == output_path.resolve():
                    continue
            except OSError:
                continue
            rel = fpath.relative_to(root).as_posix()
            kind = classify(fpath)
            if kind is None:
                continue
            try:
                size = fpath.stat().st_size
            except OSError:
                continue
            result.all_paths.append(rel)
            depth = rel.count("/")
            candidates.append((file_priority(rel), depth, size, fpath, rel, kind))

    candidates.sort(key=lambda c: (c[0], c[1], c[2], c[4]))

    for _prio, _depth, size, fpath, rel, kind in candidates:
        if kind == "image":
            if not include_images:
                result.skipped.append((rel, "images disabled"))
                continue
            if len(result.images) >= max_images:
                result.skipped.append((rel, "image limit reached"))
                continue
            if size > 15 * 1024 * 1024:
                result.skipped.append((rel, "image too large"))
                continue
            img = load_image(fpath)
            if img is None:
                result.skipped.append((rel, "unreadable image"))
                continue
            img.rel = rel
            result.images.append(img)
            continue

        loaded = read_text(fpath, max_file_bytes)
        if loaded is None:
            result.skipped.append((rel, "binary/unreadable"))
            continue
        content, truncated = loaded
        if not content.strip():
            result.skipped.append((rel, "empty"))
            continue
        if result.total_chars + len(content) > max_chars:
            result.skipped.append((rel, "text budget reached"))
            continue
        result.texts.append(TextFile(rel, content, truncated))
        result.total_chars += len(content)

    return result


# --------------------------------------------------------------------------- #
# Prompt formatter
# --------------------------------------------------------------------------- #

GENERAL_RULES = """You are a senior technical writer and software architect.
Below is a snapshot of a software project: a file listing, the contents of its
source/config files, and possibly architecture diagrams or screenshots attached
as images. Write high-quality documentation for it in GitHub-flavored Markdown.

RULES:
- Ground every statement in the provided material. Do NOT invent features,
  commands, environment variables, URLs, badges or version numbers.
- Take install/run commands from manifests, scripts and entry points. If something
  is not evident, say so briefly instead of guessing.
- Study every attached image. Explain what each diagram shows (components, data
  flow, relationships) in an Architecture section, and embed it with
  ![short caption](relative/path) using the EXACT path given in its <image> tag.
- Secret values were redacted. Never output secrets; refer to env vars by name only.
- Some files may be truncated to keep the request small; document only what you can see.
- Use clear headings, short paragraphs, tables for options/config, and fenced
  code blocks with language tags.
- Return ONLY the Markdown document. No preface, no closing remarks, and do not
  wrap the whole answer in a code fence."""

README_STRUCTURE = """Produce a concise, welcoming README.md with these sections (omit any that
do not apply):
1. Title and one-line tagline
2. Overview (what it does and why)
3. Features (bullets)
4. Architecture (only if diagrams or clear structure exist)
5. Tech stack
6. Prerequisites and Installation
7. Configuration (environment variables from the provided files)
8. Usage (real commands/examples)
9. Project structure (short annotated tree)
10. Contributing and License (only if a license/contributing file exists)"""

DOCS_STRUCTURE = """Produce thorough developer documentation (DOCUMENTATION.md) with these sections
(omit any that do not apply):
1. Introduction and purpose
2. Architecture and design (components, data flow, diagram explanations)
3. Project structure (annotated tree)
4. Module-by-module reference (purpose, key functions/classes, inputs/outputs)
5. Configuration reference (table of env vars / settings / flags)
6. CLI or API reference (commands, options, endpoints, with examples)
7. Setup and development guide (install, run, test, build)
8. Extending the project
9. Troubleshooting and FAQ (only issues evident from the code)"""


def build_contents(
    project_name: str,
    scan: ScanResult,
    doc_type: DocType,
    scale: float = 1.0,
    max_chars: int = DEFAULT_MAX_CHARS,
    max_file_kb: int = DEFAULT_MAX_FILE_KB,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> types.Content:
    """Build the multimodal prompt. `scale` (0-1] shrinks the payload on retries."""
    structure = README_STRUCTURE if doc_type is DocType.readme else DOCS_STRUCTURE

    char_budget = max(2000, int(max_chars * scale))
    per_file = max(800, int(max_file_kb * 1024 * scale))
    listing_limit = max(100, int(300 * scale))
    image_limit = 0
    if scan.images:
        image_limit = max(1, int(round(len(scan.images) * scale)))

    selected: List[Tuple[str, str, bool]] = []
    used = 0
    for tf in scan.texts:
        content, truncated = tf.content, tf.truncated
        if len(content) > per_file:
            content, truncated = content[:per_file], True
        if used + len(content) > char_budget:
            remaining = char_budget - used
            if remaining < 1500:
                continue
            content, truncated = content[:remaining], True
        used += len(content)
        selected.append((tf.rel, content, truncated))

    listing = "\n".join(scan.all_paths[:listing_limit])
    if len(scan.all_paths) > listing_limit:
        listing += "\n... (" + str(len(scan.all_paths) - listing_limit) + " more files not shown)"

    word_budget = max(150, int(max_output_tokens * 0.5))
    length_rule = (
        "\n- LENGTH LIMIT: the entire document must stay under about "
        + str(word_budget)
        + " words so it finishes completely. Be concise: prioritise overview, setup, "
        "usage and architecture, and skip minor details."
    )
    chunks = [
        GENERAL_RULES + length_rule,
        "",
        structure,
        "",
        "PROJECT NAME: " + project_name,
        "",
        "FILE LISTING (relative to project root):",
        listing,
        "",
        "FILE CONTENTS:",
    ]
    for rel, content, truncated in selected:
        note = " truncated=\"true\"" if truncated else ""
        chunks.append("<file path=\"" + rel + "\"" + note + ">")
        chunks.append(content.rstrip("\n"))
        chunks.append("</file>")
    if image_limit:
        chunks.append("")
        chunks.append("The architecture diagrams / images follow as attachments.")

    parts: List[types.Part] = [types.Part.from_text(text="\n".join(chunks))]
    for img in scan.images[:image_limit]:
        parts.append(
            types.Part.from_text(
                text="<image path=\"" + img.rel + "\" size=\"" + str(img.width) + "x" + str(img.height) + "\">"
            )
        )
        parts.append(types.Part.from_bytes(data=img.data, mime_type=img.mime))
    parts.append(types.Part.from_text(text="Now write the Markdown documentation."))
    return types.Content(role="user", parts=parts)


# --------------------------------------------------------------------------- #
# Gemma 4 via the Gemini API (with shrinking + fallback)
# --------------------------------------------------------------------------- #

RETRYABLE = {429, 500, 502, 503, 504}


class EmptyResponseError(Exception):
    """The model answered with no text."""


def clean_markdown(text: str) -> str:
    text = text.strip()
    match = re.match(r"^```(?:markdown|md)?\s*\n(.*)\n```$", text, flags=re.DOTALL)
    if match:
        text = match.group(1).strip()
    return text + "\n"


def build_attempt_plan(models: List[str]) -> List[Tuple[str, float]]:
    """Ordered (model, payload_scale) attempts.

    First model: full payload, then 50%. Fallback models start at 50% (a server
    that just overloaded us is happier with a smaller request), then 30%.
    """
    plan: List[Tuple[str, float]] = []
    for index, model in enumerate(models):
        scales = (1.0, 0.5) if index == 0 else (0.5, 0.3)
        for scale in scales:
            plan.append((model, scale))
    return plan


def generate_docs(
    client: genai.Client,
    models: List[str],
    make_contents: Callable[[float], types.Content],
    max_output_tokens: int,
    think: bool = False,
    on_progress: Optional[Callable[[int], None]] = None,
) -> Tuple[str, str, float, bool]:
    """Return (markdown, model_used, scale_used, was_truncated).

    Streams the response (so progress is visible and stalls are detected), turns
    Gemma's thinking mode off for speed unless `think` is set, shrinks the payload
    on overload errors, and falls back to the next model.
    """
    plan = build_attempt_plan(models)
    dead_models: set = set()
    thinking_unsupported: set = set()
    warned = {"sdk": False}
    last_error: Optional[Exception] = None

    def make_config(model: str) -> types.GenerateContentConfig:
        kwargs = {"temperature": 0.3, "max_output_tokens": max_output_tokens}
        if model not in thinking_unsupported:
            try:
                if model.startswith("gemma"):
                    # "minimal" disables Gemma 4's thinking pass: much faster.
                    level = "high" if think else "minimal"
                    kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=level)
                elif not think:
                    kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
            except Exception:  # noqa: BLE001  (old SDK without thinking_level)
                thinking_unsupported.add(model)
                if not warned["sdk"]:
                    warned["sdk"] = True
                    console.print(
                        "[yellow]Your google-genai SDK cannot turn thinking off, so responses "
                        "will be slower. Fix: pip install -U google-genai[/yellow]"
                    )
        return types.GenerateContentConfig(**kwargs)

    def stream_once(model: str, scale: float) -> Tuple[str, object]:
        pieces: List[str] = []
        finish: object = None
        total = 0
        for chunk in client.models.generate_content_stream(
            model=model, contents=make_contents(scale), config=make_config(model)
        ):
            piece = chunk.text
            if piece:
                pieces.append(piece)
                total += len(piece)
                if on_progress:
                    on_progress(total)
            try:
                reason = chunk.candidates[0].finish_reason
                if reason:
                    finish = reason
            except Exception:  # noqa: BLE001
                pass
        return "".join(pieces), finish

    def call(model: str, scale: float) -> Tuple[str, object]:
        try:
            return stream_once(model, scale)
        except genai_errors.APIError as exc:
            if (
                getattr(exc, "code", None) == 400
                and model not in thinking_unsupported
                and "think" in str(exc).lower()
            ):
                thinking_unsupported.add(model)  # this model rejects the setting
                return stream_once(model, scale)
            raise

    for index, (model, scale) in enumerate(plan):
        if model in dead_models:
            continue
        try:
            text, finish = call(model, scale)
            text = text.strip()
            if not text:
                raise EmptyResponseError("The model returned an empty response.")
            truncated = "MAX_TOKENS" in str(finish)
            return clean_markdown(text), model, scale, truncated
        except (genai_errors.APIError, EmptyResponseError) as exc:
            last_error = exc
            code = getattr(exc, "code", None)

            if code == 404:
                dead_models.add(model)
                console.print("[yellow]Model '" + escape(model) + "' is not available.[/yellow]")
                continue
            if code is not None and code not in RETRYABLE:
                raise  # auth errors, bad requests, etc.: retrying will not help

            nxt = next(((m, s) for m, s in plan[index + 1:] if m not in dead_models), None)
            if nxt is None:
                continue
            reason = "HTTP " + str(code) if code else "empty response"
            delay = 20 if code == 429 else (2 if nxt[0] != model else 4)
            console.print(
                "[yellow]" + escape(model) + " failed (" + reason + "). Retrying in "
                + str(delay) + "s with " + escape(nxt[0]) + " at "
                + str(int(nxt[1] * 100)) + "% payload...[/yellow]"
            )
            time.sleep(delay)

    if last_error:
        raise last_error
    raise RuntimeError("No Gemma 4 model could be used.")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def show_scan_summary(scan: ScanResult) -> None:
    table = Table(title="Scan summary", show_lines=False)
    table.add_column("Type")
    table.add_column("Count", justify="right")
    table.add_column("Details")
    table.add_row("Text files", str(len(scan.texts)), "~" + f"{scan.total_chars // 4:,}" + " tokens")
    table.add_row("Images", str(len(scan.images)), ", ".join(escape(i.rel) for i in scan.images[:4]) or "-")
    table.add_row("Skipped", str(len(scan.skipped)), "budget/limits/unreadable")
    console.print(table)


@app.command()
def main(
    path: Path = typer.Argument(
        Path("."),
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        help="Project directory to document.",
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Output file (relative paths resolve inside the project). "
             "Default: DOCUMENTATION.md, or README.md with --type readme.",
    ),
    doc_type: DocType = typer.Option(DocType.docs, "--type", "-t", help="'docs' (in-depth) or 'readme' (concise)."),
    model: Optional[str] = typer.Option(
        None, "--model", "-m",
        help="Preferred model ID, tried first (or set $DOCGEN_MODEL). Default: "
             "gemma-4-26b-a4b-it (fast), falling back to gemma-4-31b-it.",
    ),
    think: bool = typer.Option(
        False, "--think", help="Enable Gemma's thinking mode (slower, deeper reasoning)."
    ),
    with_gemini: bool = typer.Option(
        False, "--with-gemini", help="Also allow Gemini Flash models as extra fallbacks."
    ),
    max_output_tokens: int = typer.Option(
        DEFAULT_MAX_OUTPUT_TOKENS, "--max-output-tokens",
        help="Cap on generated tokens. Raise (e.g. 4096) for longer docs.",
    ),
    max_chars: int = typer.Option(DEFAULT_MAX_CHARS, "--max-chars", help="Total text budget sent to the model."),
    max_file_kb: int = typer.Option(DEFAULT_MAX_FILE_KB, "--max-file-kb", help="Per-file size cap in KB."),
    max_images: int = typer.Option(DEFAULT_MAX_IMAGES, "--max-images", help="Maximum number of images to attach."),
    no_images: bool = typer.Option(False, "--no-images", help="Do not attach images/diagrams."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Scan and list files; do not call the API."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Overwrite the output file without asking."),
    preview: bool = typer.Option(False, "--preview", help="Render the generated Markdown in the terminal."),
) -> None:
    """Scan PATH and generate Markdown documentation with Gemma 4."""
    default_name = "README.md" if doc_type is DocType.readme else "DOCUMENTATION.md"
    out_path = output if output is not None else Path(default_name)
    if not out_path.is_absolute():
        out_path = path / out_path

    console.print(
        Panel.fit(
            "[bold cyan]gemma-docgen[/bold cyan]  -  multimodal docs powered by Gemma 4\n"
            "Project: " + escape(str(path)),
            border_style="cyan",
        )
    )

    with console.status("Scanning project..."):
        scan = scan_project(
            root=path,
            output_path=out_path,
            max_chars=max_chars,
            max_file_bytes=max_file_kb * 1024,
            max_images=max_images,
            include_images=not no_images,
        )

    if not scan.texts and not scan.images:
        console.print("[red]No documentable files found in that directory.[/red]")
        raise typer.Exit(code=1)

    show_scan_summary(scan)

    if dry_run:
        tree = Table(title="Files that would be sent", show_lines=False)
        tree.add_column("Kind")
        tree.add_column("Path")
        tree.add_column("Note")
        for tf in scan.texts:
            tree.add_row("text", escape(tf.rel), "truncated" if tf.truncated else "")
        for img in scan.images:
            tree.add_row("image", escape(img.rel), str(img.width) + "x" + str(img.height))
        console.print(tree)
        for rel, reason in scan.skipped[:30]:
            console.print("[dim]skipped " + escape(rel) + " (" + reason + ")[/dim]")
        console.print("[green]Dry run complete. No API call was made.[/green]")
        raise typer.Exit()

    api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key or api_key == "your_gemini_api_key_here":
        console.print(
            Panel(
                "GEMINI_API_KEY is missing.\n\n"
                "1. Get a free key at https://aistudio.google.com/apikey\n"
                "2. Copy .env.example to .env\n"
                "3. Paste your key: GEMINI_API_KEY=...",
                title="API key required",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)

    if out_path.exists() and not yes:
        typer.confirm("'" + str(out_path) + "' already exists. Overwrite?", abort=True)

    preferred = model or os.getenv("DOCGEN_MODEL") or os.getenv("GEMMA_MODEL")
    base = GEMMA_MODELS + (FAST_MODELS if with_gemini else [])
    models = ([preferred] if preferred else []) + [m for m in base if m != preferred]

    def make_contents(scale: float) -> types.Content:
        return build_contents(
            path.name, scan, doc_type, scale, max_chars, max_file_kb, max_output_tokens
        )

    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=90_000))

    try:
        started = time.perf_counter()
        with console.status("Waiting for the model...") as status:
            def on_progress(count: int) -> None:
                status.update("Writing docs... " + f"{count:,}" + " characters received")

            markdown, used_model, used_scale, truncated = generate_docs(
                client, models, make_contents, max_output_tokens, think, on_progress
            )
        elapsed = time.perf_counter() - started
    except genai_errors.APIError as exc:
        code = getattr(exc, "code", None)
        hint = ""
        if code in (400, 401, 403):
            hint = "\nCheck that your GEMINI_API_KEY is valid and has access to Gemma models."
        elif code in RETRYABLE:
            hint = (
                "\nAll models are overloaded right now. Wait a minute and retry, "
                "or try --no-images --max-chars 15000."
            )
        console.print(Panel("API error (HTTP " + str(code) + "): " + escape(str(exc)) + hint,
                            title="Generation failed", border_style="red"))
        raise typer.Exit(code=1)
    except Exception as exc:  # noqa: BLE001
        console.print(Panel(escape(str(exc)), title="Generation failed", border_style="red"))
        raise typer.Exit(code=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")

    console.print(
        Panel.fit(
            "[bold green]Documentation written[/bold green]\n"
            "File:    " + escape(str(out_path)) + "\n"
            "Model:   " + escape(used_model) + "\n"
            "Payload: " + str(int(used_scale * 100)) + "% of budget\n"
            "Time:    " + f"{elapsed:.1f}" + "s\n"
            "Size:    " + f"{len(markdown):,}" + " characters, " + str(markdown.count(chr(10))) + " lines",
            border_style="green",
        )
    )
    if truncated:
        console.print(
            "[yellow]Output hit the " + str(max_output_tokens) + "-token cap and may be cut off. "
            "Re-run with --max-output-tokens 4096 for the complete document.[/yellow]"
        )
    if preview:
        console.print(Markdown(markdown))


if __name__ == "__main__":
    app()