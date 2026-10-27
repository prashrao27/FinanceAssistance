from collections import Counter
import re
from pathlib import Path

import pymupdf
from tokenizers import Tokenizer


_NEWS_PDF_NAME = "News_Systhesizer.pdf"
_NEWS_HEADING_FONT_SIZE = 11.0
_NEWS_CHUNK_SIZE = 512
_NEWS_CHUNK_OVERLAP = 64
_NEWS_TOKEN_RESERVE = 16
_NEWS_STOPWORDS = frozenset(
    """
    about above after again against all also although among an and any are as at
    be because been before being below between both but by can could did do does
    doing down during each few for from further had has have having he her here
    hers herself him himself his how i if in into is it its itself just me more
    may most my myself no nor not of off on once only or other our ours ourselves
    out over own same she should so some such than that the their theirs them
    themselves then there these they this those through to too under until up
    very was we were what when where which while who whom why will with would
    well you your yours yourself yourselves
    """.split()
)
_KEYWORD_PATTERN = re.compile(r"\b[A-Za-z][A-Za-z0-9'-]{2,}\b")
_ACRONYM_PATTERN = re.compile(r"\b[A-Z][A-Z0-9-]{1,}\b")

_CHAPTER_NUMBER = re.compile(r"^\d{1,3}$")
_SUBSECTION_HEADING = re.compile(
    r"^\s*(?P<number>\d+\.\d+(?:\.\d+)*)\s{2,}(?P<title>\S.{0,119})$"
)
_TOKENIZER_NAME = "BAAI/bge-large-en-v1.5"
_TOKEN_BOUNDARY_RESERVE = 16
PDF_CHUNK_METADATA_FIELDS = (
    "source_file",
    "document_title",
    "document_author",
    "Chapter",
    "ChapterTitle",
    "Subsection",
    "SubsectionTitle",
    "section_path",
    "section_title",
    "keywords",
    "page_start",
    "page_end",
    "dense_text",
    "sparse_text",
    "combined_text",
)


def _normalize_heading(text: str) -> str:
    return "".join(character.lower() for character in text if character.isalnum())


def _news_paragraphs(page: pymupdf.Page) -> list[tuple[str, float]]:
    paragraphs: list[tuple[str, float]] = []
    page_data = page.get_text("dict", sort=True)
    for block in page_data["blocks"]:
        lines = block.get("lines")
        if not lines:
            continue
        text = " ".join(
            "".join(span["text"] for span in line["spans"]).strip()
            for line in lines
        ).strip()
        if not text or text.startswith("BIS Annual Economic Report 2026"):
            continue
        font_size = max(
            span["size"] for line in lines for span in line["spans"]
        )
        paragraphs.append((text, font_size))
    return paragraphs


def _split_news_paragraphs(
    paragraphs: list[str],
    tokenizer: Tokenizer,
    token_limit: int,
    overlap_tokens: int,
) -> list[str]:
    separator_ids = tokenizer.encode("\n\n").ids
    chunks: list[str] = []
    current_ids: list[int] = []

    def save_current() -> None:
        if current_ids:
            text = tokenizer.decode(current_ids).strip()
            if text:
                chunks.append(text)

    for paragraph in paragraphs:
        paragraph_ids = tokenizer.encode(paragraph).ids
        if not paragraph_ids:
            continue

        if len(paragraph_ids) > token_limit:
            save_current()
            current_ids.clear()
            step = token_limit - overlap_tokens
            for start in range(0, len(paragraph_ids), step):
                piece = paragraph_ids[start : start + token_limit]
                if piece:
                    chunks.append(tokenizer.decode(piece).strip())
                if start + token_limit >= len(paragraph_ids):
                    break
            continue

        addition = separator_ids + paragraph_ids if current_ids else paragraph_ids
        if current_ids and len(current_ids) + len(addition) > token_limit:
            previous_ids = current_ids.copy()
            save_current()
            current_ids[:] = previous_ids[-overlap_tokens:] if overlap_tokens else []
            if current_ids:
                current_ids.extend(separator_ids)

        if current_ids and len(current_ids) + len(separator_ids) + len(paragraph_ids) > token_limit:
            save_current()
            current_ids.clear()
        if current_ids:
            current_ids.extend(separator_ids)
        current_ids.extend(paragraph_ids)

    save_current()
    return chunks


def _news_keywords(text: str, section_path: list[str]) -> list[str]:
    keywords: list[str] = []
    seen: set[str] = set()
    for title in section_path:
        for term in _KEYWORD_PATTERN.findall(title):
            key = term.casefold()
            if key not in seen:
                seen.add(key)
                keywords.append(term)

    for term in _ACRONYM_PATTERN.findall(text):
        key = term.casefold()
        if key not in seen:
            seen.add(key)
            keywords.append(term)
            if len(keywords) >= 32:
                return keywords

    frequencies = Counter(
        term.casefold()
        for term in _KEYWORD_PATTERN.findall(text)
        if term.casefold() not in _NEWS_STOPWORDS
    )
    for term, _ in frequencies.most_common(12):
        if term not in seen:
            seen.add(term)
            keywords.append(term)
            if len(keywords) >= 32:
                break
    return keywords


def _make_pdf_chunk(
    *,
    source_file: str,
    document_title: str,
    document_author: str,
    chapter: str,
    chapter_title: str,
    subsection: str,
    subsection_title: str,
    section_path: list[str],
    keywords: list[str],
    page_start: int,
    page_end: int,
    dense_body: str,
) -> dict[str, object]:
    section_header = " > ".join(section_path)
    dense_text = f"{section_header}\n{dense_body}".strip()
    sparse_text = (
        f"Document: {document_title}. Author: {document_author}. "
        f"Chapter: {chapter} {chapter_title}. "
        f"Subsection: {subsection} {subsection_title}. "
        f"Section: {section_header}. Keywords: {', '.join(keywords)}. "
        f"Content: {dense_body}"
    )
    return {
        "source_file": source_file,
        "document_title": document_title,
        "document_author": document_author,
        "Chapter": chapter,
        "ChapterTitle": chapter_title,
        "Subsection": subsection,
        "SubsectionTitle": subsection_title,
        "section_path": section_path.copy(),
        "section_title": section_path[-1] if section_path else "",
        "keywords": keywords,
        "page_start": page_start,
        "page_end": page_end,
        "dense_text": dense_text,
        "sparse_text": sparse_text,
        "combined_text": dense_text,
    }


def read_news_synthesizer_chunks(
    pdf_path: Path | None = None,
    chunk_size: int = _NEWS_CHUNK_SIZE,
    overlap_tokens: int = _NEWS_CHUNK_OVERLAP,
) -> list[dict[str, object]]:
    """Read the BIS report and prepare section-aware dense and sparse chunk text."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap_tokens < 0 or overlap_tokens >= chunk_size:
        raise ValueError("overlap_tokens must be between 0 and chunk_size - 1")

    if pdf_path is None:
        pdf_path = Path(__file__).resolve().parents[1] / "data" / _NEWS_PDF_NAME
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")

    tokenizer = Tokenizer.from_pretrained(_TOKENIZER_NAME)
    chunks: list[dict[str, object]] = []
    document_sections: list[tuple[int, str]] = []
    section_path: list[str] = []
    section_paragraphs: list[str] = []
    section_start_page = 1
    first_content_page = 1

    with pymupdf.open(pdf_path) as document:
        document_title = document.metadata.get("title") or pdf_path.stem
        document_author = document.metadata.get("author") or ""

        for level, title, page_number in document.get_toc():
            if level == 2 and _normalize_heading(title) not in {
                "contents",
                "conventions",
                "graphs",
                "countrycodes",
                "currencycodes",
            }:
                first_content_page = page_number
                break

        for level, title, _ in document.get_toc():
            if level >= 2:
                document_sections.append((level, title.strip()))
        headings_by_normalized_text = {
            _normalize_heading(title): (level, title)
            for level, title in document_sections
        }

        def flush_section(end_page: int) -> None:
            if not section_paragraphs:
                return

            header = " > ".join(section_path)
            header_token_count = len(tokenizer.encode(header).ids) if header else 0
            keywords = _news_keywords(" ".join(section_paragraphs), section_path)
            sparse_prefix = (
                f"Document: {document_title}. Author: {document_author}. "
                f"Chapter: {section_path[0] if section_path else ''}. "
                f"Subsection: {section_path[1] if len(section_path) > 1 else ''}. "
                f"Section: {header}. Keywords: {', '.join(keywords)}. Content:"
            )
            sparse_prefix_token_count = len(tokenizer.encode(sparse_prefix).ids)
            body_limit = min(
                chunk_size - header_token_count - _NEWS_TOKEN_RESERVE,
                chunk_size - sparse_prefix_token_count - _NEWS_TOKEN_RESERVE,
            )
            if body_limit <= overlap_tokens:
                raise ValueError(
                    f"Section metadata leaves insufficient room in a {chunk_size}-token chunk"
                )

            dense_chunks = _split_news_paragraphs(
                section_paragraphs,
                tokenizer,
                body_limit,
                overlap_tokens,
            )
            for dense_body in dense_chunks:
                top_level_heading = section_path[0] if section_path else ""
                chapter_match = re.match(
                    r"^\s*(?P<number>[IVX]+|\d+)\.?\s+(?P<title>.+)$",
                    top_level_heading,
                    re.IGNORECASE,
                )
                chapter = chapter_match.group("number") if chapter_match else ""
                chapter_title = (
                    chapter_match.group("title").strip()
                    if chapter_match
                    else top_level_heading
                )
                subsection_heading = (
                    section_path[1] if len(section_path) > 1 else ""
                )
                subsection_match = re.match(
                    r"^\s*(?P<number>\d+(?:\.\d+)*\.?)\s+(?P<title>.+)$",
                    subsection_heading,
                )
                subsection = (
                    subsection_match.group("number").rstrip(".")
                    if subsection_match
                    else ""
                )
                subsection_title = (
                    subsection_match.group("title").strip()
                    if subsection_match
                    else subsection_heading
                )
                chunks.append(
                    _make_pdf_chunk(
                        source_file=pdf_path.name,
                        document_title=document_title,
                        document_author=document_author,
                        chapter=chapter,
                        chapter_title=chapter_title,
                        subsection=subsection,
                        subsection_title=subsection_title,
                        section_path=section_path,
                        keywords=keywords,
                        page_start=section_start_page,
                        page_end=max(
                            section_start_page,
                            end_page - int(end_page > section_start_page),
                        ),
                        dense_body=dense_body,
                    )
                )
            section_paragraphs.clear()

        for page_index, page in enumerate(document):
            page_number = page_index + 1
            if page_number < first_content_page:
                continue

            blocks = _news_paragraphs(page)
            block_index = 0
            while block_index < len(blocks):
                text, font_size = blocks[block_index]
                heading_match: tuple[int, str] | None = None
                heading_block_count = 1

                if font_size >= _NEWS_HEADING_FONT_SIZE:
                    for candidate_count in range(1, 4):
                        candidate_blocks = blocks[
                            block_index : block_index + candidate_count
                        ]
                        if len(candidate_blocks) != candidate_count or any(
                            size < _NEWS_HEADING_FONT_SIZE
                            for _, size in candidate_blocks
                        ):
                            break
                        candidate = " ".join(
                            candidate_text for candidate_text, _ in candidate_blocks
                        )
                        heading_match = headings_by_normalized_text.get(
                            _normalize_heading(candidate)
                        )
                        if heading_match is not None:
                            heading_block_count = candidate_count
                            break

                if heading_match is not None:
                    flush_section(page_number)
                    level, title = heading_match
                    path_index = level - 2
                    section_path[:] = section_path[:path_index] + [title]
                    section_start_page = page_number
                    block_index += heading_block_count
                    continue

                if section_path and font_size < _NEWS_HEADING_FONT_SIZE:
                    section_paragraphs.append(text)
                    if not section_paragraphs[:-1]:
                        section_start_page = page_number
                block_index += 1

        flush_section(len(document))

    if not chunks:
        raise ValueError(
            f"No sectioned text was extracted from {pdf_path}; check its bookmarks and text layer"
        )
    return chunks


def embed_news_synthesizer_pdf(
    pdf_path: Path | None = None,
    chunk_size: int = _NEWS_CHUNK_SIZE,
    overlap_tokens: int = _NEWS_CHUNK_OVERLAP,
) -> list[dict[str, object]]:
    """Create dense vectors from semantic text and sparse vectors from enriched text."""
    from rag.embedding import make_dense_embedding, make_sparse_embedding

    chunks = read_news_synthesizer_chunks(pdf_path, chunk_size, overlap_tokens)
    dense_vectors = make_dense_embedding(
        [str(chunk["dense_text"]) for chunk in chunks]
    )
    sparse_vectors = make_sparse_embedding(
        [str(chunk["sparse_text"]) for chunk in chunks]
    )

    for chunk, dense_vector, sparse_vector in zip(
        chunks, dense_vectors, sparse_vectors, strict=True
    ):
        chunk["dense_embedding"] = dense_vector
        chunk["sparse_embedding"] = sparse_vector
    return chunks


def read_pdf_chunks(
    pdf_path: Path,
    chunk_size: int = 512,
    overlap_tokens: int = 64,
) -> list[dict[str, object]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap_tokens < 0 or overlap_tokens >= chunk_size:
        raise ValueError("overlap_tokens must be between 0 and chunk_size - 1")
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")

    tokenizer = Tokenizer.from_pretrained(_TOKENIZER_NAME)
    chunks: list[dict[str, object]] = []
    chapter_number = ""
    chapter_title = ""
    subsection_number = ""
    subsection_title = ""
    section_lines: list[str] = []
    section_start_page = 1
    section_end_page = 1
    has_subsections = False

    def flush_section() -> None:
        nonlocal section_start_page, section_end_page
        body = "\n".join(line for line in section_lines if line).strip()
        if not body:
            section_lines.clear()
            return

        header_parts = []
        if chapter_number:
            header_parts.append(f"Chapter {chapter_number}: {chapter_title}".rstrip())
        if subsection_number:
            header_parts.append(
                f"Subsection {subsection_number}: {subsection_title}".rstrip()
            )
        header = "\n".join(header_parts)
        header_ids = tokenizer.encode(header).ids if header else []
        body_ids = tokenizer.encode(body).ids
        body_window_size = chunk_size - len(header_ids) - _TOKEN_BOUNDARY_RESERVE
        if body_window_size <= 0:
            raise ValueError(
                f"PDF heading is too long for the {chunk_size}-token chunk limit"
            )
        if not body_ids:
            section_lines.clear()
            return

        section_path = []
        if chapter_number or chapter_title:
            section_path.append(
                f"Chapter {chapter_number}: {chapter_title}".strip().rstrip(":")
            )
        if subsection_number or subsection_title:
            section_path.append(
                f"Subsection {subsection_number}: {subsection_title}"
                .strip()
                .rstrip(":")
            )
        keywords = _news_keywords(body, section_path)
        sparse_prefix = (
            f"Document: {document_title}. Author: {document_author}. "
            f"Chapter: {chapter_number} {chapter_title}. "
            f"Subsection: {subsection_number} {subsection_title}. "
            f"Section: {' > '.join(section_path)}. "
            f"Keywords: {', '.join(keywords)}. Content:"
        )
        sparse_prefix_ids = tokenizer.encode(sparse_prefix).ids
        body_window_size = min(
            body_window_size,
            chunk_size - len(sparse_prefix_ids) - _TOKEN_BOUNDARY_RESERVE,
        )
        if body_window_size <= overlap_tokens:
            raise ValueError(
                f"Section metadata leaves insufficient room in a {chunk_size}-token chunk"
            )
        step = body_window_size - overlap_tokens

        for start in range(0, len(body_ids), step):
            token_slice = body_ids[start : start + body_window_size]
            if not token_slice:
                break
            text = tokenizer.decode(token_slice).strip()
            if text:
                chunks.append(
                    _make_pdf_chunk(
                        source_file=pdf_path.name,
                        document_title=document_title,
                        document_author=document_author,
                        chapter=chapter_number,
                        chapter_title=chapter_title,
                        subsection=subsection_number,
                        subsection_title=subsection_title,
                        section_path=section_path,
                        keywords=keywords,
                        page_start=section_start_page,
                        page_end=section_end_page,
                        dense_body=text,
                    )
                )
            if start + body_window_size >= len(body_ids):
                break
        section_lines.clear()

    with pymupdf.open(pdf_path) as document:
        document_title = document.metadata.get("title") or pdf_path.stem
        document_author = document.metadata.get("author") or ""
        for page_index, page in enumerate(document, start=1):
            lines = [
                line.strip()
                for line in page.get_text("text", sort=True).splitlines()
                if line.strip()
            ]
            if (
                len(lines) > 1
                and _CHAPTER_NUMBER.fullmatch(lines[0])
                and len(lines[1]) <= 120
            ):
                flush_section()
                chapter_number = lines[0]
                chapter_title = lines[1]
                subsection_number = ""
                subsection_title = ""
                section_start_page = page_index
                section_end_page = page_index
                section_lines.clear()
                continue

            for line in lines:
                subsection_match = _SUBSECTION_HEADING.match(line)
                if subsection_match:
                    flush_section()
                    subsection_number = subsection_match.group("number")
                    subsection_title = subsection_match.group("title").strip()
                    if not chapter_number:
                        chapter_number = subsection_number.split(".", maxsplit=1)[0]
                    section_start_page = page_index
                    section_end_page = page_index
                    has_subsections = True
                    continue

                if subsection_number:
                    section_lines.append(line)
                    section_end_page = page_index

    flush_section()
    if not has_subsections:
        raise ValueError(
            f"No numbered subsection headings (for example, 1.1) were found in {pdf_path}"
        )
    if not chunks:
        raise ValueError(f"No extractable text chunks were found in {pdf_path}")
    return chunks
