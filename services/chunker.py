import re


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences, handling common abbreviations."""
    # Negative lookbehind for common abbreviations
    abbrevs = r"(?<!Mr)(?<!Mrs)(?<!Dr)(?<!Prof)(?<!Inc)(?<!Ltd)(?<!Jr)(?<!Sr)(?<!vs)(?<!etc)(?<!e\.g)(?<!i\.e)"
    pattern = rf"{abbrevs}(?<=[.!?])\s+(?=[A-Z])"
    sentences = re.split(pattern, text)
    return [s.strip() for s in sentences if s.strip()]


def semantic_chunk(
    text: str,
    max_size: int = 20000,
    overlap_sentences: int = 3,
) -> list[str]:
    """Split text into chunks at paragraph/section boundaries with sentence-aware overlap."""
    if not text.strip():
        return []

    if len(text) <= max_size:
        return [text]

    # Split by section markers (--- Page X ---) and double newlines
    sections = re.split(r"\n{2,}", text)

    chunks: list[str] = []
    current_parts: list[str] = []
    current_size = 0

    for section in sections:
        section = section.strip()
        if not section:
            continue
        section_size = len(section)

        # If adding this section exceeds limit, finalize current chunk
        if current_size + section_size > max_size and current_parts:
            chunk_text = "\n\n".join(current_parts)
            chunks.append(chunk_text)

            # Compute overlap: last N sentences from finished chunk
            all_sentences = split_into_sentences(chunk_text)
            overlap = all_sentences[-overlap_sentences:] if len(all_sentences) >= overlap_sentences else all_sentences
            current_parts = [" ".join(overlap)] if overlap else []
            current_size = sum(len(p) for p in current_parts)

        # If a single section is larger than max_size, split by sentences
        if section_size > max_size:
            sentences = split_into_sentences(section)
            sub_parts: list[str] = []
            sub_size = 0

            for sent in sentences:
                sent_size = len(sent)
                if sub_size + sent_size > max_size and sub_parts:
                    current_parts.append(" ".join(sub_parts))
                    chunk_text = "\n\n".join(current_parts)
                    chunks.append(chunk_text)

                    overlap = sub_parts[-overlap_sentences:]
                    current_parts = [" ".join(overlap)] if overlap else []
                    current_size = sum(len(p) for p in current_parts)
                    sub_parts = list(overlap)
                    sub_size = sum(len(s) for s in sub_parts)

                sub_parts.append(sent)
                sub_size += sent_size

            if sub_parts:
                joined = " ".join(sub_parts)
                current_parts.append(joined)
                current_size += len(joined)
        else:
            current_parts.append(section)
            current_size += section_size

    # Don't forget the last chunk
    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return chunks
