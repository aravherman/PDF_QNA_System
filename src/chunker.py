def create_chunks(pages, chunk_size=150, overlap=30):
    """
    Divide page text into overlapping chunks.

    Args:
        pages: List of extracted PDF pages
        chunk_size: Number of words in each chunk
        overlap: Number of overlapping words

    Returns:
        List of chunks with metadata
    """

    chunks = []
    chunk_id = 0

    for page in pages:

        words = page["text"].split()

        if not words:
            continue

        start = 0

        while start < len(words):

            end = start + chunk_size

            chunk_words = words[start:end]

            if not chunk_words:
                break

            chunk_text = " ".join(chunk_words)

            chunks.append({
                "chunk_id": chunk_id,
                "pdf_name": page["pdf_name"],
                "page_number": page["page_number"],
                "text": chunk_text
            })

            chunk_id += 1

            if end >= len(words):
                break

            start = end - overlap

    return chunks