import os
import pymupdf as fitz


def extract_text_from_pdf(uploaded_file):
    """
    Extract text from an uploaded PDF (supports both file paths and file-like objects).

    Returns:
        list: Each item contains PDF name, page number and page text.
    """
    if isinstance(uploaded_file, str):
        file_name = os.path.basename(uploaded_file)
        document = fitz.open(uploaded_file)
    else:
        file_name = getattr(uploaded_file, "name", "document.pdf")
        pdf_bytes = uploaded_file.read()
        document = fitz.open(stream=pdf_bytes, filetype="pdf")

    pages = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text("text")

        if text.strip():
            pages.append({
                "pdf_name": file_name,
                "page_number": page_number,
                "text": text.strip()
            })

    document.close()

    return pages