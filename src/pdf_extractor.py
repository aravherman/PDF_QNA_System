import fitz


def extract_text_from_pdf(uploaded_file):
    """
    Extract text from an uploaded PDF.

    Returns:
        list: Each item contains PDF name, page number and page text.
    """

    pdf_bytes = uploaded_file.read()

    document = fitz.open(stream=pdf_bytes, filetype="pdf")

    pages = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text("text")

        if text.strip():
            pages.append({
                "pdf_name": uploaded_file.name,
                "page_number": page_number,
                "text": text.strip()
            })

    document.close()

    return pages