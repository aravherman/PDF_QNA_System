import streamlit as st

from src.pdf_extractor import extract_text_from_pdf
from src.preprocessor import clean_text
from src.chunker import create_chunks
from src.embeddings import (
    load_embedding_model,
    create_embeddings,
    create_question_embedding
)
from src.retriever import (
    create_faiss_index,
    search_faiss
)
from src.qa_model import (
    load_qa_model,
    get_answer
)


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="PDF Question Answering System",
    page_icon="📚",
    layout="wide"
)


# --------------------------------------------------
# Initialize Session State
# --------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pdf_pages" not in st.session_state:
    st.session_state.pdf_pages = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

if "qa_model" not in st.session_state:
    st.session_state.qa_model = None


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("📚 PDF Question Answering System")

st.write(
    "Upload one or more PDF documents and ask questions "
    "about their content."
)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.header("Upload Documents")

uploaded_files = st.sidebar.file_uploader(
    "Choose PDF files",
    type=["pdf"],
    accept_multiple_files=True
)


# Clear Chat History
if st.sidebar.button("🗑️ Clear Chat History"):

    st.session_state.chat_history = []

    st.rerun()


# --------------------------------------------------
# PDF Processing
# --------------------------------------------------

if uploaded_files:

    st.success(
        f"{len(uploaded_files)} PDF(s) uploaded successfully."
    )

    st.subheader("Uploaded Documents")

    for file in uploaded_files:
        st.write(f"📄 **{file.name}**")

    st.divider()

    # --------------------------------------------------
    # Extract Text
    # --------------------------------------------------

    st.subheader("PDF Text Extraction")

    all_pages = []

    for file in uploaded_files:

        with st.spinner(
            f"Extracting text from {file.name}..."
        ):

            pages = extract_text_from_pdf(file)

        for page in pages:
            page["text"] = clean_text(page["text"])

        all_pages.extend(pages)

        st.write(
            f"Done extracting {file.name}: "
            f"{len(pages)} page(s) with text extracted"
        )

    st.session_state.pdf_pages = all_pages


    # --------------------------------------------------
    # Create Chunks
    # --------------------------------------------------

    chunks = create_chunks(
        all_pages,
        chunk_size=150,
        overlap=30
    )

    st.session_state.chunks = chunks

    st.subheader("Text Chunks")

    st.write(
        f"Created **{len(chunks)} chunks** "
        f"from the uploaded PDFs."
    )


    # --------------------------------------------------
    # Create Embeddings
    # --------------------------------------------------

    with st.spinner("Creating text embeddings..."):

        embedding_model = load_embedding_model()

        embeddings = create_embeddings(
            embedding_model,
            chunks
        )

    st.session_state.embedding_model = embedding_model
    st.session_state.embeddings = embeddings


    # --------------------------------------------------
    # Create FAISS Index
    # --------------------------------------------------

    faiss_index = create_faiss_index(
        embeddings
    )

    st.session_state.faiss_index = faiss_index


    # --------------------------------------------------
    # Load QA Model
    # --------------------------------------------------

    with st.spinner("Loading Question Answering model..."):

        qa_model = load_qa_model()

    st.session_state.qa_model = qa_model


    # --------------------------------------------------
    # Chunk Preview
    # --------------------------------------------------

    if chunks:

        st.subheader("Text Chunk Preview")

        selected_chunk = st.selectbox(
            "Select a chunk to preview:",
            range(len(chunks))
        )

        chunk = chunks[selected_chunk]

        st.write(
            f"**Chunk ID:** {chunk['chunk_id']}"
        )

        st.write(
            f"**PDF:** {chunk['pdf_name']}"
        )

        st.write(
            f"**Page:** {chunk['page_number']}"
        )

        st.text_area(
            "Chunk Text",
            chunk["text"],
            height=200
        )


    # --------------------------------------------------
    # PDF Page Preview
    # --------------------------------------------------

    if all_pages:

        st.subheader("PDF Text Preview")

        selected_page = st.selectbox(
            "Select a page to preview:",
            range(len(all_pages))
        )

        page = all_pages[selected_page]

        st.write(
            f"**PDF:** {page['pdf_name']}"
        )

        st.write(
            f"**Page:** {page['page_number']}"
        )

        st.text_area(
            "Extracted Text",
            page["text"],
            height=300
        )


    st.divider()


    # --------------------------------------------------
    # Question Answering
    # --------------------------------------------------

    st.subheader("Ask a Question")

    question = st.text_input(
        "Enter your question:",
        placeholder="e.g., What is Natural Language Processing?"
    )


    if st.button("Ask Question"):

        if question.strip():

            # ------------------------------------------
            # Semantic Retrieval
            # ------------------------------------------

            with st.spinner(
                "Searching the PDF..."
            ):

                question_embedding = create_question_embedding(
                    st.session_state.embedding_model,
                    question
                )

                results = search_faiss(
                    st.session_state.faiss_index,
                    question_embedding,
                    st.session_state.chunks,
                    top_k=3
                )


            if results:

                # --------------------------------------
                # Create Context
                # --------------------------------------

                context = " ".join(
                    result["text"]
                    for result in results
                )


                # --------------------------------------
                # Generate Answer
                # --------------------------------------

                with st.spinner(
                    "Generating answer..."
                ):

                    answer_result = get_answer(
                        st.session_state.qa_model,
                        question,
                        context
                    )


                answer = answer_result["answer"]


                # --------------------------------------
                # Save Chat History
                # --------------------------------------

                chat_entry = {
                    "question": question,
                    "answer": answer,
                    "sources": [
                        {
                            "pdf_name": result["pdf_name"],
                            "page_number": result["page_number"]
                        }
                        for result in results
                    ]
                }

                st.session_state.chat_history.append(
                    chat_entry
                )


                # --------------------------------------
                # Display Answer
                # --------------------------------------

                st.subheader("🤖 Answer")

                st.success(answer)


                # --------------------------------------
                # Display Main Source
                # --------------------------------------

                st.subheader("📚 Answer Source")

                best_source = results[0]

                st.markdown(
                    f"""
                    **PDF:** `{best_source['pdf_name']}`

                    **Page:** `{best_source['page_number']}`

                    **Source Distance:** `{best_source['distance']:.4f}`
                    """
                )

                st.info(
                    "The answer was generated from the most relevant "
                    "retrieved document context shown below."
                )


                with st.expander("🔎 View Source Text", expanded=True):

                    st.write(
                        best_source["text"]
                    )


                # --------------------------------------
                # Display Other Retrieved Sources
                # --------------------------------------

                if len(results) > 1:

                    st.subheader("📖 Other Relevant Sources")

                    for i, result in enumerate(
                        results[1:],
                        start=2
                    ):

                        with st.expander(
                            f"Source {i}: "
                            f"{result['pdf_name']} — "
                            f"Page {result['page_number']}"
                        ):

                            st.write(
                                result["text"]
                            )


            else:

                st.warning(
                    "No relevant content found."
                )


        else:

            st.warning(
                "Please enter a question."
            )


    # --------------------------------------------------
    # Chat History
    # --------------------------------------------------

    st.divider()

    st.subheader("💬 Chat History")

    if st.session_state.chat_history:

        for i, chat in enumerate(
            st.session_state.chat_history,
            start=1
        ):

            st.markdown(
                f"### Question {i}"
            )

            st.write(
                f"**You:** {chat['question']}"
            )

            st.write(
                f"**Answer:** {chat['answer']}"
            )

            st.write("**Sources:**")

            for source in chat["sources"]:

                st.write(
                    f"📄 {source['pdf_name']} — "
                    f"Page {source['page_number']}"
                )

            st.divider()

    else:

        st.info(
            "No questions asked yet."
        )


else:

    st.info(
        "Please upload at least one PDF to get started."
    )