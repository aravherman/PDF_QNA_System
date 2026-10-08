import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

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

from evaluation.evaluate import (
    evaluate_answer,
    calculate_overall_metrics
)


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Evaluation Dashboard",
    page_icon="📊",
    layout="wide"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("📊 Evaluation Dashboard")

st.write(
    "Evaluate the performance of the PDF Question "
    "Answering System using a predefined test dataset."
)

st.divider()


# --------------------------------------------------
# Load Test Dataset
# --------------------------------------------------

try:

    test_data = pd.read_csv(
        "evaluation/test_questions.csv"
    )

except FileNotFoundError:

    st.error(
        "evaluation/test_questions.csv "
        "was not found."
    )

    st.stop()


st.subheader("📋 Test Dataset")

st.write(
    f"Total questions: **{len(test_data)}**"
)

st.dataframe(
    test_data,
    use_container_width=True
)


st.divider()


# --------------------------------------------------
# Upload Evaluation PDFs
# --------------------------------------------------

st.subheader("📚 Evaluation Documents")

evaluation_files = st.file_uploader(
    "Upload the PDFs used in the test dataset",
    type=["pdf"],
    accept_multiple_files=True,
    key="evaluation_pdfs"
)


if not evaluation_files:

    st.info(
        "Upload the PDFs listed in test_questions.csv "
        "to start evaluation."
    )

    st.stop()


# --------------------------------------------------
# Extract PDF Text
# --------------------------------------------------

all_pages = []

with st.spinner("Extracting PDF text..."):

    for file in evaluation_files:

        pages = extract_text_from_pdf(file)

        for page in pages:

            page["text"] = clean_text(
                page["text"]
            )

        all_pages.extend(pages)


# --------------------------------------------------
# Create Chunks
# --------------------------------------------------

chunks = create_chunks(
    all_pages,
    chunk_size=150,
    overlap=30
)

st.success(
    f"Created {len(chunks)} text chunks."
)


# --------------------------------------------------
# Create Embeddings
# --------------------------------------------------

with st.spinner(
    "Creating document embeddings..."
):

    embedding_model = load_embedding_model()

    embeddings = create_embeddings(
        embedding_model,
        chunks
    )


# --------------------------------------------------
# Create FAISS Index
# --------------------------------------------------

faiss_index = create_faiss_index(
    embeddings
)


# --------------------------------------------------
# Load QA Model
# --------------------------------------------------

with st.spinner(
    "Loading Question Answering model..."
):

    qa_model = load_qa_model()


st.divider()


# --------------------------------------------------
# Run Evaluation
# --------------------------------------------------

st.subheader("🚀 Run Evaluation")

if st.button(
    "Run Evaluation",
    type="primary"
):

    evaluation_results = []

    progress_bar = st.progress(0)

    total_questions = len(
        test_data
    )

    for index, row in test_data.iterrows():

        question = row["question"]

        expected_answer = row[
            "expected_answer"
        ]


        # ------------------------------------------
        # Question Embedding
        # ------------------------------------------

        question_embedding = (
            create_question_embedding(
                embedding_model,
                question
            )
        )


        # ------------------------------------------
        # Retrieve Relevant Chunks
        # ------------------------------------------

        results = search_faiss(
            faiss_index,
            question_embedding,
            chunks,
            top_k=3
        )


        if not results:

            evaluation_results.append({

                "question": question,

                "expected": expected_answer,

                "predicted": "No answer found",

                "confidence": 0,

                "exact_match": 0,

                "f1_score": 0

            })

            continue


        # ------------------------------------------
        # Create Context
        # ------------------------------------------

        context = " ".join(
            result["text"]
            for result in results
        )


        # ------------------------------------------
        # Generate Answer
        # ------------------------------------------

        answer_result = get_answer(
            qa_model,
            question,
            context
        )


        predicted_answer = (
            answer_result["answer"]
        )

        confidence = (
            answer_result["score"]
        )


        # ------------------------------------------
        # Evaluate Answer
        # ------------------------------------------

        scores = evaluate_answer(
            predicted_answer,
            expected_answer
        )


        evaluation_results.append({

            "question": question,

            "expected": expected_answer,

            "predicted": predicted_answer,

            "confidence": confidence,

            "exact_match": scores[
                "exact_match"
            ],

            "f1_score": scores[
                "f1_score"
            ]

        })


        progress = (
            (index + 1)
            / total_questions
        )

        progress_bar.progress(
            progress
        )


    # --------------------------------------------------
    # Overall Metrics
    # --------------------------------------------------

    metrics = calculate_overall_metrics(
        evaluation_results
    )


    st.session_state[
        "evaluation_results"
    ] = evaluation_results

    st.session_state[
        "evaluation_metrics"
    ] = metrics


    st.success(
        "Evaluation completed successfully!"
    )


# --------------------------------------------------
# Display Results
# --------------------------------------------------

if "evaluation_metrics" in st.session_state:

    metrics = st.session_state[
        "evaluation_metrics"
    ]

    results = st.session_state[
        "evaluation_results"
    ]


    st.divider()

    st.subheader(
        "📈 Overall Performance"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Exact Match",
            f"{metrics['exact_match'] * 100:.2f}%"
        )


    with col2:

        st.metric(
            "F1 Score",
            f"{metrics['f1_score'] * 100:.2f}%"
        )


    with col3:

        st.metric(
            "Average Confidence",
            f"{metrics['confidence'] * 100:.2f}%"
        )


    st.divider()


    # --------------------------------------------------
    # Results Table
    # --------------------------------------------------

    st.subheader(
        "📝 Question-wise Results"
    )


    results_df = pd.DataFrame(
        results
    )

    results_df["confidence"] = (
        results_df["confidence"] * 100
    )

    results_df["f1_score"] = (
        results_df["f1_score"] * 100
    )


    results_df = results_df.rename(
        columns={
            "question": "Question",
            "expected": "Expected Answer",
            "predicted": "Predicted Answer",
            "confidence": "Confidence (%)",
            "exact_match": "Exact Match",
            "f1_score": "F1 Score (%)"
        }
    )


    st.dataframe(
        results_df,
        use_container_width=True
    )


    st.divider()


    # --------------------------------------------------
    # Performance Chart
    # --------------------------------------------------

    st.subheader(
        "📊 Performance Metrics"
    )


    chart_data = pd.DataFrame({

        "Metric": [
            "Exact Match",
            "F1 Score",
            "Confidence"
        ],

        "Score": [

            metrics["exact_match"] * 100,

            metrics["f1_score"] * 100,

            metrics["confidence"] * 100

        ]

    })


    fig, ax = plt.subplots()

    ax.bar(
        chart_data["Metric"],
        chart_data["Score"]
    )

    ax.set_ylabel(
        "Score (%)"
    )

    ax.set_ylim(
        0,
        100
    )

    ax.set_title(
        "PDF QA System Performance"
    )

    st.pyplot(fig)