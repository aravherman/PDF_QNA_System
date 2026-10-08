import streamlit as st
import torch
from transformers import pipeline


# Instruction-tuned chat model (works with the "text-generation" pipeline).
# For a faster / lighter model on a slow CPU, use "Qwen/Qwen2.5-0.5B-Instruct".
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about PDF documents. "
    "Answer ONLY using the provided context. Keep the answer clear and concise. "
    "If the answer is not in the context, reply exactly: "
    "\"I could not find the answer in the uploaded document.\""
)


@st.cache_resource
def load_qa_model():
    """
    Load the chat model using the text-generation pipeline.
    """

    use_gpu = torch.cuda.is_available()

    return pipeline(
        "text-generation",
        model=MODEL_NAME,
        dtype=torch.float16 if use_gpu else torch.float32,
        device=0 if use_gpu else -1
    )


def get_answer(qa_model, question, context):
    """
    Generate an answer from the given context.
    """

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }
    ]

    output = qa_model(
        messages,
        max_new_tokens=200,
        do_sample=False,
        return_full_text=False
    )

    generated = output[0]["generated_text"]

    # Chat pipelines may return either a string or a list of messages
    if isinstance(generated, list):
        generated = generated[-1]["content"]

    return {
        "answer": generated.strip()
    }