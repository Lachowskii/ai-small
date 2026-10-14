
import streamlit as st
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import os

st.set_page_config(
    page_title="Qwen AI Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("Qwen AI Assistant")
st.caption("Running a quantized language model locally on the app server.")

MODEL_REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
MODEL_FILE = "qwen2.5-1.5b-instruct-q4_k_m.gguf"


@st.cache_resource(show_spinner=False)
def load_llm():
    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE
    )

    return Llama(
        model_path=model_path,
        n_ctx=1024,
        n_threads=2,
        n_batch=128,
        verbose=False
    )


if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

if not st.session_state.model_loaded:
    st.info("The model must be downloaded and initialized before use.")

    if st.button("Load AI Model", type="primary"):
        try:
            with st.spinner(
                "Downloading model if needed and initializing it..."
            ):
                llm = load_llm()

            st.session_state.model_loaded = True
            st.success("Model loaded successfully.")
            st.rerun()

        except Exception as e:
            st.error("Failed to load the model.")
            st.exception(e)

    st.stop()


enable_reasoning = st.checkbox("Enable reasoning-style responses")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask Qwen something...")

if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    system_prompt = (
        "You are a helpful assistant. Explain your reasoning clearly "
        "and provide a concise final answer."
        if enable_reasoning
        else "You are a concise and helpful assistant. "
             "Answer directly and accurately."
    )

    try:
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                llm = load_llm()

                response = llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        *st.session_state.messages
                    ],
                    temperature=0.3,
                    max_tokens=256
                )

                answer = response["choices"][0]["message"]["content"]
                st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

    except Exception as e:
        st.error(f"Generation failed: {e}")
