import streamlit as st
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# Download and cache Qwen 1.5B GGUF
@st.cache_resource
def load_llm():
    model_path = hf_hub_download(
        repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF", 
        filename="qwen2.5-1.5b-instruct-q4_k_m.gguf"
    )
    return Llama(model_path=model_path, n_ctx=2048, verbose=False)

st.title("🤖 Qwen 1.5B Assistant")

# Reasoning Toggle Checkbox
enable_reasoning = st.checkbox("Enable Reasoning Mode (<think> blocks)")

user_input = st.text_input("Ask something:")

if user_input:
    llm = load_llm()
    
    # Configure reasoning logic
    system_prompt = (
        "You are a helpful assistant. Think step-by-step inside <think></think> tags before answering."
        if enable_reasoning
        else "You are a concise assistant. Provide direct answers immediately without <think> tags."
    )
    
    with st.spinner("Generating response..."):
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0.7 if enable_reasoning else 0.2,
            max_tokens=512
        )
        
    st.write(response["choices"][0]["message"]["content"])
