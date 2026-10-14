
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

st.set_page_config(page_title="Qwen AI", page_icon="🤖")
st.title("Qwen AI Assistant")

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

@st.cache_resource(show_spinner="Downloading and loading Qwen...")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    model.eval()
    return tokenizer, model

try:
    tokenizer, model = load_model()
    st.success("Model loaded.")
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask Qwen...")

if prompt:
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful, concise assistant."
        }
    ] + st.session_state.messages

    try:
        with st.chat_message("assistant"):
            with st.spinner("Generating..."):
                inputs = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt"
                )

                with torch.inference_mode():
                    outputs = model.generate(
                        inputs,
                        max_new_tokens=128,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )

                answer = tokenizer.decode(
                    outputs[0][inputs.shape[-1]:],
                    skip_special_tokens=True
                )

                st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

    except Exception as e:
        st.error(f"Generation failed: {e}")
