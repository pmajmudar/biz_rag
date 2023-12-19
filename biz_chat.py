import streamlit as st
from inference_service import chat_infer, chat_infer2, query_hf

st.title("Biz chat")

# Use session state to store chat history across re-runs
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("What would you like to know?"):
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role":"user", "content": prompt})


    with st.chat_message("assistant"):
        # st.markdown(response)
        message_placeholder = st.empty()
        response = chat_infer2(prompt)
        while True:
            message_placeholder.markdown(response + "▌")
            if not response.strip().endswith('</s>'):
                response = query_hf(response)
            else:
                break
        message_placeholder.markdown(response)
    st.session_state.messages.append({"role":"assistant", "content": response})
