import streamlit as st
import chatbot as cb

st.title("Query any website")

txt_input = ""
with st.form(key='txt_input'):
    txt_input = st.text_area('Enter url', '', height=80)
    agent = cb.get_agent(txt_input)
    st.form_submit_button(label='Enter') 


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
            st.markdown(message["content"])    

if prompt := st.chat_input("What is up?"):
    with st.chat_message("user"):
        st.markdown(prompt) 

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = agent(prompt)['output']
        message_placeholder.markdown(full_response)
