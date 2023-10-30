import streamlit as st
import chatbot as cb

st.title("Query any website")

txt_input = ""
agent = any
with st.form(key='txt_input'):
    txt_input = st.text_area('Enter url', '', height=80)
    submit_button = st.form_submit_button(label='Enter')
    if submit_button:
          agent = cb.get_agent(txt_input)
          st.session_state['agent'] = agent

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
        agent = st.write(st.session_state.agent)
        full_response = agent(prompt)['output']
        message_placeholder.markdown(full_response)
