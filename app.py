import streamlit as st
import chatbot as cb
from streamlit_chat import message


def generate_response(prompt,agent):
    st.session_state['messages'].append({"role": "user", "content": prompt})
    response = agent(prompt)['output']
    st.session_state['messages'].append({"role": "assistant", "content": response})
    return response


# Page title
st.set_page_config(page_title='🦜🔗 Chat with a website App')
st.title('🦜🔗 Chat with any website')

url_container = st.container()
# container for chat history
response_container = st.container()
# container for text box
container = st.container()

# Text input
txt_input = st.text_area('Enter url', '', height=80)
agent = cb.get_agent(txt_input)

with st.form(key='chat_app', clear_on_submit=True):
    user_input = st.text_area("You:", key='input', height=100)
    submit_button = st.form_submit_button(label='Send') 

if submit_button and user_input and txt_input:
    output = generate_response(user_input,agent)
    st.session_state['past'].append(user_input)
    st.session_state['generated'].append(output)

if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
                message(st.session_state["generated"][i], key=str(i))      
