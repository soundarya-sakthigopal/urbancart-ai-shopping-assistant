
import streamlit as st
import requests
import uuid

st.set_page_config(
    page_title="UrbanCart AI Assistant",
    page_icon="🛍️"
)

API_URL = "http://127.0.0.1:8000/api/chat"

st.title("🛍️ UrbanCart AI Shopping Assistant")
st.write("Ask questions about products, orders, inventory and company policies.")

# Maintain conversation state
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Customer input
if prompt := st.chat_input("How can I help you today?"):

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("UrbanCart AI is thinking..."):
            try:
                response = requests.post(
                    API_URL,
                    json={
                        "message": prompt,
                        "thread_id": st.session_state.thread_id
                    },
                    timeout=120
                )

                response.raise_for_status()

                result = response.json()
                answer = result["response"]

                st.markdown(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except requests.exceptions.RequestException as e:
                st.error(f"Unable to connect to UrbanCart API: {e}")

            except (ValueError, KeyError) as e:
                st.error(f"Invalid API response: {e}")
