import streamlit as st
import requests

# 1. Page Configuration
st.set_page_config(
    page_title="AI Support Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. High-Contrast Custom CSS
st.markdown("""
<style>
    /* Card Styles */
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        color: #ffffff !important;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    
    .main-header h1 {
        color: #ffffff !important;
        margin: 0;
        font-size: 2rem;
    }
    
    .main-header p {
        color: #cbd5e1 !important;
        margin: 5px 0 0 0;
    }
    
    /* Sources Tag */
    .source-tag {
        display: inline-block;
        background-color: #2563eb;
        color: #ffffff;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 6px;
        margin-top: 4px;
    }
    
    /* Ticket Card */
    .ticket-card {
        background-color: #fef3c7;
        color: #92400e;
        border: 1px solid #fde68a;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        margin-top: 8px;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# 3. Sidebar Setup
with st.sidebar:
    st.title("⚙️ Support Dashboard")
    st.caption("Powered by RAG & Groq LLM")
    st.markdown("---")
    
    api_url = st.text_input("FastAPI Endpoint", value="http://localhost:8000")
    session_id = st.text_input("Active Session ID", value="session-001")
    
    st.markdown("---")
    st.subheader("System Status")
    st.success("🟢 Vector DB: Active")
    st.info("🟢 LLM: Connected")
    
    if st.button("🗑️ Reset Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# 4. Header Area
st.markdown("""
<div class="main-header">
    <h1>🤖 AI Customer Support Agent</h1>
    <p>Ask questions about shipping, returns, or request a support ticket.</p>
</div>
""", unsafe_allow_html=True)

# 5. Quick Interactive Prompt Buttons
st.subheader("💬 Quick Actions")
col1, col2, col3 = st.columns(3)

prompt_input = None
if col1.button("📦 Return Policy", use_container_width=True):
    prompt_input = "What is your return policy?"
if col2.button("🚚 Shipping Details", use_container_width=True):
    prompt_input = "How long does shipping take?"
if col3.button("🎟️ Open Support Ticket", use_container_width=True):
    prompt_input = "I need to open a support ticket for a missing item."

st.markdown("---")

# 6. Initialize History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I assist you today?", "sources": [], "ticket_id": None}
    ]

# 7. Render Chat Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        
        if msg.get("sources"):
            st.markdown("**Knowledge Sources:**")
            sources_html = "".join([f'<span class="source-tag">📄 {src}</span>' for src in msg["sources"]])
            st.markdown(sources_html, unsafe_allow_html=True)
            
        if msg.get("ticket_id"):
            st.markdown(f"""
            <div class="ticket-card">
                🎟️ <strong>Support Ticket Created:</strong> <code>{msg['ticket_id']}</code>
            </div>
            """, unsafe_allow_html=True)

# 8. User Interaction Handling
chat_input = st.chat_input("Type your message...")
user_message = prompt_input or chat_input

if user_message:
    st.session_state.messages.append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.write(user_message)

    with st.chat_message("assistant"):
        with st.spinner("Processing request..."):
            try:
                endpoint = f"{api_url.rstrip('/')}/chat"
                payload = {"session_id": session_id, "message": user_message}
                
                res = requests.post(endpoint, json=payload, timeout=30)
                if res.status_code == 200:
                    data = res.json()
                    bot_text = data.get("response", "")
                    sources = data.get("sources", [])
                    ticket_id = data.get("ticket_id", None)
                    
                    st.write(bot_text)
                    
                    if sources:
                        st.markdown("**Knowledge Sources:**")
                        sources_html = "".join([f'<span class="source-tag">📄 {src}</span>' for src in sources])
                        st.markdown(sources_html, unsafe_allow_html=True)
                        
                    if ticket_id:
                        st.markdown(f"""
                        <div class="ticket-card">
                            🎟️ <strong>Support Ticket Created:</strong> <code>{ticket_id}</code>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": bot_text,
                        "sources": sources,
                        "ticket_id": ticket_id
                    })
                else:
                    st.error(f"Error {res.status_code}: Unable to process request.")
            except Exception as e:
                st.error(f"Failed to connect to backend: {str(e)}")