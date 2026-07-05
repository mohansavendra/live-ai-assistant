import gradio as gr
from agent import run_agent
from memory import MemoryManager

# ─── GLOBAL MEMORY MANAGER ────────────────────────────────────────────────────
memory = MemoryManager()


# ─── CORE CHAT FUNCTION ───────────────────────────────────────────────────────

def chat(
    user_input: str,
    history: list,
    use_persistent_memory: bool,
) -> tuple:
    """
    Main chat handler called by Gradio on every message.
    Returns updated history and status string.
    """
    if not user_input.strip():
        return history, memory.status()

    # Sync persistent memory toggle
    memory.use_persistent = use_persistent_memory

    # Add user message to session memory
    memory.add_user(user_input)

    # Retrieve context (session + optional persistent)
    context = memory.get_context(user_input)

    # Run the LangGraph agent
    response = run_agent(
        user_input=user_input,
        session_history=context["session_history"][:-1],  # exclude current user msg
        persistent_context=context["persistent_memories"],
    )

    # Store assistant response
    memory.add_assistant(response, user_input=user_input)

    # Update Gradio history format
    history.append((user_input, response))

    return history, memory.status()


def clear_session_memory():
    memory.clear_session()
    return [], memory.status()


def clear_all_memory():
    memory.clear_all()
    return [], memory.status()


# ─── GRADIO UI ────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-primary: #0d0f12;
    --bg-secondary: #13161b;
    --bg-card: #1a1d24;
    --bg-input: #1e2128;
    --accent: #6ee7b7;
    --accent-dim: #34d39920;
    --accent-border: #6ee7b730;
    --text-primary: #e8eaf0;
    --text-secondary: #8b92a5;
    --text-muted: #4a5166;
    --border: #252930;
    --user-bubble: #1e2a3a;
    --bot-bubble: #161a20;
    --radius: 12px;
}

* { box-sizing: border-box; }

body, .gradio-container {
    background: var(--bg-primary) !important;
    font-family: 'Inter', sans-serif !important;
    color: var(--text-primary) !important;
}

/* Header */
.app-header {
    padding: 28px 0 8px;
    text-align: center;
    border-bottom: 1px solid var(--border);
    margin-bottom: 20px;
}
.app-title {
    font-size: 22px;
    font-weight: 600;
    color: var(--accent);
    letter-spacing: -0.3px;
    font-family: 'JetBrains Mono', monospace;
}
.app-subtitle {
    font-size: 13px;
    color: var(--text-secondary);
    margin-top: 4px;
}

/* Chatbot */
.chatbot-wrap .wrap {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}

/* Input row */
.input-row {
    display: flex;
    gap: 8px;
    align-items: flex-end;
    margin-top: 10px;
}

textarea, input[type="text"] {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
}
textarea:focus, input:focus {
    border-color: var(--accent-border) !important;
    outline: none !important;
    box-shadow: 0 0 0 2px var(--accent-dim) !important;
}

/* Buttons */
button.primary-btn {
    background: var(--accent) !important;
    color: #0a0d0f !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 13px !important;
    padding: 10px 20px !important;
    cursor: pointer !important;
    transition: opacity 0.15s !important;
}
button.primary-btn:hover { opacity: 0.85 !important; }

button.secondary-btn {
    background: var(--bg-card) !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    font-size: 12px !important;
    padding: 8px 14px !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
}
button.secondary-btn:hover {
    border-color: var(--accent-border) !important;
    color: var(--accent) !important;
}

/* Status bar */
.status-bar {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 8px 14px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    color: var(--text-muted) !important;
}

/* Toggle */
.memory-toggle label {
    color: var(--text-secondary) !important;
    font-size: 13px !important;
}
.memory-toggle input[type="checkbox"] {
    accent-color: var(--accent) !important;
}

/* Section labels */
.section-label {
    font-size: 11px !important;
    font-weight: 500 !important;
    color: var(--text-muted) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
    margin-bottom: 6px !important;
}
"""

with gr.Blocks(css=CSS, title="Live AI Assistant", theme=gr.themes.Base()) as demo:

    # Header
    gr.HTML("""
        <div class="app-header">
            <div class="app-title">⬡ LIVE AI ASSISTANT</div>
            <div class="app-subtitle">Real-time search · Tool calling · Persistent memory</div>
        </div>
    """)

    with gr.Row():
        # Main chat column
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                label="",
                height=520,
                show_label=False,
                bubble_full_width=False,
                elem_classes=["chatbot-wrap"],
                avatar_images=(None, "https://api.dicebear.com/7.x/bottts-neutral/svg?seed=assistant"),
            )

            with gr.Row(elem_classes=["input-row"]):
                user_input = gr.Textbox(
                    placeholder="Ask anything — I can search the web for current info...",
                    show_label=False,
                    lines=1,
                    max_lines=4,
                    scale=5,
                )
                send_btn = gr.Button("Send", elem_classes=["primary-btn"], scale=1)

        # Sidebar
        with gr.Column(scale=1, min_width=200):
            gr.HTML('<div class="section-label">Memory</div>')

            persistent_toggle = gr.Checkbox(
                label="Persistent memory (ChromaDB)",
                value=False,
                elem_classes=["memory-toggle"],
                info="Remembers across sessions"
            )

            gr.HTML('<div class="section-label" style="margin-top:16px">Actions</div>')

            clear_session_btn = gr.Button(
                "Clear session",
                elem_classes=["secondary-btn"],
            )
            clear_all_btn = gr.Button(
                "Clear all memory",
                elem_classes=["secondary-btn"],
            )

            gr.HTML('<div class="section-label" style="margin-top:16px">Status</div>')

            status_box = gr.Textbox(
                value=memory.status(),
                show_label=False,
                interactive=False,
                elem_classes=["status-bar"],
                lines=3,
            )

            gr.HTML("""
                <div style="margin-top:20px; padding:12px; background:#13161b;
                     border:1px solid #252930; border-radius:10px;">
                    <div style="font-size:11px; color:#4a5166; font-weight:500;
                         text-transform:uppercase; letter-spacing:0.8px; margin-bottom:8px;">
                        Capabilities
                    </div>
                    <div style="font-size:12px; color:#6b7280; line-height:1.8;">
                        🌐 Web search (Tavily)<br>
                        ✅ Fact verification<br>
                        🧠 Session memory<br>
                        💾 Persistent memory<br>
                        ⚡ LLaMA 3.1 70B (Groq)
                    </div>
                </div>
            """)

    # ─── EVENT WIRING ─────────────────────────────────────────────────────────

    def handle_submit(user_msg, history, use_persistent):
        return chat(user_msg, history, use_persistent)

    send_btn.click(
        fn=handle_submit,
        inputs=[user_input, chatbot, persistent_toggle],
        outputs=[chatbot, status_box],
    ).then(
        fn=lambda: "",
        outputs=[user_input],
    )

    user_input.submit(
        fn=handle_submit,
        inputs=[user_input, chatbot, persistent_toggle],
        outputs=[chatbot, status_box],
    ).then(
        fn=lambda: "",
        outputs=[user_input],
    )

    clear_session_btn.click(
        fn=clear_session_memory,
        outputs=[chatbot, status_box],
    )

    clear_all_btn.click(
        fn=clear_all_memory,
        outputs=[chatbot, status_box],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,       # set True to get a public Gradio link instantly
        show_error=True,
    )
