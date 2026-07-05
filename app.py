import gradio as gr
from agent import run_agent
from memory import MemoryManager

memory = MemoryManager()

def chat(user_input, history, use_persistent_memory):
    if not user_input.strip():
        return history, memory.status()
    memory.use_persistent = use_persistent_memory
    memory.add_user(user_input)
    context = memory.get_context(user_input)
    response = run_agent(
        user_input=user_input,
        session_history=context["session_history"][:-1],
        persistent_context=context["persistent_memories"],
    )
    memory.add_assistant(response, user_input=user_input)
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": response})
    return history, memory.status()

def clear_session_memory():
    memory.clear_session()
    return [], memory.status()

def clear_all_memory():
    memory.clear_all()
    return [], memory.status()

with gr.Blocks(title="Live AI Assistant") as demo:
    gr.HTML("""
        <div style="text-align:center; padding:20px 0 10px; border-bottom:1px solid #252930; margin-bottom:20px;">
            <div style="font-size:22px; font-weight:600; color:#6ee7b7; font-family:monospace;">⬡ LIVE AI ASSISTANT</div>
            <div style="font-size:13px; color:#8b92a5; margin-top:4px;">Real-time search · Tool calling · Persistent memory</div>
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                label="",
                height=520,
                show_label=False,
            )
            with gr.Row():
                user_input = gr.Textbox(
                    placeholder="Ask anything — I can search the web for current info...",
                    show_label=False,
                    lines=1,
                    max_lines=4,
                    scale=5,
                )
                send_btn = gr.Button("Send", scale=1, variant="primary")

        with gr.Column(scale=1, min_width=200):
            persistent_toggle = gr.Checkbox(
                label="Persistent memory (ChromaDB)",
                value=False,
                info="Remembers across sessions"
            )
            clear_session_btn = gr.Button("Clear session")
            clear_all_btn = gr.Button("Clear all memory")
            status_box = gr.Textbox(
                value=memory.status(),
                show_label=False,
                interactive=False,
                lines=3,
                label="Status"
            )
            gr.HTML("""
                <div style="margin-top:16px; padding:12px; background:#13161b; border:1px solid #252930; border-radius:10px;">
                    <div style="font-size:11px; color:#4a5166; font-weight:500; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:8px;">Capabilities</div>
                    <div style="font-size:12px; color:#6b7280; line-height:1.8;">
                        🌐 Web search (Tavily)<br>
                        ✅ Fact verification<br>
                        🧠 Session memory<br>
                        💾 Persistent memory<br>
                        ⚡ LLaMA 3.1 70B (Groq)
                    </div>
                </div>
            """)

    send_btn.click(
        fn=chat,
        inputs=[user_input, chatbot, persistent_toggle],
        outputs=[chatbot, status_box],
    ).then(fn=lambda: "", outputs=[user_input])

    user_input.submit(
        fn=chat,
        inputs=[user_input, chatbot, persistent_toggle],
        outputs=[chatbot, status_box],
    ).then(fn=lambda: "", outputs=[user_input])

    clear_session_btn.click(fn=clear_session_memory, outputs=[chatbot, status_box])
    clear_all_btn.click(fn=clear_all_memory, outputs=[chatbot, status_box])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)
