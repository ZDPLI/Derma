import gradio as gr
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image

from rag import RAGMemory
from auth import AuthManager

MODEL_ID = "ContactDoctor/Bio-Medical-MultiModal-Llama-3-8B-V1"

# load model once on startup
model, processor = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True), AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

rag_memory = RAGMemory.from_file("data/knowledge_base.txt")
auth = AuthManager()


def generate_reply(message, image, history, user):
    if not user or not user.get("subscription"):
        reply = "Your subscription is inactive."
        history.append((message, reply))
        return history, history

    context_docs = rag_memory.query(message, top_k=3)
    context = "\n".join(context_docs)
    prompt = f"Context:\n{context}\n\nUser: {message}\nAssistant:"

    inputs = processor(text=prompt, images=image, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=300)
    reply = processor.decode(outputs[0], skip_special_tokens=True)

    rag_memory.add_document(f"Question: {message}\nAnswer: {reply}")
    history.append((message, reply))
    return history, history


def handle_login(username, password):
    user = auth.authenticate(username, password)
    if user:
        return user, f"Welcome {username}!"
    return None, "Invalid credentials"


def get_user_table():
    rows = ["| Username | Subscription | Admin |", "| --- | --- | --- |"]
    for u in auth.get_users_info():
        rows.append(f"| {u['username']} | {'✅' if u['subscription'] else '❌'} | {'✅' if u['is_admin'] else '❌'} |")
    return "\n".join(rows)


def toggle_sub(username):
    if auth.toggle_subscription(username):
        return get_user_table()
    return "User not found"


def add_user(username, password):
    if auth.add_user(username, password):
        return get_user_table()
    return "Failed to add user"


def show_panels(user):
    if user:
        chat_vis = user.get("subscription", False)
        admin_vis = user.get("is_admin", False)
        return gr.update(visible=False), gr.update(visible=chat_vis), gr.update(visible=admin_vis)
    return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)


with gr.Blocks(theme=gr.themes.Base()) as demo:
    user_state = gr.State(value=None)

    with gr.Column() as login_col:
        gr.Markdown("## Login")
        username = gr.Textbox(label="Username")
        password = gr.Textbox(type="password", label="Password")
        login_btn = gr.Button("Login")
        login_msg = gr.Markdown()

    with gr.Column(visible=False) as chat_col:
        gr.Markdown("# Bio-Medical MultiModal Llama 3 Chatbot")
        chatbot = gr.Chatbot()
        msg = gr.Textbox(label="Your question")
        img = gr.Image(type="pil", label="Optional image input")
        history = gr.State([])
        submit = gr.Button("Send")

        submit.click(generate_reply, [msg, img, history, user_state], [chatbot, history])
        msg.submit(generate_reply, [msg, img, history, user_state], [chatbot, history])

    with gr.Column(visible=False) as admin_col:
        gr.Markdown("## Admin Panel")
        users_md = gr.Markdown(get_user_table())
        refresh_btn = gr.Button("Refresh")
        toggle_name = gr.Textbox(label="Toggle subscription for user")
        toggle_btn = gr.Button("Toggle")
        new_user = gr.Textbox(label="New username")
        new_pass = gr.Textbox(type="password", label="New password")
        add_btn = gr.Button("Add user")

        refresh_btn.click(lambda: get_user_table(), None, users_md)
        toggle_btn.click(toggle_sub, toggle_name, users_md)
        add_btn.click(add_user, [new_user, new_pass], users_md)

    login_btn.click(handle_login, [username, password], [user_state, login_msg]).then(
        show_panels, user_state, [login_col, chat_col, admin_col]
    )


def main():
    demo.launch()


if __name__ == "__main__":
    main()

