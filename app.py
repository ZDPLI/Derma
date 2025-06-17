import gradio as gr
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image

from rag import RAGMemory

MODEL_ID = "ContactDoctor/Bio-Medical-MultiModal-Llama-3-8B-V1"


def load_model():
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    return model, processor


model, processor = load_model()
rag_memory = RAGMemory.from_file("data/knowledge_base.txt")


def generate_reply(message, image, history):
    context_docs = rag_memory.query(message, top_k=3)
    context = "\n".join(context_docs)
    prompt = f"Context:\n{context}\n\nUser: {message}\nAssistant:"

    inputs = processor(text=prompt, images=image, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=300)
    reply = processor.decode(outputs[0], skip_special_tokens=True)

    rag_memory.add_document(f"Question: {message}\nAnswer: {reply}")
    history.append((message, reply))
    return history, history


def main():
    with gr.Blocks(theme=gr.themes.Base()) as demo:
        gr.Markdown("# Bio-Medical MultiModal Llama 3 Chatbot")
        chatbot = gr.Chatbot()
        msg = gr.Textbox(label="Your question")
        img = gr.Image(type="pil", label="Optional image input")
        state = gr.State([])
        submit = gr.Button("Send")

        submit.click(generate_reply, [msg, img, state], [chatbot, state])
        msg.submit(generate_reply, [msg, img, state], [chatbot, state])
    demo.launch()


if __name__ == "__main__":
    main()
