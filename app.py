import os
import torch
import gradio as gr
from PIL import Image
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# --- Professional "Cyber-Dark" CSS ---
custom_css = """
.gradio-container {background-color: #050505; color: #e5e7eb;}
.feedback-card {border: 1px solid #6366f1; padding: 20px; border-radius: 12px; background: #0f172a; margin: 10px 0;}
#header-text {text-align: center; background: linear-gradient(to right, #818cf8, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.5rem; font-weight: 800;}
button.primary {background: linear-gradient(90deg, #6366f1, #a855f7) !important; color: white !important; font-weight: bold !important; border: none !important; transition: 0.3s;}
button.primary:hover {transform: scale(1.02); opacity: 0.9;}
.tabs {border: none !important;}
.gradio-container .prose h3 {color: #818cf8;}
footer {visibility: hidden}
"""

# --- Path Configuration ---
# Since files are in the same directory as app.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
LORA_PATH = CURRENT_DIR 

# --- Model Loading ---
print(f"Loading Base Model: {MODEL_ID}")
base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
)

print(f"Loading Adapters from: {LORA_PATH}")
model = PeftModel.from_pretrained(base_model, LORA_PATH)
processor = AutoProcessor.from_pretrained(LORA_PATH)
model.eval()

# --- Inference Logic ---
def analyze_document(input_img, custom_prompt):
    if input_img is None:
        return "Please upload an image first."
    
    if not custom_prompt:
        custom_prompt = "Convert this document image into structured Markdown."

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": input_img},
                {"type": "text", "text": custom_prompt},
            ],
        }
    ]

    # Preprocessing
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt"
    ).to("cuda")

    # Generation
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=1024)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )

    return output_text[0]

# --- UI Construction ---
with gr.Blocks(css=custom_css, theme=gr.themes.Monochrome()) as demo:
    with gr.Column():
        gr.Markdown("# 🔮 STRUCTURA-VL: DOCUMENT INTELLIGENCE", elem_id="header-text")
        gr.Markdown("Fine-tuned Vision-Language Model specializing in OCR and Layout Analysis.")

    with gr.Tabs(elem_classes="tabs"):
        with gr.TabItem("🚀 Conversion Lab"):
            with gr.Row():
                with gr.Column(scale=1, variant="panel"):
                    gr.Markdown("### Input Source")
                    img_input = gr.Image(type="pil", label="Upload Document/Scan")
                    
                    with gr.Accordion("Advanced Settings", open=False):
                        prompt_input = gr.Textbox(
                            label="System Prompt", 
                            placeholder="Convert this document into structured Markdown...",
                            lines=2
                        )
                    
                    submit_btn = gr.Button("✨ EXTRACT MARKDOWN", variant="primary")
                
                with gr.Column(scale=2):
                    gr.Markdown("### Markdown Output")
                    md_output = gr.Markdown(label="Rendered Result", value="*Result will appear here...*")
                    with gr.Accordion("Raw Text", open=False):
                        raw_output = gr.Code(label="Source Code", language="markdown")

    # Mapping logic
    submit_btn.click(
        fn=analyze_document, 
        inputs=[img_input, prompt_input], 
        outputs=[md_output]
    ).then(
        fn=lambda x: x,
        inputs=[md_output],
        outputs=[raw_output]
    )

if __name__ == "__main__":
    demo.launch()