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
footer {visibility: hidden}
"""

# --- Path Configuration ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
# Files are in the root directory alongside app.py
LORA_PATH = CURRENT_DIR 

# --- Model Loading ---
print(f"Initializing Base Model: {MODEL_ID}")
base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
)

print(f"Applying LoRA Adapters from: {LORA_PATH}")
model = PeftModel.from_pretrained(base_model, LORA_PATH)
processor = AutoProcessor.from_pretrained(LORA_PATH)
model.eval()

# --- Inference Logic ---
def analyze_document(input_img, custom_prompt):
    if input_img is None:
        return "Error: No image provided. Please upload a document."
    
    if not custom_prompt:
        custom_prompt = "Convert this document image into structured Markdown."

    try:
        # Prepare content for Qwen2-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": input_img},
                    {"type": "text", "text": custom_prompt},
                ],
            }
        ]

        # 1. Apply Template
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # 2. Extract Vision Info
        image_inputs, video_inputs = process_vision_info(messages)
        
        # 3. Process Inputs (Force float16 to match model)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(model.device).to(torch.float16)

        # 4. Generate
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=1024,
                do_sample=False # Keep it deterministic for document tasks
            )
            
            # Trim the prompt tokens
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )

        return output_text[0]
    
    except Exception as e:
        return f"System Error: {str(e)}"

# --- UI Layout ---
with gr.Blocks(css=custom_css, theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# 🔮 STRUCTURA-VL: DOCUMENT INTELLIGENCE", elem_id="header-text")
    
    with gr.Row():
        with gr.Column(scale=1, variant="panel"):
            img_input = gr.Image(type="pil", label="Input Document")
            prompt_input = gr.Textbox(
                label="System Prompt", 
                value="Convert this document image into structured Markdown.",
                lines=2
            )
            submit_btn = gr.Button("✨ EXTRACT MARKDOWN", variant="primary")
        
        with gr.Column(scale=2):
            md_output = gr.Markdown(label="Rendered Result")
            with gr.Accordion("Raw Markdown Source", open=False):
                raw_output = gr.Code(label="Code", language="markdown")

    # Map Events
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