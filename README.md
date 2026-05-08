# 🔮 Structura-VL: Intelligent Document-to-Markdown Engine

[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/AliHamza852/Structura-VL)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Structura-VL** is a high-fidelity vision-language engine designed to transform complex document scans into structured, editable Markdown. By leveraging fine-tuned VLMs, it goes beyond traditional OCR to understand visual hierarchy, headers, and nested layouts.

---

## 🚀 Live Demo
Experience the "Cyber-Dark" interface and real-time conversion here:
**[Structura-VL on Hugging Face Spaces](https://huggingface.co/spaces/AliHamza852/Structura-VL)**

---

## ✨ Key Features
* **Layout Awareness:** Recognizes the difference between main headers (`#`), sub-headers (`###`), and bulleted lists.
* **VLM Powered:** Built on the **Qwen2-VL-2B-Instruct** architecture for state-of-the-art vision understanding.
* **Optimized Performance:** Fine-tuned using **QLoRA (4-bit Quantization)** to enable high-speed inference on T4 GPUs.
* **Cyber-Dark UI:** A custom-styled Gradio interface designed for a professional developer experience.

---

## 🛠️ Technical Implementation

### Model & Training
* **Base Model:** Qwen2-VL-2B-Instruct
* **Fine-Tuning Technique:** QLoRA (Rank 16, Alpha 32)
* **Dataset:** Nougat (Document-Markdown pairs)
* **Compute:** Trained on Kaggle Dual T4 GPUs

### Evaluation Metrics
The model was evaluated using ROUGE metrics to ensure high structural fidelity:
| Metric | Score |
| :--- | :--- |
| **ROUGE-1** | 0.7470 |
| **ROUGE-2** | 0.7317 |
| **ROUGE-L** | 0.7447 |

---

## 📂 Project Structure
```text
├── app.py                  # Gradio Interface & Inference Logic
├── requirements.txt        # System Dependencies
├── adapter_config.json     # LoRA configuration
├── adapter_model.safetensors # Fine-tuned weights (LFS)
└── README.md               # Documentation

💻 Local Setup

    Clone the repository:
    Bash

git clone [https://github.com/Ali-Hamza852/Structura-VL.git](https://github.com/Ali-Hamza852/Structura-VL.git)
cd Structura-VL

Install dependencies:
Bash

pip install -r requirements.txt

Run the App:
Bash

    python app.py

🤝 Team

Developed by Ali Hamza
