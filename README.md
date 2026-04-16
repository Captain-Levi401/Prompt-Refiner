# 🚀 Prompt Quality Scorer & Refiner

An end-to-end NLP application that evaluates and enhances natural language prompts using **DistilBERT** and **T5-base** models. Built with a clean Streamlit interface, this tool helps transform raw prompts into structured, high-quality inputs for better AI performance.

---

## ✨ Overview

This project combines **prompt classification** and **prompt refinement** into a unified pipeline:

* 🧠 **DistilBERT** scores prompt quality
* ✍️ **T5-base** rewrites prompts into optimized formats
* ⚡ Runs efficiently with optional GPU acceleration
* 🎯 Designed for real-world prompt engineering workflows

---

## 🔥 Key Features

* **Prompt Quality Scoring**
  Evaluate how effective a prompt is using a fine-tuned DistilBERT model

* **Prompt Refinement**
  Convert raw prompts into structured, high-quality versions using T5

* **Streamlit UI**
  Minimal, dark-themed interface with copy-ready outputs

* **GPU Support**
  Automatically utilizes CUDA if available

* **CI Integration**
  GitHub Actions workflow for syntax validation and code health

---

## 🛠️ Tech Stack

* Python
* Hugging Face Transformers
* PyTorch
* Streamlit

---

## 🏗️ Architecture

```mermaid
graph TD
    UI[User Browser / Streamlit UI] --> App[Streamlit app (`app.py`)]
    App --> Scorer[DistilBERT Prompt Quality Scorer]
    App --> Refiner[T5 Prompt Refiner]
    Scorer --> ModelA[DistilBERT weights (`./model`)]
    Refiner --> ModelB[T5 weights (`./final_prompt_refiner`)]
    App --> Device[Compute Device: CPU / GPU (CUDA)]
    App --> Output[Result Display & Copyable Output]
    subgraph Model Storage
        ModelA
        ModelB
    end
```

---

## 📈 Model Performance Metrics

* **DistilBERT Scorer**
  - Quality band classification with averaged probability inference for stability
  - Score range: 0–100 with labels `Worst`, `Good`, `Elite`
  - Confidence, robustness, and efficiency metrics are computed per prompt

* **T5 Refiner**
  - Converts raw prompts into structured objective, requirements, and output format
  - Uses beam search, repetition penalty, and EOS forcing for coherent generation
  - Includes CUDA-aware inference and fallback handling for GPU OOM recovery

* **Inference performance**
  - Optional GPU acceleration when CUDA is available
  - Tokenization input limit: 256 tokens for both scorer and refiner
  - Designed for interactive Streamlit usage with responsive output display

---

## 📁 Project Structure

```
prompt-quality-scorer-refiner/
│── app.py                      # Main Streamlit app
│── check_gpu.py                # GPU diagnostics script
│── requirements.txt           # Dependencies
│── pyproject.toml             # Project metadata
│
│── model/                     # DistilBERT scorer model
│── final_prompt_refiner/      # T5 refiner model
│
│── .gitignore
│── README.md
```

---

## ⚙️ Setup & Installation

### 1. Create Virtual Environment

**PowerShell**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

---

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

---

### 3. Verify Model Files

Ensure the following directories contain model weights and tokenizer files:

* `model/` → DistilBERT scorer
* `final_prompt_refiner/` → T5 refiner

---

### 4. Run the Application

```powershell
streamlit run app.py
```

---

## 🖥️ GPU Diagnostics

To verify CUDA and PyTorch configuration:

```powershell
python check_gpu.py
```

---

## 📊 Example Workflow

**Input Prompt:**

```
write about ai
```

**Output:**

* Score: *Low quality*
* Refined Prompt:

```
🎯 Objective
You are a senior product manager at a mid-size tech company with deep expertise in emerging technologies.

📋 Requirements
Context: I'm a junior product manager who writes for a high-growth startup.
I need to build a compelling, engaging product that builds on the strengths of my previous product.
Task: Write a concise, actionable product brief that builds upon the strengths of our current product and makes it stand out from the crowd.
📤 Output Format
Format: - The core problem: how to write a compelling product or service description - The 5 most important strengths (with examples) - The 10 most common failures people make - Key strengths (not just strengths) Rules: - Avoid clichés like "good news" as an example - Don't overstate what you're talking about - Be honest about why your product isn't working - Include a short story format that doesn't require a long-form approach - Address the root causes directly Think step by step before answering
```

---

## ⚠️ Important Notes

* `venv/` is excluded via `.gitignore`
* Large model files (`.bin`, `.ckpt`, `.safetensors`) are not tracked
* Store model weights externally (Google Drive / Hugging Face Hub) if needed

---

## 🌱 Future Improvements

* API deployment (FastAPI / Flask)
* Model hosting via Hugging Face Hub
* Prompt history tracking
* Evaluation metrics dashboard

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 🤝 Contribution

Contributions are welcome!
Feel free to fork the repo, open issues, or submit pull requests.

---

## ⭐ Acknowledgements

* Hugging Face Transformers
* Streamlit

---

> Built for improving prompt engineering workflows and making AI outputs more reliable.
