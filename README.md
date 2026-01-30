# 🪄 NLP Language Model: LSTM Text Generator 🪄

Welcome to my **Natural Language Understanding** project! This repository contains a deep learning model capable of generating human-like news summaries using an LSTM architecture.

[Hugg<img width="1915" height="800" alt="Screenshot 2026-01-30 at 6 19 11 pm" src="https://github.com/user-attachments/assets/77f2d196-d02b-42af-8bda-38328302ffd1" />
ing Face: CNN/DailyMail](https://huggingface.co/datasets/abisee/cnn_dailymail)

---

## 📚 Task 1: Dataset Acquisition
We used the **CNN/DailyMail (Version 2.0.0)** dataset, a staple in the NLP world for summarization and language modeling.

* **Source:** 📰 
* **Size:** Filtered to **10,000 samples** for efficient training.
* **Credit:** Huge thanks to **Abigail See et al.** for their work on *Pointer-Generator Networks* which popularized this dataset.

---

## 🧠 Task 2: Model Architecture & Training
I built a multi-layer LSTM to capture the "vibe" and structure of news reporting.

### 🏗️ The Setup
* **Layers:** 3 Stacked LSTMs
* **Hidden Dimension:** 512
* **Dropout:** 0.65 (to prevent the model from just "memorizing" the news)
* **Vocabulary:** 14,174 unique tokens 🏷️

### 📊 Training Summary
| Metric 📈 | Value 💎 |
| :--- | :--- |
| **Device Used** | 💻 CPU |
| **Training Loss** | 5.4306 (calculated from final perplexity) |
| **Training Perplexity** | 228.283 |
| **Total Trainable Params** | 20,832,094 |

---

## 💻 Task 3: Interactive Web App
The project includes a **Flask** web application so you can talk to the model yourself!

* **Front-end:** HTML5/CSS3 with a clean, modern UI.
* **Back-end:** Flask (Python) connecting the user's prompt to the PyTorch model.
* **Feature:** Adjustable **Temperature** (0.1 - 1.5) to control how "creative" or "random" the model gets. 🧪

https://github.com/user-attachments/assets/7f14bc5e-5c96-46a2-b851-2a487f8b1549



---

## 🛠️ How to Run Locally

1. **Clone the repo:**
   ```bash
   git clone [https://github.com/Santhosh01161/NLP_Language_Model.git](https://github.com/Santhosh01161/NLP_Language_Model.git)
