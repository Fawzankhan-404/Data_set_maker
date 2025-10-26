# 🧠 Data_set_maker – Local AI Chat to Alpaca Dataset Generator  
by **Fawzankhan-404**

**Data_set_maker** is a privacy-focused Python-based AI assistant that not only chats with you but also automatically builds a **fine-tuning-ready dataset** in **Alpaca format**. Perfect for anyone looking to train or fine-tune their own LLM using real conversational data.

---

## ✅ What This Project Does

✔️ Lets you chat with an AI assistant (Groq / LLaMA 3.3)  
✔️ Detects user **sentiment & emotion** using Hugging Face Transformers  
✔️ Saves memory for continuity  
✔️ Automatically logs each conversation into an **Alpaca-style JSONL dataset**  
✔️ Creates training-ready data for fine-tuning your own model  
✔️ 100% local + private (no data leaks)

---

## 📂 Dataset Format (Alpaca JSONL)

Each entry inside `alpaca_dataset.jsonl` will look like this:

```json
{
  "instruction": "User prompt",
  "input": "",
  "output": "AI response"
}
```

📦 Data_set_maker
```
├── main.py                     # Main AI logic
├── ai_memory.json              # Assistant memory (auto-updated)
├── alpaca_dataset.jsonl        # Exported dataset for fine-tuning
├── requirements.txt            # Dependencies
└── .env                        # Groq API key (not included)
```
🚀 Setup & Run
1️⃣ Clone the repo 
```
git clone https://github.com/Fawzankhan-404/Data_set_maker.git
cd Data_set_maker
```
2️⃣ Install requirements
```
pip install -r requirements.txt
```
3️⃣ Add your Groq API key

Create a .env file:
```
groq_api_key=YOUR_API_KEY_HERE
```
4️⃣ Run the assistant
```
python main.py
```

💬 Start chatting — every conversation will generate dataset entries automatically in alpaca_dataset.jsonl!

📦 Features Overview-Feature	Status
```
Local chat-based AI	✅
Groq LLaMA response generator	✅
Sentiment + Emotion analysis	✅
Memory persistence	✅
Alpaca-style dataset generation	✅
Fine-tuning ready	✅

```

🏗️ Ideal For
```
✅ AI researchers
✅ LLM fine-tuning beginners
✅ Open-source builders
✅ People building custom chatbots
✅ Dataset collectors
```

🤝 Contributing 

Pull requests, issues, and suggestions are welcome! Let’s make dataset collection easier for everyone.

⭐ Support

If this helped you, a ⭐ on the repo helps support future open-source updates!

👨‍💻 Author

Fawzan (404)
🔗 GitHub: [Fawzankhan-404](https://fawzankhan-404.github.io/)
