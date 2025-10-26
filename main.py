from transformers import pipeline
from groq import Groq
from dotenv import load_dotenv
import os
import json
from datetime import datetime
from colorama import init, Fore, Style
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Initialize colorama
init()

# Load environment variables
load_dotenv()

# Files for memory and dataset
MEMORY_FILE = "ai_memory.json"
DATASET_FILE = "alpaca_dataset.jsonl"

# Color definitions
COLORS = {
    'title': Fore.CYAN + Style.BRIGHT,
    'prompt': Fore.GREEN + Style.BRIGHT,
    'response': Fore.WHITE + Style.BRIGHT,
    'info': Fore.YELLOW,
    'error': Fore.RED,
    'reset': Style.RESET_ALL,
    'sentiment': Fore.MAGENTA,
    'thinking': Fore.CYAN
}

# Load memory
def load_memory():
    try:
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, 'r') as f:
                return json.load(f)
        return {"conversation_history": [], "last_updated": str(datetime.now())}
    except Exception as e:
        print(f"{COLORS['error']}Error loading memory: {e}{COLORS['reset']}")
        return {"conversation_history": [], "last_updated": str(datetime.now())}

# Save memory
def save_memory(memory_data):
    try:
        memory_data["last_updated"] = str(datetime.now())
        with open(MEMORY_FILE, 'w') as f:
            json.dump(memory_data, f, indent=2)
    except Exception as e:
        print(f"{COLORS['error']}Error saving memory: {e}{COLORS['reset']}")

# Append a conversation turn to the Alpaca-style dataset
def append_to_dataset(user_prompt, ai_response, input_text=""):
    example = {
        "instruction": user_prompt,
        "input": input_text,
        "output": ai_response
    }
    with open(DATASET_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(example, ensure_ascii=False) + "\n")

# Initialize sentiment and emotion classifiers
sentiment_classifier = None
emotion_classifier = None

def initialize_classifiers():
    global sentiment_classifier, emotion_classifier
    try:
        sentiment_classifier = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1
        )
        emotion_classifier = pipeline(
            "text-classification",
            model="bhadresh-savani/distilbert-base-uncased-emotion",
            device=-1
        )
    except Exception as e:
        print(f"{COLORS['error']}Error initializing classifiers: {e}{COLORS['reset']}")

# Analyze sentiment and emotion
def get_sentiment_and_emotion(text):
    try:
        if not sentiment_classifier or not emotion_classifier:
            return f"{COLORS['error']}(sentiment analysis not initialized){COLORS['reset']}"
        
        sentiment_result = sentiment_classifier(text)[0]
        emotion_result = emotion_classifier(text)[0]
        
        return f"{COLORS['sentiment']}(sentiment: {sentiment_result['label'].lower()}: {sentiment_result['score']:.2%}, emotion: {emotion_result['label']}){COLORS['reset']}"
    except Exception:
        return f"{COLORS['error']}(sentiment analysis unavailable){COLORS['reset']}"

# Get AI response using Groq API
def get_ai_response(prompt, memory):
    try:
        client = Groq(api_key=os.getenv("groq_api_key"))
        
        messages = [
                
                {"role": "system", "content": (
                "You are a friendly, empathetic AI companion. "
                "You chat like a caring friend who is always ready to listen, give advice, joke around, "
                "and provide useful tips in a casual, supportive way. "
                "You remember the conversation context and respond naturally, warmly, and engagingly. "
                "Avoid being overly formal; your goal is to make the user feel comfortable, supported, "
                "and understood, just like a real-life best friend."
            )}
        ]
        
        # Add last conversation if exists
        if memory["conversation_history"]:
            messages.append({"role": "user", "content": memory["conversation_history"][0]["user"]})
            messages.append({"role": "assistant", "content": memory["conversation_history"][0]["assistant"]})
        
        # Add current prompt
        messages.append({"role": "user", "content": prompt})
        
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=800
        )
        
        return chat_completion.choices[0].message.content.strip()
            
    except Exception as e:
        print(f"{COLORS['error']}Error: {str(e)}{COLORS['reset']}")
        return "I apologize, but I'm experiencing technical difficulties."

# Update memory with new conversation
def update_consolidated_memory(memory, new_user_input, new_assistant_response):
    if not memory["conversation_history"]:
        memory["conversation_history"] = [{
            "user": new_user_input,
            "assistant": new_assistant_response,
            "timestamp": str(datetime.now())
        }]
    else:
        memory["conversation_history"][0]["user"] += f", {new_user_input}"
        memory["conversation_history"][0]["assistant"] = f"I am Jarvis4H. {new_assistant_response}"
        memory["conversation_history"][0]["timestamp"] = str(datetime.now())
    return memory

# Main chat loop
def main():
    memory = load_memory()
    
    print(f"\n{COLORS['title']}=== Advanced AI Assistant ==={COLORS['reset']}")
    print(f"{COLORS['info']}Initializing AI models... Please wait...{COLORS['reset']}")
    initialize_classifiers()
    
    print(f"\n{COLORS['info']}Ready! You can start chatting.{COLORS['reset']}")
    print("Type 'quit' to exit")
    print(f"{COLORS['info']}─" * 50 + f"{COLORS['reset']}")
    
    while True:
        user_input = input(f"\n{COLORS['prompt']}You: {COLORS['reset']}").strip()
        
        if not user_input:
            continue
        if user_input.lower() == 'quit':
            print(f"\n{COLORS['info']}Goodbye!{COLORS['reset']}")
            break
        
        # Analyze sentiment/emotion
        analysis = get_sentiment_and_emotion(user_input)
        print(f"Message analysis: {analysis}")
        
        # AI response
        print(f"{COLORS['thinking']}Thinking...{COLORS['reset']}", end='\r')
        ai_response = get_ai_response(user_input, memory)
        print(" " * 20, end='\r')  # clear thinking
        
        # Save memory
        memory = update_consolidated_memory(memory, user_input, ai_response)
        save_memory(memory)
        
        # Append to Alpaca-style dataset
        append_to_dataset(user_input, ai_response)
        
        print(f"\n{COLORS['response']}Assistant: {ai_response}{COLORS['reset']}")
        print(f"{COLORS['info']}─" * 50 + f"{COLORS['reset']}")

if __name__ == "__main__":
    main()
