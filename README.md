# 🌾 Farmer Assistant AI  

![Farmer Assistant Screenshot](templates/bg.png)

## 📌 Project Overview  
Farmer Assistant AI is an intelligent web application built using **LangChain** and **Flask** that helps farmers ask questions in **Hindi** or **English** and get instant answers.  
The app also includes **Voice Typing** (speech-to-text) and **Text-to-Speech** support for better accessibility.  

This project aims to assist farmers with agricultural queries in their own language, with a simple and user-friendly interface.  

---

## 🚀 Features  
- ✅ Ask questions in **Hindi or English**  
- ✅ **Multilingual Support** (automatic detection for Hindi & English)  
- ✅ **Voice Typing** (speech-to-text using browser microphone)  
- ✅ **Text-to-Speech** (listen to AI-generated answers)  
- ✅ **Retriever-based model** to suggest **YouTube links** and references  
- ✅ **LLM response generation** using the model in `model.py`  
- ✅ Simple **Flask backend** to handle queries  
- ✅ Beautiful web UI built with **HTML + Inline CSS + JavaScript**  
- ✅ Clickable links in responses  

---

## 🛠️ Tech Stack  
- **Backend:** Flask (Python)  
- **AI Processing:** LangChain  
  - Retriever for fetching **YouTube recommendations**  
  - LLM (from `model.py`) for **answer generation**  
- **Frontend:** HTML, CSS, JavaScript  
- **Speech Support:** Web Speech API (SpeechRecognition + SpeechSynthesis)  
- **Languages:** Hindi, English  

---

## 📂 Project Structure  
📦 Farmer-Assistant-AI
│── app.py # Flask backend
│── model.py # LangChain retriever + LLM for answers
│── templates/
│ └── index.html # Main UI
│── static/
│ └── bg.png # Background image
│── README.md # Project documentation
│── requirements.txt # Python dependencies


---

## 🎤 Voice Features  

### Voice Input (Speech-to-Text)  
- Click on **🎤 बोलें** button to speak your query  
- The recognized speech will auto-fill the input box  

### Voice Output (Text-to-Speech)  
- Click **🔊 सुनें** to hear the answer  
- Supports **Hindi (hi-IN)** and **English (en-US)** voices  

---

## 📸 Demo Screenshot  
Here’s how it looks in action:  

![Farmer Assistant Screenshot](images/screenshot.png)  

---

## 🌱 Future Improvements  
- Add support for more regional languages  
- Provide offline mode using local models  
- Add FAQs and pre-trained agricultural datasets  
