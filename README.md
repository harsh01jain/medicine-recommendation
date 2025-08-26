# CureWise

CureWise is a Flask-based medicine recommendation system that combines handwriting OCR, multiple machine-learning models, and an LLM (DeepSeek-R1 via Ollama) to provide disease predictions, medication suggestions, nutrition and workout plans, and document analysis. The full project report (blackbook) is included in this repository and covers design diagrams, test cases, and implementation details.

---

## Key features

- Disease prediction (three severity levels: Mild / Moderate / Severe) using trained Keras models.  
- Medicine recommendation module that suggests likely medications for predicted diseases.  
- Nutrition plan generation (concise diet suggestions) and workout plan suggestions.  
- OCR and medical document analysis: upload handwritten prescriptions (image/PDF) → extract text → highlight medical terms and summarize. :contentReference[oaicite:1]{index=1}  
- Chatbot interface (DeepSeek-R1 via Ollama) for follow-up questions and concise nutrition/diet answers. :contentReference[oaicite:2]{index=2}  
- Medical text analyzer: spell correction, term-highlighting, and summary generation. :contentReference[oaicite:3]{index=3}  
- Webcam image capture and live MJPEG video streaming endpoints (used for capture/verification). :contentReference[oaicite:4]{index=4}  
- User management: signup/login, email confirmation, password reset using JWT tokens.  
- Dashboard giving quick access to all modules (prediction, recommender, OCR, chatbot, analyzer). :contentReference[oaicite:5]{index=5}

---

## Quickstart (developer)
### 1. Clone
```bash
git clone https://github.com/harsh01jain/medicine-recommendation.git
cd curewise
```
### 2. Create Venv and install
```bash
python -m venv venv
# Activate the venv:
# Linux / Mac
source venv/bin/activate
# Windows (PowerShell)
venv\Scripts\Activate.ps1
```
### 3. Install the requirements
```bash
pip install -r requirements.txt
```
### 4. Copy .env.example to .env and fill in real values:
```bash
cp .env.example .env
Notes:
-Keep .env private and do not commit it. Add it to .gitignore.
-Use consistent uppercase names (as shown) to avoid mismatches.
```

### 5. Ollama and DeepSeek-R1 (required)
```bash
The app uses the DeepSeek-R1 7B model via Ollama. The model is not included in the repo.

Install Ollama (follow OS instructions): https://ollama.ai/download

Start Ollama commands(cmd):
1.ollama serve
2.Pull the model:
   ollama pull deepseek-r1:7b
(Optional) Quick test:
ollama run deepseek-r1:7b "Hello world"

-The application expects Ollama at the URL set by OLLAMA_URL (default http://127.0.0.1:11434). 
```
### 6. Database setup
```bash
Create the MySQL database:

CREATE DATABASE curewise;
-copy the structure mentioned in Database structure file
-- then grant privileges if needed
--also Update MYSQL_* variables in .env.

Run the application
# ensure venv and .env are active
python app.py


Open: http://127.0.0.1:5000 (default). If you set FLASK_DEBUG=true in .env, use caution — do not enable in production.
```
