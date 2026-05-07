from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import joblib
import pandas as pd
import numpy as np
import datetime
import os
from dotenv import load_dotenv

# Load local .env file if it exists
load_dotenv()

try:
    from google import genai as genai_new
    from google.genai import types as genai_types
    _genai_available = True
except Exception:
    genai_new = None
    genai_types = None
    _genai_available = False

# ── Gemini AI Setup ──────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
_genai_client = None
if GEMINI_API_KEY and _genai_available:
    try:
        _genai_client = genai_new.Client(api_key=GEMINI_API_KEY)
    except Exception as _e:
        print(f"Warning: Could not initialise Gemini client: {_e}")

HEART_SYSTEM_PROMPT = """
You are CardioBot, a friendly and knowledgeable AI health assistant embedded in a Heart Disease Risk Prediction portal.
Your role is to help users understand:

1. FORM INPUTS — Explain what each of the 6 user-entered parameters means in simple language:
   - Age: current age in years
   - Sex: biological sex at birth (Male=1, Female=0)
   - Chest Pain Type (cp): 0=Typical Angina, 1=Atypical Angina, 2=Non-anginal, 3=Asymptomatic
   - Resting Blood Pressure (trestbps): mmHg at rest, normal <120
   - Cholesterol (chol): mg/dl from blood test, normal <200
   - Max Heart Rate (thalach): highest BPM during exercise
   Note: The app only collects these 6 inputs. Advanced clinical markers are automatically inferred by the AI model.

2. HEART DISEASE — General education about what heart disease is, types, symptoms, causes.

3. PREVENTION — Diet, exercise, sleep, stress management, medications, lifestyle habits.

4. RECOMMENDATIONS — Personalized advice based on specific risk factors the user mentions.

Rules:
- Always respond in clear, simple, empathetic language — avoid medical jargon.
- Keep responses concise (3–5 sentences or bullet points).
- Never diagnose or replace a real doctor. Always remind users to consult a professional for medical decisions.
- Use emojis sparingly to keep tone friendly.
- If asked something completely unrelated to health/heart disease, politely redirect.
"""

app = Flask(__name__)
app.secret_key = 'super_secret_for_sessions'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///heart.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

CORS(app)
db = SQLAlchemy(app)

# --- Database Models ---
class User(db.Model):
    id = db.Column(db.String(100), primary_key=True) # Google subject ID or Mock ID
    name = db.Column(db.String(100))
    email = db.Column(db.String(100))
    picture = db.Column(db.String(300))

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.String(100), db.ForeignKey('user.id'))
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    
    # Inputs — FULL 13 clinical fields
    age = db.Column(db.Float)
    sex = db.Column(db.Float)
    cp = db.Column(db.Float)
    trestbps = db.Column(db.Float)
    chol = db.Column(db.Float)
    fbs = db.Column(db.Float)
    restecg = db.Column(db.Float)
    thalach = db.Column(db.Float)
    exang = db.Column(db.Float)
    oldpeak = db.Column(db.Float)
    slope = db.Column(db.Float)
    ca = db.Column(db.Float)
    thal = db.Column(db.Float)

    # Result
    prediction_result = db.Column(db.Integer)
    probability = db.Column(db.Float)

# Create tables
with app.app_context():
    db.create_all()

# --- Model Loading ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    model   = joblib.load(os.path.join(BASE_DIR, 'heart_disease_model.pkl'))
    num_imp = joblib.load(os.path.join(BASE_DIR, 'num_imputer.pkl'))
    cat_imp = joblib.load(os.path.join(BASE_DIR, 'cat_imputer.pkl'))
    scr     = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
    enc     = joblib.load(os.path.join(BASE_DIR, 'encoder.pkl'))

except Exception as e:
    print("Warning: Could not load the machine learning model. Make sure to run the pipeline first.", e)

# --- Routes ---
@app.route('/login')
def login():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/auth/mock', methods=['POST'])
def mock_auth():
    # Dev bypass route for user without OAuth Client ID setup yet
    mock_id = 'mock-user-123'
    user = db.session.get(User, mock_id)
    if not user:
        user = User(id=mock_id, name="Test User", email="test@example.com", picture="https://ui-avatars.com/api/?name=Test+User&background=3b82f6&color=fff")
        db.session.add(user)
        db.session.commit()
    
    session['user_id'] = user.id
    return jsonify({'success': True})

@app.route('/auth/login', methods=['POST'])
def simple_login():
    data = request.json
    name = data.get('name', '').strip()
    if not name:
        return jsonify({'success': False, 'error': 'Name is required'})
    
    # Create a deterministic ID based on the name for simplicity in this prototype
    user_id = f"user_{name.lower().replace(' ', '_')}"
    
    user = db.session.get(User, user_id)
    if not user:
        # Default cool avatar for all users
        picture = f"https://ui-avatars.com/api/?name={name.replace(' ', '+')}&background=0f172a&color=3b82f6&size=128&bold=true"
        user = User(id=user_id, name=name, email=f"{name.lower().replace(' ', '')}@portal.local", picture=picture)
        db.session.add(user)
        db.session.commit()
    
    session['user_id'] = user.id
    return jsonify({'success': True})

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))

@app.route('/')
def home():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return redirect(url_for('dashboard'))

@app.route('/index')
def index_page():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user = db.session.get(User, session['user_id'])
    if not user:
        session.clear()
        return redirect(url_for('login'))
        
    history = Prediction.query.filter_by(user_id=user.id).order_by(Prediction.timestamp.desc()).all()
    return render_template('dashboard.html', user=user, history=history)

@app.route('/predict_page')
def predict_page():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user = db.session.get(User, session['user_id'])
    if not user:
        session.clear()
        return redirect(url_for('login'))
    return render_template('predict.html', user=user)

def generate_health_insights(data, pred, prob):
    # Prepare patient data for the prompt
    age = data.get('age', 'N/A')
    sex = "Male" if data.get('sex') == 1 else "Female"
    cp_types = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
    cp = cp_types[int(data.get('cp', 0))] if data.get('cp') is not None else 'N/A'
    bps = data.get('trestbps', 'N/A')
    chol = data.get('chol', 'N/A')
    mhr = data.get('thalach', 'N/A')
    
    risk_status = "High Risk" if pred == 1 else "Low Risk"
    risk_prob = round(float(prob) * 100, 1)

    prompt = f"""
    The user has just received a heart disease risk prediction.
    RESULT: {risk_status} ({risk_prob}% Probability)
    
    PATIENT VITALS:
    - Age: {age}
    - Sex: {sex}
    - Chest Pain: {cp}
    - Blood Pressure: {bps} mmHg
    - Cholesterol: {chol} mg/dl
    - Max Heart Rate: {mhr} BPM
    
    Please generate a personalized health report in HTML format.
    Include the following sections:
    1. <h3>AI Risk Explanation</h3> - A simple, empathetic 2-3 sentence explanation of what these specific clinical numbers (including markers like ST depression/Oldpeak if high) mean for their heart health.
    2. <h3>Personalized Preventive Actions</h3> - 3 bullet points targeting their highest risk factors (e.g., if BP is high, focus on salt/stress; if clinical markers like ST slope or major vessels are concerning, explain what that implies).
    3. <h3>Your Tailored Daily Routine</h3> - A 3-part routine covering Diet, Physical Activity, and Stress Management.
    
    RULES:
    - Use clean HTML (h3, p, ul, li, strong).
    - Be empathetic but professional.
    - DO NOT include a title like "Health Report" at the top.
    - End with a small italic disclaimer: "This is an AI-generated insight and not a medical diagnosis. Please consult a doctor for clinical decisions."
    """

    # --- Try Gemini AI ---
    if _genai_client and GEMINI_API_KEY:
        try:
            response = _genai_client.models.generate_content(
                model='gemini-2.0-flash',
                contents=[prompt],
                config=genai_types.GenerateContentConfig(
                    system_instruction=HEART_SYSTEM_PROMPT,
                    temperature=0.7,
                    max_output_tokens=800,
                )
            )
            if response.text:
                # Basic cleaning in case of markdown blocks
                reply = response.text.replace('```html', '').replace('```', '').strip()
                return reply
        except Exception as e:
            print(f"Gemini Insights Error: {e}")

    # --- Fallback: Template-based logic if AI fails ---
    html = f"<h3>AI Preventive Insights (Standard)</h3>"
    if pred == 1:
        html += f"<p>Based on your vitals (BP: {bps}, Chol: {chol}), you fall into a higher risk category ({risk_prob}%).</p>"
        html += "<ul>"
        if float(chol or 0) > 200: html += "<li>Focus on reducing saturated fats and increasing fiber to lower cholesterol.</li>"
        if float(bps or 0) > 130: html += "<li>Monitor your salt intake and practice relaxation techniques to lower blood pressure.</li>"
        html += "<li>Schedule a consultation with a cardiologist for a thorough check-up.</li>"
        html += "</ul>"
    else:
        html += f"<p>Great news! Your risk profile is low ({risk_prob}%). Keep maintaining your healthy habits.</p>"
        html += "<ul><li>Maintain a balanced diet and aim for 150 mins of exercise weekly.</li><li>Continue regular annual health screenings.</li></ul>"
    
    html += "<p><em>Disclaimer: This is a template-based insight. Consult a professional for medical advice.</em></p>"
    return html

@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401

    try:
        data = request.json
        # The 13 clinical inputs
        user_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
        cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

        # Initialize with NaNs to ensure all columns exist for the imputer
        patient_data = {col: np.nan for col in user_cols}
        
        # Fill with actual data if provided
        for k in user_cols:
            if k in data:
                v = data.get(k)
                if v != "" and v is not None:
                    try:
                        patient_data[k] = float(v)
                    except ValueError:
                        pass # Keep as NaN if not a number

        df_patient = pd.DataFrame([patient_data])
        # Ensure correct column order for transformers
        df_patient = df_patient[num_cols + cat_cols]

        # Preprocess through trained pipeline
        df_num = pd.DataFrame(num_imp.transform(df_patient[num_cols]), columns=num_cols)
        df_cat = pd.DataFrame(cat_imp.transform(df_patient[cat_cols]), columns=cat_cols)
        df_num_scaled = pd.DataFrame(scr.transform(df_num), columns=num_cols)
        df_cat_encoded = pd.DataFrame(enc.transform(df_cat), columns=enc.get_feature_names_out(cat_cols))

        X_unseen = pd.concat([df_num_scaled, df_cat_encoded], axis=1)

        pred = model.predict(X_unseen)[0]
        prob = model.predict_proba(X_unseen)[0, 1]

        # Save all 13 collected inputs to the database
        db_prediction = Prediction(
            user_id=session['user_id'],
            age=patient_data.get('age'),
            sex=patient_data.get('sex'),
            cp=patient_data.get('cp'),
            trestbps=patient_data.get('trestbps'),
            chol=patient_data.get('chol'),
            fbs=patient_data.get('fbs'),
            restecg=patient_data.get('restecg'),
            thalach=patient_data.get('thalach'),
            exang=patient_data.get('exang'),
            oldpeak=patient_data.get('oldpeak'),
            slope=patient_data.get('slope'),
            ca=patient_data.get('ca'),
            thal=patient_data.get('thal'),
            prediction_result=int(pred),
            probability=float(prob)
        )
        db.session.add(db_prediction)
        db.session.commit()

        insights_html = generate_health_insights(patient_data, int(pred), float(prob))

        return jsonify({
            'success': True,
            'prediction': int(pred),
            'probability': float(prob),
            'insights': insights_html
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/prediction/<int:pred_id>/delete', methods=['POST'])
def delete_prediction(pred_id):
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401
    record = db.session.get(Prediction, pred_id)
    if not record:
        return jsonify({'success': False, 'error': 'Not found'}), 404
    if record.user_id != session['user_id']:
        return jsonify({'success': False, 'error': 'Forbidden'}), 403
    db.session.delete(record)
    db.session.commit()
    return jsonify({'success': True})

@app.route('/chat', methods=['POST'])
def chat():
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401

    data = request.json
    user_message = data.get('message', '').strip()
    history = data.get('history', [])  # list of {role, content}

    if not user_message:
        return jsonify({'success': False, 'error': 'Empty message'}), 400

    if not GEMINI_API_KEY or not _genai_client:
        return jsonify({
            'success': False,
            'error': '⚙️ CardioBot needs a Gemini API key to work. Please add your GEMINI_API_KEY to the .env file and restart the server.'
        }), 500

    # Try models in order of preference, fall back if one is rate-limited
    MODELS_TO_TRY = ['gemini-2.0-flash', 'gemini-1.5-flash', 'gemini-1.5-pro']

    # Build full conversation contents
    contents = []
    for msg in history[-10:]:
        role = 'user' if msg['role'] == 'user' else 'model'
        contents.append(genai_types.Content(role=role, parts=[genai_types.Part(text=msg['content'])]))
    contents.append(genai_types.Content(role='user', parts=[genai_types.Part(text=user_message)]))

    last_error = None
    for model_name in MODELS_TO_TRY:
        try:
            response = _genai_client.models.generate_content(
                model=model_name,
                contents=contents,
                config=genai_types.GenerateContentConfig(
                    system_instruction=HEART_SYSTEM_PROMPT,
                    temperature=0.7,
                    max_output_tokens=512,
                )
            )
            reply = response.text
            if reply:
                return jsonify({'success': True, 'reply': reply})
        except Exception as e:
            last_error = str(e)
            print(f"CardioBot: Model {model_name} failed — {last_error}")
            # Only try next model if this was a rate-limit or availability error
            if '429' not in last_error and '503' not in last_error and 'quota' not in last_error.lower():
                break  # Non-rate-limit error, don't retry other models

    # All models failed
    import traceback
    traceback.print_exc()
    if last_error and ('429' in last_error or 'quota' in last_error.lower()):
        friendly = "😴 CardioBot is busy right now (API rate limit reached). Please wait 30 seconds and try again!"
    elif last_error and 'API_KEY' in last_error.upper():
        friendly = "🔑 Your Gemini API key appears to be invalid. Please check the key in your .env file."
    else:
        friendly = f"⚠️ CardioBot encountered an error. Please try again in a moment."

    return jsonify({'success': False, 'error': friendly}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5001)
