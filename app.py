import os
import shutil
import re
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from analyzer.resume_parser import extract_text_from_pdf, extract_text_from_docx
from analyzer.resume_analyzer import compute_score, update_calibration
from datetime import datetime
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()

# --- MongoDB Setup ---
try:
    from pymongo import MongoClient
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    client = MongoClient(mongo_uri)
    db = client["resume_db"]
    scans_collection = db["scans"]
    print("✅ Connected to MongoDB.")
except Exception as e:
    print(f"⚠️ MongoDB Connection Failed: {e}")
    scans_collection = None

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev_secret_key")

# Enable Jinja2 'do' extension for list manipulation in templates
app.jinja_env.add_extension('jinja2.ext.do')

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Routes ---

@app.route('/')
def index():
    """Landing Page"""
    return render_template('landing.html')

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    """Resume Upload & Analysis Page"""
    if request.method == 'POST':
        # Check if file is provided
        if 'resume' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['resume']
        
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # --- Analysis ---
            try:
                # 1. Extract Text
                if filename.lower().endswith('.pdf'):
                    text = extract_text_from_pdf(filepath)
                else:
                    text = extract_text_from_docx(filepath)
                
                # 2. Get Job Description (if any)
                jd_text = request.form.get('jd_text', "")

                # 3. Analyze
                result = compute_score(text, jd_text)
                
                # --- Persistence (MongoDB) ---
                if scans_collection is not None:
                    # Extract Contact Info for DB
                    email = re.search(r'[\w\.-]+@[\w\.-]+', text)
                    phone = re.search(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text)
                    
                    scan_data = {
                        "filename": filename,
                        "candidate_email": email.group(0) if email else "N/A",
                        "candidate_phone": phone.group(0) if phone else "N/A",
                        "total_score": result['total_score'],
                        "breakdown": result['breakdown'],
                        "timestamp": datetime.now(),
                        "has_jd": False
                    }
                    scans_collection.insert_one(scan_data)
                    print(f"💾 Saved scan for {filename} to MongoDB.")

                return render_template('result.html', r=result, filename=filename)

            except Exception as e:
                import traceback
                traceback.print_exc() # Print full stack trace to console
                print(f"❌ ERROR: {e}")
                flash(f'Error processing file: {str(e)}')
                return redirect(request.url)

    # Get total scans for display
    total_scans = 12408 # Fallback/Default
    if scans_collection is not None:
        try:
            total_scans = scans_collection.count_documents({})
        except:
            pass
            
    return render_template('upload.html', total_scans=total_scans)

@app.route('/dashboard')
def dashboard():
    if scans_collection is None:
        return render_template('dashboard.html', scans=[]) 
    
    # Fetch all scans, sorted by newest
    scans = list(scans_collection.find().sort("timestamp", -1))
    return render_template('dashboard.html', scans=scans)

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.json
    feedback_type = data.get('type')
    if feedback_type in ['too_low', 'accurate', 'too_high']:
        new_calibration = update_calibration(feedback_type)
        return jsonify({"status": "success", "new_calibration": new_calibration})
    return jsonify({"status": "error", "message": "Invalid type"}), 400

    
# --- Production Readiness: Preload Models ---
# This ensures models are loaded when Gunicorn starts workers, not on the first request.
from analyzer.resume_analyzer import ModelManager
print("🚀 Starting AI Resume Analyzer...")
ModelManager.preload_models()

if __name__=="__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
