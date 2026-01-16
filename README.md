# 🧠 AI Resume Analyzer
> *Master your career with the power of Artificial Intelligence.*


**AI Resume Analyzer** is a cutting-edge web application designed to help job seekers optimize their resumes for Applicant Tracking Systems (ATS) and modern recruitment standards. 

Unlike traditional keyword counters, this system uses **Advanced Natural Language Processing (NLP)** and **Semantic Analysis** to understand the *content* and *context* of a resume, matching it effectively against specific Job Descriptions (JDs).

![Project Screenshot](assets/screenshot.png)

## ✨ Key Features

### 🤖 Advanced AI Analysis
- **Semantic Matching**: Uses `Sentence-Transformers` (SBERT) to compare the *meaning* of your experience against job requirements, not just exact words.
- **Impact Scoring**: Automatically detects and scores "high-impact" bullet points (e.g., "Increased revenue by 20%").
- **Weakness Detection**: Flags passive language (e.g., "Worked on") and suggests strong action verbs.

### 🎨 Premium User Experience
- **Cyberpunk / Glassmorphism UI**: A visually stunning, dark-mode interface with neon glows and smooth animations.
- **Real-Time Feedback**: Interactive "HUD" elements show analysis progress and capability checks.
- **Responsive Design**: Fully optimized for Desktop, Tablet, and Mobile.

### 📊 Comprehensive Reporting
- **resume_analyzer.py** Core Engine:
    - **ATS Score**: precise 0-100 score based on 6 weighted factors.
    - **Skills Gap Analysis**: Identifies missing hard/soft skills.
    - **Formatting Checks**: Analyzes bullet point usage, sentence length, and readability (Flesch Score).
    - **Grammar & Syntax**: Integrated `LanguageTool` for error detection.

## 🛠️ Tech Stack

### Backend
- **Python 3.10+**: Core language.
- **Flask**: Lightweight web framework.
- **Spacy & NLTK**: NLP for entity recognition (NER) and tokenization.
- **Sentence-Transformers**: For deep semantic similarity search.
- **MongoDB**: NoSQL database for storing scan history and feedback.

### Frontend
- **HTML5 & CSS3**: Custom "Glassmorphism" styling (No heavy CSS frameworks).
- **JavaScript (Vanilla)**: Dynamic DOM manipulation and async file handling.
- **Jinja2**: Templating engine for server-side rendering.

## 🚀 How It Works
1.  **Upload**: User uploads a PDF or DOCX resume.
2.  **Parse**: System extracts text using `pdfminer` or `python-docx`.
3.  **Analyze**: 
    - Text is cleaned and normalized.
    - NLP models extract entities (Skills, Dates, Organizations).
    - Semantic engine compares text vectors with the Job Description.
4.  **Report**: A detailed score and actionable feedback are generated.
5.  **Improve**: User updates their resume based on AI suggestions.

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-resume-analyzer.git

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

## 🤝 Contributing
Contributions are welcome! Please fork the repository and submit a Pull Request.

---
*Built with ❤️ by [Your Name]*
