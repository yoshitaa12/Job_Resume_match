# NLP-Based Resume–Job Matching System

## 📌 Overview

The **NLP-Based Resume–Job Matching System** is a Python-based web application that uses Natural Language Processing (NLP) techniques to compare a candidate's resume with a job description.

Users can upload a resume in PDF format and enter a job description to receive a match score, identify matching and missing skills, discover relevant job opportunities, and explore career-related insights.

The application combines traditional NLP techniques and semantic similarity to provide a more informative comparison of resumes and job requirements.

## ✨ Features

- **Resume Parsing:** Extracts text from PDF resumes.
- **Text Preprocessing:** Performs text cleaning, stopword removal, and lemmatization.
- **TF-IDF Similarity:** Measures lexical similarity between resumes and job descriptions.
- **Semantic Similarity:** Uses sentence embeddings to estimate contextual similarity.
- **Skill Extraction:** Identifies relevant technical skills.
- **Skill Gap Analysis:** Displays matching and missing skills.
- **Job Domain Classification:** Classifies job descriptions into domains such as AI/ML, Web Development, Data Science, Cloud, Software Development, and Cybersecurity.
- **Job Description Summarization:** Generates a concise summary of job requirements.
- **Named Entity Recognition:** Extracts named entities using spaCy.
- **Job Search:** Suggests relevant jobs based on skill overlap with the available dataset.
- **Career Assistant:** Answers basic questions about match scores and skill gaps.
- **Question Answering:** Retrieves relevant information from the job description.
- **Visual Dashboard:** Displays match scores and analysis results through a Streamlit interface.

## 🛠️ Technologies Used

- Python
- Streamlit
- Natural Language Toolkit (NLTK)
- spaCy
- scikit-learn
- Sentence Transformers
- Pandas and NumPy
- pdfplumber
- Matplotlib
- TF-IDF and Cosine Similarity
- Logistic Regression

## ⚙️ Project Structure

```text
NLP-Resume-Job-Matching/
│
├── app.py
├── main.py
├── config.py
├── requirements.txt
│
├── modules/
│   ├── preprocessing.py
│   ├── resume_parser.py
│   ├── text_representation.py
│   ├── similarity.py
│   ├── skill_extractor.py
│   ├── keyphrase_extractor.py
│   ├── job_classifier.py
│   ├── embeddings.py
│   ├── score_calculator.py
│   ├── ner_extractor.py
│   ├── summarizer.py
│   ├── job_search.py
│   ├── visualization.py
│   ├── career_chatbot.py
│   └── qa_system.py
│
├── data/
│   ├── job_dataset.csv
│   ├── jobs.csv
│   ├── reviews.csv
│   ├── sample_job.txt
│   └── skills.txt
│
├── assets/
└── requirements.txt
```

## 🚀 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/yoshitaa12/NLP-Resume-Job-Matching.git
cd NLP-Resume-Job-Matching
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

**Windows PowerShell:**

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
streamlit run app.py
```

The application will open in your browser, usually at `http://localhost:8501`.

## 🧠 How It Works

1. The user uploads a resume in PDF format.
2. The user provides the target job description.
3. The system extracts and preprocesses the resume text.
4. TF-IDF vectors and semantic embeddings are used to compare the documents.
5. The system extracts skills and identifies matching and missing requirements.
6. The job description is classified into a supported job domain.
7. Additional modules generate a summary, suggest relevant jobs, and answer career-related questions.
8. Results are presented through the Streamlit dashboard.

## 📊 Match Score Calculation

The overall score combines three components:

- TF-IDF similarity: 20%
- Semantic similarity: 40%
- Skill match percentage: 40%

**Formula:**

`Overall Match = (0.20 × TF-IDF Score) + (0.40 × Semantic Score) + (0.40 × Skill Match)`

The score is a project-specific similarity estimate. It is not a guarantee of recruitment success or interview selection.

## 🎯 Project Objectives

- Apply NLP concepts to a practical recruitment-related problem.
- Compare resume content with job requirements.
- Identify relevant skills and potential skill gaps.
- Explore text representation, classification, information extraction, and semantic similarity.
- Build an interactive web application using Python and Streamlit.

## 🔮 Future Enhancements

- Expand the job-classification dataset.
- Improve skill extraction and normalization.
- Support scanned resumes using OCR.
- Add resume improvement recommendations.
- Evaluate the matching system using larger, representative datasets.
- Improve job recommendations using more comprehensive job data.

## 👩‍💻 Author

**Name:** Your Name  
**Roll Number:** Your Roll Number

## 📄 Disclaimer

This project is intended for academic and educational purposes. Matching scores and job recommendations are estimates based on the implemented NLP techniques and available datasets. They should not be used as the sole basis for recruitment decisions.
