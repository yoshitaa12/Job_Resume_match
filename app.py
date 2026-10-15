import streamlit as st

from modules.resume_parser import extract_text_from_pdf
from modules.preprocessing import preprocess_text
from modules.text_representation import create_tfidf_vectors
from modules.similarity import calculate_similarity
from modules.skill_extractor import (
    load_skills,
    extract_skills,
    compare_skills
)
from modules.keyphrase_extractor import extract_keyphrases
from modules.job_classifier import train_job_classifier, classify_job
from modules.embeddings import calculate_semantic_similarity
from modules.score_calculator import calculate_overall_score
from modules.ner_extractor import extract_entities
from modules.summarizer import summarize_text
from modules.job_search import search_jobs
from modules.visualization import create_score_chart
from modules.career_chatbot import career_chatbot
from modules.qa_system import answer_question


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="NLP Resume–Job Matcher",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    /* ---------- Global ---------- */
    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }

    /* ---------- Hero ---------- */
    .hero {
        padding: 34px 36px;
        border-radius: 24px;
        background: linear-gradient(135deg, #f4f7ff 0%, #eefaf8 100%);
        border: 1px solid #e5eaf5;
        margin-bottom: 24px;
    }

    .hero-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        background: #ffffff;
        border: 1px solid #dfe5f2;
        font-size: 13px;
        font-weight: 600;
        color: #566074;
        margin-bottom: 12px;
    }

    .hero h1 {
        font-size: 42px;
        line-height: 1.1;
        margin: 0;
        color: #182033;
    }

    .hero p {
        font-size: 17px;
        color: #667085;
        margin: 12px 0 0 0;
        max-width: 760px;
    }

    /* ---------- Section headings ---------- */
    .section-heading {
        font-size: 25px;
        font-weight: 700;
        color: #182033;
        margin: 28px 0 12px 0;
    }

    .section-subtitle {
        color: #667085;
        font-size: 14px;
        margin-top: -5px;
        margin-bottom: 15px;
    }

    /* ---------- Input cards ---------- */
    .input-label {
        font-size: 18px;
        font-weight: 700;
        color: #182033;
        margin-bottom: 5px;
    }

    .input-hint {
        color: #667085;
        font-size: 13px;
        margin-bottom: 8px;
    }

    /* ---------- Score cards ---------- */
    .score-card {
        padding: 18px 16px;
        border-radius: 18px;
        border: 1px solid #e7eaf0;
        background: #ffffff;
        box-shadow: 0 4px 16px rgba(16, 24, 40, 0.05);
        text-align: center;
        min-height: 105px;
    }

    .score-label {
        color: #667085;
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 7px;
    }

    .score-value {
        color: #182033;
        font-size: 29px;
        font-weight: 800;
    }

    /* ---------- Result cards ---------- */
    .result-card {
        padding: 18px 20px;
        border-radius: 18px;
        border: 1px solid #e7eaf0;
        background: #ffffff;
        box-shadow: 0 3px 14px rgba(16, 24, 40, 0.04);
        margin-bottom: 12px;
    }

    .result-title {
        font-size: 17px;
        font-weight: 700;
        color: #182033;
        margin-bottom: 6px;
    }

    .result-text {
        color: #667085;
        font-size: 14px;
        line-height: 1.55;
    }

    /* ---------- Skill pills ---------- */
    .skill-pill {
        display: inline-block;
        padding: 8px 13px;
        margin: 4px 5px 4px 0;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 600;
    }

    .skill-match {
        background: #eaf8f0;
        color: #137333;
        border: 1px solid #ccebd8;
    }

    .skill-missing {
        background: #fff0f0;
        color: #b42318;
        border: 1px solid #f4cccc;
    }

    /* ---------- Job cards ---------- */
    .job-card {
        padding: 18px 20px;
        border-radius: 18px;
        border: 1px solid #e7eaf0;
        background: #ffffff;
        margin-bottom: 12px;
        box-shadow: 0 3px 14px rgba(16, 24, 40, 0.04);
    }

    .job-title {
        font-size: 18px;
        font-weight: 750;
        color: #182033;
    }

    .job-company {
        color: #667085;
        font-size: 14px;
        margin-top: 3px;
    }

    .job-score {
        font-size: 18px;
        font-weight: 800;
        color: #4353ff;
    }

    /* ---------- Footer ---------- */
    .footer {
        text-align: center;
        color: #98a2b3;
        font-size: 12px;
        padding-top: 28px;
    }

    /* ---------- Streamlit cleanup ---------- */
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e7eaf0;
        padding: 14px;
        border-radius: 16px;
        box-shadow: 0 3px 14px rgba(16, 24, 40, 0.04);
    }

    button[kind="primary"] {
        border-radius: 12px;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">NLP • AI • Resume Intelligence</div>
        <h1>📄 Resume–Job Matcher</h1>
        <p>
            Compare a resume with any job description using TF-IDF,
            semantic embeddings, skill extraction, classification,
            and NLP-based analysis.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# INPUT SECTION
# ============================================================

st.markdown(
    '<div class="section-heading">Analyze a Job Opportunity</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'Upload your resume and paste the job description to get a detailed match report.'
    '</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown(
        '<div class="input-label">📄 Your Resume</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="input-hint">Upload a text-based PDF resume.</div>',
        unsafe_allow_html=True
    )

    resume_file = st.file_uploader(
        "Resume PDF",
        type=["pdf"],
        label_visibility="collapsed"
    )

    if resume_file:
        st.success(f"✓ {resume_file.name}")

with col2:
    st.markdown(
        '<div class="input-label">💼 Job Description</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="input-hint">Paste the complete job description.</div>',
        unsafe_allow_html=True
    )

    job_text = st.text_area(
        "Job Description",
        height=205,
        placeholder=(
            "Example:\n"
            "We are looking for an AI/ML Engineer...\n\n"
            "Requirements:\n"
            "Python, SQL, Machine Learning, NLP..."
        ),
        label_visibility="collapsed"
    )

st.write("")

analyze_clicked = st.button(
    "🚀  Analyze Resume & Job Match",
    type="primary",
    use_container_width=True
)


# ============================================================
# ANALYSIS
# ============================================================

if analyze_clicked:

    if resume_file is None:
        st.warning("Please upload your resume PDF.")

    elif not job_text.strip():
        st.warning("Please paste a job description.")

    else:

        with st.spinner("Analyzing your resume and job description..."):

            # 1. Extract resume text
            raw_resume = extract_text_from_pdf(resume_file)
            raw_job = job_text

            if not raw_resume.strip():
                st.error(
                    "No readable text was found in the PDF. "
                    "Please upload a text-based PDF."
                )
                st.stop()

            # 2. NER
            resume_entities = extract_entities(raw_resume)
            job_entities = extract_entities(raw_job)

            # 3. Skills
            skills = load_skills("data/skills.txt")

            resume_skills = extract_skills(
                raw_resume,
                skills
            )

            job_skills = extract_skills(
                raw_job,
                skills
            )

            (
                matching_skills,
                missing_skills,
                skill_match_percentage
            ) = compare_skills(
                resume_skills,
                job_skills
            )

            # 4. Preprocessing
            resume_tokens = preprocess_text(raw_resume)
            job_tokens = preprocess_text(raw_job)

            clean_resume = " ".join(resume_tokens)
            clean_job = " ".join(job_tokens)

            # 5. TF-IDF
            _, vectors = create_tfidf_vectors(
                clean_resume,
                clean_job
            )

            match_score = round(
                calculate_similarity(vectors),
                2
            )

            # 6. Semantic similarity
            semantic_score = round(
                calculate_semantic_similarity(
                    raw_resume,
                    raw_job
                ),
                2
            )

            # 7. Overall score
            overall_score = round(
                calculate_overall_score(
                    match_score,
                    semantic_score,
                    skill_match_percentage
                ),
                2
            )

            # 8. Keyphrases
            resume_keyphrases = extract_keyphrases(
                raw_resume,
                top_n=10
            )

            job_keyphrases = extract_keyphrases(
                raw_job,
                top_n=10
            )

            # 9. Job classification
            classifier_vectorizer, classifier_model = (
                train_job_classifier(
                    "data/job_dataset.csv"
                )
            )

            job_category = classify_job(
                raw_job,
                classifier_vectorizer,
                classifier_model
            )

            # 10. Job summary
            job_summary = summarize_text(
                raw_job,
                num_sentences=3
            )

            # 11. Job retrieval
            matching_jobs = search_jobs(
                resume_skills,
                "data/jobs.csv",
                top_n=5
            )

            # 12. Visualization
            score_chart = create_score_chart(
                match_score,
                semantic_score,
                skill_match_percentage,
                overall_score
            )

            # 13. Save results
            st.session_state["analysis_done"] = True
            st.session_state["overall_score"] = overall_score
            st.session_state["match_score"] = match_score
            st.session_state["semantic_score"] = semantic_score
            st.session_state["skill_match_percentage"] = (
                skill_match_percentage
            )

            st.session_state["matching_skills"] = matching_skills
            st.session_state["missing_skills"] = missing_skills

            st.session_state["job_category"] = job_category
            st.session_state["job_summary"] = job_summary

            st.session_state["resume_keyphrases"] = (
                resume_keyphrases
            )
            st.session_state["job_keyphrases"] = (
                job_keyphrases
            )

            st.session_state["resume_entities"] = (
                resume_entities
            )
            st.session_state["job_entities"] = (
                job_entities
            )

            st.session_state["matching_jobs"] = matching_jobs
            st.session_state["raw_job"] = raw_job
            st.session_state["score_chart"] = score_chart

        st.success("Analysis completed successfully! 🎉")


# ============================================================
# RESULTS
# ============================================================

if st.session_state.get("analysis_done", False):

    st.divider()

    st.markdown(
        '<div class="section-heading">🎯 Match Overview</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # SCORE CARDS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4, gap="medium")

    with col1:
        st.markdown(
            f"""
            <div class="score-card">
                <div class="score-label">🎯 Overall Match</div>
                <div class="score-value">
                    {st.session_state['overall_score']:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="score-card">
                <div class="score-label">📝 TF-IDF Similarity</div>
                <div class="score-value">
                    {st.session_state['match_score']:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="score-card">
                <div class="score-label">🧠 Semantic Similarity</div>
                <div class="score-value">
                    {st.session_state['semantic_score']:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            f"""
            <div class="score-card">
                <div class="score-label">🛠️ Skill Match</div>
                <div class="score-value">
                    {st.session_state['skill_match_percentage']:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # CHART
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-heading">📊 Matching Analysis</div>',
        unsafe_allow_html=True
    )

    chart_col1, chart_col2 = st.columns([2.2, 1], gap="large")

    with chart_col1:
        st.pyplot(
            st.session_state["score_chart"],
            use_container_width=True
        )

    with chart_col2:
        st.markdown(
            """
            <div class="result-card">
                <div class="result-title">How the score works</div>
                <div class="result-text">
                    <b>20%</b> TF-IDF similarity<br>
                    <b>40%</b> semantic similarity<br>
                    <b>40%</b> skill overlap
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.caption(
            "The score is an NLP-based similarity measure, "
            "not a hiring prediction."
        )

    # --------------------------------------------------------
    # DOMAIN + SUMMARY
    # --------------------------------------------------------

    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        st.markdown(
            '<div class="section-heading">🧠 Job Domain</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-title">
                    {st.session_state["job_category"].replace("_", " ")}
                </div>
                <div class="result-text">
                    Predicted from the job description using
                    TF-IDF features and Logistic Regression.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            '<div class="section-heading">📝 Job Summary</div>',
            unsafe_allow_html=True
        )

        with st.container(border=True):
            for sentence in st.session_state["job_summary"]:
                st.write(f"• {sentence}")

    # --------------------------------------------------------
    # SKILLS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-heading">🛠️ Skill Analysis</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown(
            '<div class="result-title">✅ Matching Skills</div>',
            unsafe_allow_html=True
        )

        if st.session_state["matching_skills"]:

            pills = ""

            for skill in st.session_state["matching_skills"]:
                pills += (
                    f'<span class="skill-pill skill-match">'
                    f'✓ {skill}</span>'
                )

            st.markdown(
                f'<div class="result-card">{pills}</div>',
                unsafe_allow_html=True
            )

        else:
            st.info("No matching skills detected.")

    with col2:
        st.markdown(
            '<div class="result-title">❌ Missing Skills</div>',
            unsafe_allow_html=True
        )

        if st.session_state["missing_skills"]:

            pills = ""

            for skill in st.session_state["missing_skills"]:
                pills += (
                    f'<span class="skill-pill skill-missing">'
                    f'✗ {skill}</span>'
                )

            st.markdown(
                f'<div class="result-card">{pills}</div>',
                unsafe_allow_html=True
            )

        else:
            st.success("No major missing skills detected.")

    # --------------------------------------------------------
    # MATCHING JOBS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-heading">💼 Matching Jobs</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Jobs from the local sample dataset ranked by skill overlap.'
        '</div>',
        unsafe_allow_html=True
    )

    if st.session_state["matching_jobs"]:

        for job in st.session_state["matching_jobs"]:

            with st.container(border=True):

                col1, col2 = st.columns([4, 1])

                with col1:
                    st.markdown(
                        f"### {job['title']}"
                    )
                    st.caption(job["company"])
                    st.write(job["description"])

                    if job["matching_skills"]:
                        st.write(
                            "**Matching:** "
                            + ", ".join(job["matching_skills"])
                        )

                with col2:
                    st.metric(
                        "Skill Match",
                        f"{job['match_percentage']:.0f}%"
                    )

    else:
        st.info(
            "No jobs with sufficient skill overlap were found."
        )

    # --------------------------------------------------------
    # CAREER ASSISTANT
    # --------------------------------------------------------

    st.divider()

    st.markdown(
        '<div class="section-heading">💬 Career Assistant</div>',
        unsafe_allow_html=True
    )

    chatbot_question = st.text_input(
        "Ask about your match",
        placeholder=(
            "Example: What skills am I missing?"
        ),
        key="career_question"
    )

    if chatbot_question:

        response = career_chatbot(
            chatbot_question,
            st.session_state["overall_score"],
            st.session_state["job_category"],
            st.session_state["matching_skills"],
            st.session_state["missing_skills"]
        )

        st.info(response)

    # --------------------------------------------------------
    # JOB QA
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-heading">❓ Ask About the Job</div>',
        unsafe_allow_html=True
    )

    qa_question = st.text_input(
        "Ask a question about the job description",
        placeholder=(
            "Example: What cloud technology is mentioned?"
        ),
        key="job_question"
    )

    if qa_question:

        answer = answer_question(
            qa_question,
            st.session_state["raw_job"]
        )

        st.info(answer)


# ============================================================
# EMPTY STATE
# ============================================================

else:

    st.write("")

    st.markdown(
        """
        <div class="result-card">
            <div class="result-title">✨ What you will get</div>
            <div class="result-text">
                <b>🎯 Match Score</b> — combined NLP similarity analysis<br>
                <b>🛠️ Skill Analysis</b> — matching and missing skills<br>
                <b>🧠 Job Classification</b> — predicted job domain<br>
                <b>📊 Visual Analysis</b> — easy-to-read comparison chart<br>
                <b>💼 Job Retrieval</b> — relevant jobs from the sample dataset<br>
                <b>💬 Career Assistant</b> — questions about your analysis
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        NLP Resume–Job Matching System • Academic Project
    </div>
    """,
    unsafe_allow_html=True
)
