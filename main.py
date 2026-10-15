from modules.resume_parser import extract_text_from_pdf
from modules.preprocessing import preprocess_text
from modules.text_representation import create_tfidf_vectors
from modules.similarity import calculate_similarity
from modules.skill_extractor import (load_skills,extract_skills,compare_skills)
from modules.keyphrase_extractor import extract_keyphrases
from modules.job_classifier import train_job_classifier, classify_job
from modules.embeddings import calculate_semantic_similarity
from modules.score_calculator import calculate_overall_score
from modules.ner_extractor import extract_entities
from modules.summarizer import summarize_text
from modules.job_search import search_jobs
from modules.career_chatbot import career_chatbot
from modules.qa_system import answer_question
from modules.visualization import create_score_chart






pdf_path = "sample_resume.pdf"

raw_resume = extract_text_from_pdf(pdf_path)
resume_entities = extract_entities(raw_resume)


with open("data/sample_job.txt", "r", encoding="utf-8") as file:
    raw_job = file.read()
job_entities = extract_entities(raw_job)
job_summary = summarize_text(raw_job, num_sentences=3)
classifier_vectorizer, classifier_model = train_job_classifier("data/job_dataset.csv")

job_category = classify_job(
    raw_job,
    classifier_vectorizer,
    classifier_model
)

skills = load_skills("data/skills.txt")

resume_skills = extract_skills(
    raw_resume,
    skills
)

job_skills = extract_skills(
    raw_job,
    skills
)

matching_skills, missing_skills, skill_match_percentage = compare_skills(
    resume_skills,
    job_skills
)

matching_jobs = search_jobs(
    resume_skills,
    "data/jobs.csv",
    top_n=5
)

resume_keyphrases = extract_keyphrases(raw_resume, top_n=10)
job_keyphrases = extract_keyphrases(raw_job, top_n=10)
resume_tokens = preprocess_text(raw_resume)
job_tokens = preprocess_text(raw_job)

clean_resume = " ".join(resume_tokens)
clean_job = " ".join(job_tokens)


vectorizer, vectors = create_tfidf_vectors(
    clean_resume,
    clean_job
)



match_score = calculate_similarity(vectors)
semantic_score = calculate_semantic_similarity(raw_resume,raw_job)
overall_score = calculate_overall_score(match_score,semantic_score,skill_match_percentage)
score_chart = create_score_chart(
    match_score,
    semantic_score,
    skill_match_percentage,
    overall_score
)
chatbot_question = input(
    "\nAsk the Career Assistant: "
)

chatbot_response = career_chatbot(
    chatbot_question,
    overall_score,
    job_category,
    matching_skills,
    missing_skills
)
qa_question = "What cloud technology is mentioned?"

qa_answer = answer_question(
    qa_question,
    raw_job
)
print("\n====================================")
print("\n========== MATCHING JOBS ==========")

for job in matching_jobs:
    print(f"\n💼 {job['title']}")
    print(f"Company: {job['company']}")
    print(f"Skill Match: {job['match_percentage']}%")
    print(f"Matching Skills: {job['matching_skills']}")
print("     RESUME–JOB MATCHING SYSTEM")
print("====================================")

print(f"\nTF-IDF Match Score: {match_score}%")
print(f"Semantic Similarity Score: {semantic_score}%")
print(f"Skill Match Score: {skill_match_percentage}%")
print(f"\nOverall Resume–Job Match: {overall_score}%")

print("\n========== SKILL ANALYSIS ==========")

print("\nMatching Skills:")
for skill in matching_skills:
    print("✓", skill)

print("\nMissing Skills:")
for skill in missing_skills:
    print("✗", skill)

print("\n========== RESUME NAMED ENTITIES ==========")
for entity in resume_entities:
    print(f"• {entity['text']} → {entity['label']}")

print("\n========== JOB NAMED ENTITIES ==========")
for entity in job_entities:
    print(f"• {entity['text']} → {entity['label']}")

print("\n========== RESUME KEYPHRASES ==========")
for phrase in resume_keyphrases:
    print("•", phrase)

print("\n========== JOB KEYPHRASES ==========")
for phrase in job_keyphrases:
    print("•", phrase)

print("\n========== JOB CLASSIFICATION ==========")
print(f"Predicted Job Domain: {job_category.replace('_', ' ')}")
print("\n========== JOB DESCRIPTION SUMMARY ==========")

for sentence in job_summary:
    print("•", sentence)
print("\n====================================") 
print("\n========== CAREER ASSISTANT ==========")
print(f"User: {chatbot_question}")
print(f"Bot: {chatbot_response}")
print("\n========== JOB QUESTION ANSWERING ==========")
print(f"Question: {qa_question}")
print(f"Answer: {qa_answer}")