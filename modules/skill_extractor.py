import re


# Skill aliases: different names that represent the same skill
SKILL_ALIASES = {
    "nlp": "Natural Language Processing",
    "natural language processing": "Natural Language Processing",
    "ml": "Machine Learning",
    "machine learning": "Machine Learning",
    "ai": "Artificial Intelligence",
    "artificial intelligence": "Artificial Intelligence",
    "sklearn": "Scikit-learn",
    "scikit learn": "Scikit-learn",
    "scikit-learn": "Scikit-learn",
}


def load_skills(file_path):
    """Load skills from the skill database."""
    with open(file_path, "r", encoding="utf-8") as file:
        skills = [line.strip() for line in file if line.strip()]

    return skills


def normalize_skill(skill):
    """Convert skill aliases to a standard name."""
    skill_lower = skill.lower().strip()

    return SKILL_ALIASES.get(
        skill_lower,
        skill.strip()
    )


def extract_skills(text, skills):
    """Find skills present in the given text."""

    found_skills = []
    text_lower = text.lower()

    for skill in skills:
        pattern = r"\b" + re.escape(skill.lower()) + r"\b"

        if re.search(pattern, text_lower):
            normalized = normalize_skill(skill)

            if normalized not in found_skills:
                found_skills.append(normalized)

    return found_skills


def compare_skills(resume_skills, job_skills):
    """Compare resume skills with required job skills."""

    resume_set = set(
        normalize_skill(skill).lower()
        for skill in resume_skills
    )

    job_set = set(
        normalize_skill(skill).lower()
        for skill in job_skills
    )

    matching_skills = resume_set.intersection(job_set)
    missing_skills = job_set.difference(resume_set)

    if len(job_set) > 0:
        skill_match_percentage = (
            len(matching_skills) / len(job_set)
        ) * 100
    else:
        skill_match_percentage = 0

    return (
        sorted(matching_skills),
        sorted(missing_skills),
        round(skill_match_percentage, 2)
    )
