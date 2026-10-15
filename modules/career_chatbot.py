def career_chatbot(
    question,
    overall_score,
    job_category,
    matching_skills,
    missing_skills
):
    """
    Simple rule-based career assistant.
    """

    question = question.lower()

    if "score" in question or "match" in question:
        return (
            f"Your overall resume-job match score is "
            f"{overall_score}%."
        )

    elif "missing" in question or "lack" in question:
        if missing_skills:
            skills = ", ".join(missing_skills)
            return f"The main missing skills are: {skills}."
        else:
            return "No major missing skills were detected."

    elif "skill" in question and "match" in question:
        if matching_skills:
            skills = ", ".join(matching_skills)
            return f"Your matching skills are: {skills}."
        else:
            return "No matching skills were detected."

    elif "domain" in question or "job type" in question:
        return (
            f"The detected job domain is "
            f"{job_category.replace('_', ' ')}."
        )

    elif "recommend" in question or "improve" in question:
        if missing_skills:
            skills = ", ".join(missing_skills[:5])
            return (
                "You can improve your resume by gaining "
                f"experience in: {skills}."
            )
        else:
            return (
                "Your resume already covers the major "
                "skills detected in the job description."
            )

    else:
        return (
            "I can help you with your match score, "
            "missing skills, matching skills, job domain, "
            "and resume improvement."
        )
