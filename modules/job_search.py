import pandas as pd


def search_jobs(resume_skills, dataset_path="data/jobs.csv", top_n=5):
    """
    Find jobs that match the skills present in the resume.
    """

    jobs = pd.read_csv(dataset_path)

    resume_skill_set = {
        skill.lower()
        for skill in resume_skills
    }

    results = []

    for _, job in jobs.iterrows():

        job_skills = [
            skill.strip()
            for skill in job["skills"].split(";")
        ]

        job_skill_set = {
            skill.lower()
            for skill in job_skills
        }

        matching_skills = resume_skill_set.intersection(
            job_skill_set
        )

        if len(job_skill_set) > 0:
            skill_percentage = (
                len(matching_skills)
                / len(job_skill_set)
            ) * 100
        else:
            skill_percentage = 0

        # Ignore jobs with very weak skill overlap
        if len(matching_skills) < 2:
         continue

        results.append({
            "title": job["title"],
            "company": job["company"],
            "description": job["description"],
            "matching_skills": sorted(matching_skills),
            "match_percentage": round(skill_percentage, 2)
        })

    results.sort(
        key=lambda x: (
            x["match_percentage"],
            len(x["matching_skills"])
        ),
        reverse=True
    )

    return results[:top_n]
