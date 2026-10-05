import re


# Read resume file
with open("resumes.txt", "r") as file:
    data = file.read()


# Separate each resume
resumes = re.split(r"=+ RESUME \d+ =+", data)

skills_list = [
    "Python",
    "Java",
    "SQL",
    "Machine Learning",
    "NLP"
]

eligible_candidates = []


# Process each resume
for resume in resumes:

    if not resume.strip():
        continue

    # Extract name
    name = re.search(
        r"Name:\s*([A-Za-z ]+)",
        resume
    )

    # Extract email
    email = re.search(
        r"[\w.-]+@[\w.-]+\.\w+",
        resume
    )

    # Extract mobile number
    mobile = re.search(
        r"(?:\+91[- ]?)?[6-9]\d{9}",
        resume
    )

    # Extract experience
    experience = re.search(
        r"Experience:\s*(\d+(?:\.\d+)?)\s*years?",
        resume,
        re.IGNORECASE
    )

    # Extract technical skills
    found_skills = []

    for skill in skills_list:
        if re.search(
            r"\b" + re.escape(skill) + r"\b",
            resume,
            re.IGNORECASE
        ):
            found_skills.append(skill)

    # Get values
    candidate_name = (
        name.group(1).strip()
        if name else "Not Found"
    )

    candidate_email = (
        email.group()
        if email else "Not Found"
    )

    candidate_mobile = (
        mobile.group()
        if mobile else "Not Found"
    )

    candidate_experience = (
        float(experience.group(1))
        if experience else 0
    )

    # Check eligibility
    is_eligible = (
        candidate_experience >= 2
        and "Python" in found_skills
    )

    # Display candidate information
    print("\n----------------------------------------")
    print("Candidate Name :", candidate_name)
    print("Email          :", candidate_email)
    print("Mobile         :", candidate_mobile)
    print("Skills         :", ", ".join(found_skills))
    print("Experience     :", candidate_experience, "years")
    print(
        "Eligibility    :",
        "Eligible" if is_eligible else "Not Eligible"
    )

    # Store eligible candidates
    if is_eligible:
        eligible_candidates.append({
            "name": candidate_name,
            "email": candidate_email,
            "mobile": candidate_mobile,
            "skills": found_skills,
            "experience": candidate_experience
        })


# Final eligible candidate report
print("\n\n========================================")
print("       ELIGIBLE CANDIDATES")
print("========================================")

for i, candidate in enumerate(eligible_candidates, 1):

    print(f"\n{i}. {candidate['name']}")
    print("   Email      :", candidate["email"])
    print("   Mobile     :", candidate["mobile"])
    print("   Skills     :", ", ".join(candidate["skills"]))
    print("   Experience :", candidate["experience"], "years")


print("\n----------------------------------------")
print(
    "Total Eligible Candidates:",
    len(eligible_candidates)
)