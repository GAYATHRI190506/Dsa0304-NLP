import re


# Read student information from file
with open("students.txt", "r") as file:
    data = file.read()


# Separate students
students = re.split(r"\n(?=Student \d+)", data)


# Validation functions
def validate_register_number(register_number):
    # Register number must contain exactly 9 digits
    return bool(re.fullmatch(r"\d{9}", register_number))


def validate_email(email):
    # Institutional email must end with @saveetha.com
    return bool(
        re.fullmatch(
            r"[A-Za-z0-9._%+-]+@saveetha\.com",
            email,
            re.IGNORECASE
        )
    )


def validate_course_code(course_code):
    # Example: DSA03, NLP04
    return bool(
        re.fullmatch(r"[A-Z]{3}\d{2}", course_code)
    )


def validate_semester(semester):
    # Semester must be between 1 and 8
    return bool(
        re.fullmatch(r"[1-8]", semester)
    )


def validate_mobile(mobile):
    # Indian mobile number: 10 digits starting from 6-9
    return bool(
        re.fullmatch(r"[6-9]\d{9}", mobile)
    )


# Process every student
for student in students:

    if not student.strip():
        continue

    print("\n========================================")

    # Extract fields using Regex
    register = re.search(
        r"Register Number:\s*(\S+)",
        student
    )

    email = re.search(
        r"Email:\s*(\S+)",
        student
    )

    course = re.search(
        r"Course Code:\s*(\S+)",
        student
    )

    semester = re.search(
        r"Semester:\s*(\S+)",
        student
    )

    mobile = re.search(
        r"Mobile:\s*(\S+)",
        student
    )

    # Get extracted values
    register_number = register.group(1) if register else ""
    email_address = email.group(1) if email else ""
    course_code = course.group(1) if course else ""
    semester_value = semester.group(1) if semester else ""
    mobile_number = mobile.group(1) if mobile else ""

    # Validate each field
    register_valid = validate_register_number(register_number)
    email_valid = validate_email(email_address)
    course_valid = validate_course_code(course_code)
    semester_valid = validate_semester(semester_value)
    mobile_valid = validate_mobile(mobile_number)

    print("        UNIVERSITY REGISTRATION")
    print("========================================")

    print("\nRegister Number:", register_number)

    if register_valid:
        print("✓ Register Number: Valid")
    else:
        print("✗ Register Number: Invalid")


    print("\nEmail:", email_address)

    if email_valid:
        print("✓ Institutional Email: Valid")
    else:
        print("✗ Institutional Email: Invalid")


    print("\nCourse Code:", course_code)

    if course_valid:
        print("✓ Course Code: Valid")
    else:
        print("✗ Course Code: Invalid")


    print("\nSemester:", semester_value)

    if semester_valid:
        print("✓ Semester: Valid")
    else:
        print("✗ Semester: Invalid")


    print("\nMobile:", mobile_number)

    if mobile_valid:
        print("✓ Mobile Number: Valid")
    else:
        print("✗ Mobile Number: Invalid")


    # Final registration status
    registration_success = all([
        register_valid,
        email_valid,
        course_valid,
        semester_valid,
        mobile_valid
    ])

    print("\n========================================")

    if registration_success:
        print("FINAL STATUS: REGISTRATION SUCCESSFUL")
    else:
        print("FINAL STATUS: REGISTRATION FAILED")

    print("========================================")