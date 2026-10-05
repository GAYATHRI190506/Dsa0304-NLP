import re

text = "Python is a programming language. Contact: 9876543210, student@gmail.com"

# Match
match_result = re.match("Python", text)

# Search phone number
phone = re.search(r'\d{10}', text)

# Search email
email = re.search(r'\w+@\w+\.\w+', text)

if match_result:
    print("Match:", match_result.group())

if phone:
    print("Phone number:", phone.group())

if email:
    print("Email:", email.group())