import re

expression = input("Enter logical expression: ")

pattern = r'^(AND|OR|NOT|FORALL|EXISTS)\s*\(.*\)$'

if re.fullmatch(pattern, expression):
    print("Valid logical expression")
else:
    print("Invalid logical expression")
