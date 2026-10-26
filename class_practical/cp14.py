import nltk
from nltk import CFG

grammar = CFG.fromstring("""
S -> SG_N SG_V | PL_N PL_V
SG_N -> 'He' | 'She'
PL_N -> 'They'
SG_V -> 'runs'
PL_V -> 'run'
""")

sentence = input("Enter sentence: ").split()
parser = nltk.ChartParser(grammar)

if list(parser.parse(sentence)):
    print("Correct agreement")
else:
    print("Incorrect agreement")
