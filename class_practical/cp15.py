import nltk
from nltk import PCFG

grammar = PCFG.fromstring("""
S -> NP VP [1.0]
NP -> 'Ram' [0.5] | 'Ravi' [0.5]
VP -> 'runs' [0.5] | 'plays' [0.5]
""")

sentence = input("Enter sentence: ").split()
parser = nltk.ViterbiParser(grammar)

found = False
for tree in parser.parse(sentence):
    print(tree)
    print("Probability:", tree.prob())
    found = True

if not found:
    print("No parse found")
