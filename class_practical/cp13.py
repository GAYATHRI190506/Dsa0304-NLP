import nltk
from nltk import CFG

grammar = CFG.fromstring("""
S -> NP VP
NP -> 'Ram' | 'apple'
VP -> 'eats' NP
""")

sentence = input("Enter sentence: ").split()
parser = nltk.ChartParser(grammar)

found = False
for tree in parser.parse(sentence):
    tree.pretty_print()
    found = True

if not found:
    print("No parse tree found")
