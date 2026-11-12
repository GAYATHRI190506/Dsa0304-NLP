# ============================================================
# DSA03 NLP ASSIGNMENT 3
# Task 3 - Sentence-Level Constructions
# ============================================================

class Node:
    def __init__(self, label, children=None):
        self.label = label
        self.children = children or []


def print_tree(node, prefix="", is_last=True):
    connector = "└── " if is_last else "├── "
    print(prefix + connector + node.label)

    new_prefix = prefix + ("    " if is_last else "│   ")

    for i, child in enumerate(node.children):
        print_tree(
            child,
            new_prefix,
            i == len(node.children) - 1
        )


def bracketed(node):
    if not node.children:
        return node.label

    return "(" + node.label + " " + " ".join(
        bracketed(child) for child in node.children
    ) + ")"


# ------------------------------------------------------------
# 1. DECLARATIVE
# ------------------------------------------------------------

def declarative_tree():
    return Node("S", [
        Node("NP", [
            Node("Det", [Node("The")]),
            Node("N", [Node("student")])
        ]),
        Node("VP", [
            Node("V", [Node("needs")]),
            Node("NP", [
                Node("Det", [Node("the")]),
                Node("N", [Node("timetable")])
            ])
        ])
    ])


# ------------------------------------------------------------
# 2. IMPERATIVE
# ------------------------------------------------------------

def imperative_tree():
    return Node("S", [
        Node("VP", [
            Node("V", [Node("Check")]),
            Node("NP", [
                Node("Det", [Node("the")]),
                Node("N", [Node("timetable")])
            ])
        ])
    ])


# ------------------------------------------------------------
# 3. YES/NO QUESTION
# ------------------------------------------------------------

def yes_no_tree():
    return Node("S", [
        Node("Aux", [Node("Does")]),
        Node("NP", [
            Node("Det", [Node("the")]),
            Node("N", [Node("student")])
        ]),
        Node("VP", [
            Node("V", [Node("need")]),
            Node("NP", [
                Node("Det", [Node("the")]),
                Node("N", [Node("timetable")])
            ])
        ])
    ])


# ------------------------------------------------------------
# 4. SUBJECT WH-QUESTION
# ------------------------------------------------------------

def subject_wh_tree():
    return Node("S", [
        Node("WhNP", [
            Node("WhPron", [Node("Who")])
        ]),
        Node("VP", [
            Node("V", [Node("provides")]),
            Node("NP", [
                Node("Det", [Node("the")]),
                Node("N", [Node("schedule")])
            ])
        ])
    ])


# ------------------------------------------------------------
# 5. NON-SUBJECT WH-QUESTION
# ------------------------------------------------------------

def nonsubject_wh_tree():
    return Node("S", [
        Node("WhNP", [
            Node("WhDet", [Node("What")]),
            Node("N", [Node("information")])
        ]),
        Node("Aux", [Node("does")]),
        Node("NP", [
            Node("Det", [Node("the")]),
            Node("N", [Node("student")])
        ]),
        Node("VP", [
            Node("V", [Node("need")]),
            Node("GAP", [Node("t")])
        ])
    ])


# ------------------------------------------------------------
# 6. SUBORDINATE CLAUSE
# ------------------------------------------------------------

def subordinate_tree():
    return Node("S", [
        Node("NP", [
            Node("Det", [Node("The")]),
            Node("N", [Node("student")])
        ]),
        Node("VP", [
            Node("V", [Node("knows")]),
            Node("SComp", [
                Node("Comp", [Node("that")]),
                Node("S", [
                    Node("NP", [
                        Node("Det", [Node("the")]),
                        Node("N", [Node("exam")])
                    ]),
                    Node("VP", [
                        Node("V", [Node("starts")])
                    ])
                ])
            ])
        ])
    ])


# ------------------------------------------------------------
# 7. RELATIVE CLAUSE
# ------------------------------------------------------------

def relative_clause_tree():
    return Node("S", [
        Node("NP", [
            Node("Det", [Node("The")]),
            Node("N", [Node("student")]),
            Node("RelClause", [
                Node("RelPron", [Node("who")]),
                Node("VP", [
                    Node("V", [Node("needs")]),
                    Node("NP", [
                        Node("Det", [Node("the")]),
                        Node("N", [Node("timetable")])
                    ])
                ])
            ])
        ]),
        Node("VP", [
            Node("V", [Node("asks")]),
            Node("NP", [
                Node("Det", [Node("the")]),
                Node("N", [Node("teacher")])
            ])
        ])
    ])


# ------------------------------------------------------------
# DISPLAY
# ------------------------------------------------------------

def display(title, sentence, tree):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    print("Sentence:")
    print(sentence)

    print("\nParse tree:")
    print_tree(tree)

    print("\nBracketed notation:")
    print(bracketed(tree))


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    print("=" * 70)
    print("TASK 3 - SENTENCE-LEVEL CONSTRUCTIONS")
    print("=" * 70)

    display(
        "1. DECLARATIVE",
        "The student needs the timetable.",
        declarative_tree()
    )

    display(
        "2. IMPERATIVE",
        "Check the timetable.",
        imperative_tree()
    )

    display(
        "3. YES/NO QUESTION",
        "Does the student need the timetable?",
        yes_no_tree()
    )

    display(
        "4. SUBJECT WH-QUESTION",
        "Who provides the schedule?",
        subject_wh_tree()
    )

    display(
        "5. NON-SUBJECT WH-QUESTION",
        "What information does the student need?",
        nonsubject_wh_tree()
    )

    display(
        "6. SUBORDINATE CLAUSE",
        "The student knows that the exam starts.",
        subordinate_tree()
    )

    display(
        "7. RELATIVE CLAUSE",
        "The student who needs the timetable asks the teacher.",
        relative_clause_tree()
    )

    # --------------------------------------------------------
    # STRUCTURAL ANALYSIS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STRUCTURAL ANALYSIS")
    print("=" * 70)

    print("""
1. Declarative:
   Normal subject + verb phrase order.

2. Imperative:
   The subject is normally omitted and the sentence begins
   with a base-form verb.

3. Yes/No question:
   The auxiliary moves before the subject.

4. Subject Wh-question:
   The Wh phrase occupies the subject position.

5. Non-subject Wh-question:
   The Wh phrase moves to the front and a GAP marks its
   original object position.

6. Subordinate clause:
   A complementizer such as 'that' introduces an embedded
   sentence.

7. Relative clause:
   A relative pronoun introduces a clause modifying a noun.

8. Coordination:
   Clauses or phrases can be joined using conjunctions such
   as 'and' and 'or'.

9. Long-distance dependency:
   Wh-movement can connect a fronted phrase with a GAP inside
   an embedded clause. A simple CFG needs additional rules
   or feature mechanisms to represent this dependency.
""")

    print("=" * 70)
    print("TASK 3 IMPLEMENTATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()