# ============================================================
# DSA03 NLP ASSIGNMENT 3
# Task 2 - Derivations and Parse Trees
# ============================================================

from collections import deque


# ------------------------------------------------------------
# SIMPLE TREE NODE
# ------------------------------------------------------------

class Node:
    def __init__(self, label, children=None):
        self.label = label
        self.children = children or []


# ------------------------------------------------------------
# TREE PRINTING
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# BRACKETED TREE
# ------------------------------------------------------------

def bracketed(node):
    if not node.children:
        return node.label

    return "(" + node.label + " " + " ".join(
        bracketed(child) for child in node.children
    ) + ")"


# ------------------------------------------------------------
# TREE STATISTICS
# ------------------------------------------------------------

def tree_statistics(node):
    if not node.children:
        return 1, 1

    total_nodes = 1
    child_heights = []

    for child in node.children:
        nodes, height = tree_statistics(child)
        total_nodes += nodes
        child_heights.append(height)

    return total_nodes, 1 + max(child_heights)


# ------------------------------------------------------------
# EXAMPLE 1
# ------------------------------------------------------------

def tree1():
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
# EXAMPLE 2
# ------------------------------------------------------------

def tree2():
    return Node("S", [
        Node("NP", [
            Node("Det", [Node("The")]),
            Node("N", [Node("student")])
        ]),
        Node("VP", [
            Node("V", [Node("submits")]),
            Node("NP", [
                Node("Det", [Node("the")]),
                Node("N", [Node("assignment")])
            ])
        ])
    ])


# ------------------------------------------------------------
# EXAMPLE 3
# ------------------------------------------------------------

def tree3():
    return Node("S", [
        Node("NP", [
            Node("Det", [Node("The")]),
            Node("N", [Node("student")])
        ]),
        Node("VP", [
            Node("V", [Node("checks")]),
            Node("NP", [
                Node("Det", [Node("the")]),
                Node("N", [Node("portal")])
            ])
        ])
    ])


# ------------------------------------------------------------
# EXAMPLE 4
# AMBIGUOUS PP ATTACHMENT
# ------------------------------------------------------------

def tree4_np_attachment():
    return Node("S", [
        Node("NP", [
            Node("Det", [Node("The")]),
            Node("N", [Node("student")])
        ]),
        Node("VP", [
            Node("V", [Node("needs")]),
            Node("NP", [
                Node("Det", [Node("the")]),
                Node("N", [Node("timetable")]),
                Node("PP", [
                    Node("P", [Node("for")]),
                    Node("NP", [
                        Node("Det", [Node("the")]),
                        Node("N", [Node("exam")])
                    ])
                ])
            ])
        ])
    ])


def tree4_vp_attachment():
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
            ]),
            Node("PP", [
                Node("P", [Node("for")]),
                Node("NP", [
                    Node("Det", [Node("the")]),
                    Node("N", [Node("exam")])
                ])
            ])
        ])
    ])


# ------------------------------------------------------------
# EXAMPLE 5
# COORDINATION AMBIGUITY
# ------------------------------------------------------------

def tree5_left_coordination():
    return Node("S", [
        Node("NP", [
            Node("Det", [Node("The")]),
            Node("N", [Node("student")])
        ]),
        Node("VP", [
            Node("VP", [
                Node("V", [Node("checks")]),
                Node("NP", [
                    Node("Det", [Node("the")]),
                    Node("N", [Node("timetable")])
                ])
            ]),
            Node("Conj", [Node("and")]),
            Node("VP", [
                Node("V", [Node("downloads")]),
                Node("NP", [
                    Node("Det", [Node("the")]),
                    Node("N", [Node("form")])
                ])
            ])
        ])
    ])


def tree5_sentence_coordination():
    return Node("S", [
        Node("NP", [
            Node("Det", [Node("The")]),
            Node("N", [Node("student")])
        ]),
        Node("VP", [
            Node("V", [Node("checks")]),
            Node("NP", [
                Node("Det", [Node("the")]),
                Node("N", [Node("timetable")])
            ])
        ]),
        Node("Conj", [Node("and")]),
        Node("S", [
            Node("NP", [
                Node("Det", [Node("the")]),
                Node("N", [Node("student")])
            ]),
            Node("VP", [
                Node("V", [Node("downloads")]),
                Node("NP", [
                    Node("Det", [Node("the")]),
                    Node("N", [Node("form")])
                ])
            ])
        ])
    ])


# ------------------------------------------------------------
# LEFTMOST DERIVATION
# ------------------------------------------------------------

def leftmost_derivation():
    steps = [
        "S",
        "NP VP",
        "Det N VP",
        "The N VP",
        "The student VP",
        "The student V NP",
        "The student needs NP",
        "The student needs Det N",
        "The student needs the N",
        "The student needs the timetable"
    ]

    return steps


# ------------------------------------------------------------
# RIGHTMOST DERIVATION
# ------------------------------------------------------------

def rightmost_derivation():
    steps = [
        "S",
        "NP VP",
        "NP V NP",
        "NP V Det N",
        "NP V Det timetable",
        "NP V the timetable",
        "NP needs the timetable",
        "Det N needs the timetable",
        "The N needs the timetable",
        "The student needs the timetable"
    ]

    return steps


# ------------------------------------------------------------
# DISPLAY DERIVATION
# ------------------------------------------------------------

def show_derivation(title, steps):
    print("\n" + title)
    print("-" * 70)

    for i, step in enumerate(steps):
        print(f"{i:2d}. {step}")


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    print("=" * 70)
    print("TASK 2 - DERIVATIONS AND PARSE TREES")
    print("=" * 70)

    # --------------------------------------------------------
    # FIVE SENTENCES
    # --------------------------------------------------------

    sentences = [
        "The student needs the timetable.",
        "The student submits the assignment.",
        "The student checks the portal.",
        "The student needs the timetable for the exam.",
        "The student checks the timetable and downloads the form."
    ]

    print("\nFIVE TEST SENTENCES")
    print("-" * 70)

    for i, sentence in enumerate(sentences, 1):
        print(f"{i}. {sentence}")

    # --------------------------------------------------------
    # DERIVATIONS
    # --------------------------------------------------------

    show_derivation(
        "LEFTMOST DERIVATION - Sentence 1",
        leftmost_derivation()
    )

    show_derivation(
        "RIGHTMOST DERIVATION - Sentence 1",
        rightmost_derivation()
    )

    # --------------------------------------------------------
    # TREE 1
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PARSE TREE - SENTENCE 1")
    print("=" * 70)

    t1 = tree1()
    print_tree(t1)

    print("\nBracketed notation:")
    print(bracketed(t1))

    nodes, height = tree_statistics(t1)

    print(f"\nNode count : {nodes}")
    print(f"Tree height: {height}")

    # --------------------------------------------------------
    # TREE 2
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PARSE TREE - SENTENCE 2")
    print("=" * 70)

    t2 = tree2()
    print_tree(t2)

    print("\nBracketed notation:")
    print(bracketed(t2))

    nodes, height = tree_statistics(t2)

    print(f"\nNode count : {nodes}")
    print(f"Tree height: {height}")

    # --------------------------------------------------------
    # TREE 3
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PARSE TREE - SENTENCE 3")
    print("=" * 70)

    t3 = tree3()
    print_tree(t3)

    print("\nBracketed notation:")
    print(bracketed(t3))

    nodes, height = tree_statistics(t3)

    print(f"\nNode count : {nodes}")
    print(f"Tree height: {height}")

    # --------------------------------------------------------
    # AMBIGUITY 1
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("AMBIGUOUS SENTENCE 1 - PP ATTACHMENT")
    print("=" * 70)

    print("Sentence: The student needs the timetable for the exam.")

    print("\nParse A - PP attaches to NP:")
    t4a = tree4_np_attachment()
    print_tree(t4a)
    print("\nBracketed:")
    print(bracketed(t4a))

    print("\nParse B - PP attaches to VP:")
    t4b = tree4_vp_attachment()
    print_tree(t4b)
    print("\nBracketed:")
    print(bracketed(t4b))

    # --------------------------------------------------------
    # AMBIGUITY 2
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("AMBIGUOUS SENTENCE 2 - COORDINATION")
    print("=" * 70)

    print("Sentence: The student checks the timetable and downloads the form.")

    print("\nParse A - VP coordination:")
    t5a = tree5_left_coordination()
    print_tree(t5a)
    print("\nBracketed:")
    print(bracketed(t5a))

    print("\nParse B - S coordination:")
    t5b = tree5_sentence_coordination()
    print_tree(t5b)
    print("\nBracketed:")
    print(bracketed(t5b))

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TASK 2 IMPLEMENTATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()