# ============================================================
# DSA03 NLP ASSIGNMENT - TASK 9
# PCFG AND PROBABILISTIC PARSING
# ============================================================

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


# ============================================================
# 1. MINI TREEBANK
# ============================================================

# Each tree is represented as a list of CFG production rules.
# The 20 trees form a small training treebank.

TREEBANK = [
    [
        ("S", "NP VP"),
        ("NP", "Det N"),
        ("VP", "V NP"),
        ("Det", "the"),
        ("N", "student"),
        ("V", "needs"),
        ("NP", "Det N"),
        ("Det", "the"),
        ("N", "timetable"),
    ],
    [
        ("S", "NP VP"),
        ("NP", "Det N"),
        ("VP", "V NP"),
        ("Det", "the"),
        ("N", "student"),
        ("V", "checks"),
        ("NP", "Det N"),
        ("Det", "the"),
        ("N", "assignment"),
    ],
    [
        ("S", "NP VP"),
        ("NP", "Det N"),
        ("VP", "V NP"),
        ("Det", "the"),
        ("N", "student"),
        ("V", "needs"),
        ("NP", "Det N"),
        ("Det", "a"),
        ("N", "hostel"),
    ],
    [
        ("S", "NP VP"),
        ("NP", "Det N"),
        ("VP", "V NP"),
        ("Det", "the"),
        ("N", "student"),
        ("V", "checks"),
        ("NP", "Det N"),
        ("Det", "the"),
        ("N", "portal"),
    ],
    [
        ("S", "NP VP"),
        ("NP", "Det N"),
        ("VP", "V NP"),
        ("Det", "the"),
        ("N", "student"),
        ("V", "needs"),
        ("NP", "Det N"),
        ("Det", "the"),
        ("N", "schedule"),
    ],
    [
        ("S", "NP VP"),
        ("NP", "Det N"),
        ("VP", "V NP"),
        ("Det", "the"),
        ("N", "student"),
        ("V", "checks"),
        ("NP", "Det N"),
        ("Det", "the"),
        ("N", "result"),
    ],
    [
        ("S", "NP VP"),
        ("NP", "Det N"),
        ("VP", "V NP"),
        ("Det", "the"),
        ("N", "student"),
        ("V", "needs"),
        ("NP", "Det N"),
        ("Det", "the"),
        ("N", "form"),
    ],
    [
        ("S", "NP VP"),
        ("NP", "Det N"),
        ("VP", "V NP"),
        ("Det", "the"),
        ("N", "student"),
        ("V", "checks"),
        ("NP", "Det N"),
        ("Det", "the"),
        ("N", "notice"),
    ],
    [
        ("S", "NP VP"),
        ("NP", "Det N"),
        ("VP", "V NP"),
        ("Det", "the"),
        ("N", "student"),
        ("V", "needs"),
        ("NP", "Det N"),
        ("Det", "the"),
        ("N", "certificate"),
    ],
    [
        ("S", "NP VP"),
        ("NP", "Det N"),
        ("VP", "V NP"),
        ("Det", "the"),
        ("N", "student"),
        ("V", "checks"),
        ("NP", "Det N"),
        ("Det", "the"),
        ("N", "deadline"),
    ],
    [
        ("S", "NP VP"),
        ("NP", "Det N"),
        ("VP", "V NP"),
        ("Det", "the"),
        ("N", "student"),
        ("V", "needs"),
        ("NP", "Det N"),
        ("Det", "the"),
        ("N", "document"),
    ],
    [
        ("S", "NP VP"),
        ("NP", "Det N"),
        ("VP", "V NP"),
        ("Det", "the"),
        ("N", "student"),
        ("V", "checks"),
        ("NP", "Det N"),
        ("Det", "the"),
        ("N", "notification"),
    ],
    [
        ("S", "NP VP"),
        ("NP", "Det N"),
        ("VP", "V NP"),
        ("Det", "the"),
        ("N", "student"),
        ("V", "needs"),
        ("NP", "Det N"),
        ("Det", "the"),
        ("N", "record"),
    ],
    [
        ("S", "NP VP"),
        ("NP", "Det N"),
        ("VP", "V NP"),
        ("Det", "the"),
        ("N", "student"),
        ("V", "checks"),
        ("NP", "Det N"),
        ("Det", "the"),
        ("N", "password"),
    ],
    [
        ("S", "NP VP"),
        ("NP", "Det N"),
        ("VP", "V NP"),
        ("Det", "the"),
        ("N", "student"),
        ("V", "needs"),
        ("NP", "Det N"),
        ("Det", "the"),
        ("N", "answer"),
    ],
    [
        ("S", "NP VP"),
        ("NP", "Det N"),
        ("VP", "V NP"),
        ("Det", "the"),
        ("N", "student"),
        ("V", "checks"),
        ("NP", "Det N"),
        ("Det", "the"),
        ("N", "message"),
    ],
    [
        ("S", "NP VP"),
        ("NP", "Det N"),
        ("VP", "V NP"),
        ("Det", "the"),
        ("N", "student"),
        ("V", "needs"),
        ("NP", "Det N"),
        ("Det", "the"),
        ("N", "approval"),
    ],
    [
        ("S", "NP VP"),
        ("NP", "Det N"),
        ("VP", "V NP"),
        ("Det", "the"),
        ("N", "student"),
        ("V", "checks"),
        ("NP", "Det N"),
        ("Det", "the"),
        ("N", "status"),
    ],
    [
        ("S", "NP VP"),
        ("NP", "Det N"),
        ("VP", "V NP"),
        ("Det", "the"),
        ("N", "student"),
        ("V", "needs"),
        ("NP", "Det N"),
        ("Det", "the"),
        ("N", "permission"),
    ],
    [
        ("S", "NP VP"),
        ("NP", "Det N"),
        ("VP", "V NP"),
        ("Det", "the"),
        ("N", "student"),
        ("V", "checks"),
        ("NP", "Det N"),
        ("Det", "the"),
        ("N", "account"),
    ],
]


# ============================================================
# 2. ESTIMATE PCFG PROBABILITIES
# ============================================================

def estimate_pcfg(
    treebank: List[List[Tuple[str, str]]]
) -> Dict[Tuple[str, str], float]:

    lhs_counts = Counter()
    rule_counts = Counter()

    for tree in treebank:
        for lhs, rhs in tree:
            lhs_counts[lhs] += 1
            rule_counts[(lhs, rhs)] += 1

    probabilities = {}

    for (lhs, rhs), count in rule_counts.items():
        probabilities[(lhs, rhs)] = count / lhs_counts[lhs]

    return probabilities


def display_pcfg(probabilities):
    print("\n" + "=" * 70)
    print("2. PCFG RULE PROBABILITIES")
    print("=" * 70)

    grouped = defaultdict(list)

    for (lhs, rhs), probability in probabilities.items():
        grouped[lhs].append((rhs, probability))

    for lhs in sorted(grouped):
        print(f"\n{lhs} rules:")

        for rhs, probability in sorted(grouped[lhs]):
            print(
                f"  {lhs} -> {rhs:<15} "
                f"P = {probability:.4f}"
            )


# ============================================================
# 3. TREE PROBABILITY
# ============================================================

def tree_probability(
    tree: List[Tuple[str, str]],
    probabilities: Dict[Tuple[str, str], float]
) -> float:

    probability = 1.0

    for lhs, rhs in tree:
        rule_probability = probabilities.get((lhs, rhs), 0.0)

        if rule_probability == 0:
            return 0.0

        probability *= rule_probability

    return probability


def demonstrate_tree_probability(probabilities):
    print("\n" + "=" * 70)
    print("3. PROBABILITY OF A PARSE TREE")
    print("=" * 70)

    tree = TREEBANK[0]

    probability = tree_probability(
        tree,
        probabilities
    )

    print("\nSentence:")
    print("the student needs the timetable")

    print("\nRules used:")

    for lhs, rhs in tree:
        print(f"  {lhs} -> {rhs}")

    print(f"\nTree probability = {probability:.10f}")


# ============================================================
# 4. AMBIGUOUS PARSES
# ============================================================

AMBIGUOUS_TREE_1 = [
    ("S", "NP VP"),
    ("NP", "Det N"),
    ("VP", "VP PP"),
    ("VP", "V NP"),
    ("Det", "the"),
    ("N", "student"),
    ("V", "checks"),
    ("NP", "Det N"),
    ("Det", "the"),
    ("N", "timetable"),
    ("PP", "P NP"),
    ("P", "for"),
    ("NP", "Det N"),
    ("Det", "the"),
    ("N", "exam"),
]

AMBIGUOUS_TREE_2 = [
    ("S", "NP VP"),
    ("NP", "NP PP"),
    ("NP", "Det N"),
    ("VP", "V NP"),
    ("Det", "the"),
    ("N", "student"),
    ("V", "checks"),
    ("NP", "Det N"),
    ("Det", "the"),
    ("N", "timetable"),
    ("PP", "P NP"),
    ("P", "for"),
    ("NP", "Det N"),
    ("Det", "the"),
    ("N", "exam"),
]


def compare_ambiguous_parses(probabilities):
    print("\n" + "=" * 70)
    print("4. AMBIGUOUS PARSE PROBABILITIES")
    print("=" * 70)

    sentence = "the student checks the timetable for the exam"

    p1 = tree_probability(
        AMBIGUOUS_TREE_1,
        probabilities
    )

    p2 = tree_probability(
        AMBIGUOUS_TREE_2,
        probabilities
    )

    print("\nSentence:")
    print(sentence)

    print("\nParse 1: PP attached to VP")
    print(f"Probability = {p1:.12f}")

    print("\nParse 2: PP attached to NP")
    print(f"Probability = {p2:.12f}")

    if p1 > p2:
        print("\nVITERBI DECISION: Parse 1 is preferred.")
    elif p2 > p1:
        print("\nVITERBI DECISION: Parse 2 is preferred.")
    else:
        print("\nVITERBI DECISION: Both parses have equal probability.")


# ============================================================
# 5. CNF CONVERSION
# ============================================================

def show_cnf_grammar():
    print("\n" + "=" * 70)
    print("5. CNF REPRESENTATION")
    print("=" * 70)

    cnf_rules = [
        ("S", "NP VP"),
        ("VP", "V NP"),
        ("VP", "VP PP"),
        ("NP", "Det N"),
        ("NP", "NP PP"),
        ("PP", "P NP"),
    ]

    lexical_rules = [
        ("Det", "the"),
        ("Det", "a"),
        ("N", "student"),
        ("N", "timetable"),
        ("N", "exam"),
        ("V", "checks"),
        ("V", "needs"),
        ("P", "for"),
    ]

    print("\nBinary CFG rules:")
    for lhs, rhs in cnf_rules:
        print(f"  {lhs} -> {rhs}")

    print("\nLexical rules:")
    for lhs, rhs in lexical_rules:
        print(f"  {lhs} -> '{rhs}'")

    print(
        "\nThe grammar is already in a binary-friendly "
        "form for CKY-style dynamic programming."
    )


# ============================================================
# 6. VITERBI / CKY-STYLE PARSER
# ============================================================

@dataclass
class CKYCell:
    probability: float
    tree: str


class ViterbiCKY:
    def __init__(
        self,
        probabilities: Dict[Tuple[str, str], float]
    ):
        self.probabilities = probabilities

        self.binary_rules = []

        for (lhs, rhs), probability in probabilities.items():

            symbols = rhs.split()

            if len(symbols) == 2:
                self.binary_rules.append(
                    (lhs, symbols[0], symbols[1], probability)
                )

    def parse(
        self,
        words: List[str]
    ) -> Tuple[Optional[str], float, int]:

        n = len(words)

        chart = [
            [
                {}
                for _ in range(n + 1)
            ]
            for _ in range(n)
        ]

        operations = 0

        # ----------------------------------------------------
        # Lexical initialization
        # ----------------------------------------------------

        for i, word in enumerate(words):

            for (lhs, rhs), probability in self.probabilities.items():

                if rhs == word:

                    chart[i][i + 1][lhs] = CKYCell(
                        probability=probability,
                        tree=f"({lhs} {word})"
                    )

        # ----------------------------------------------------
        # Binary combinations
        # ----------------------------------------------------

        for span in range(2, n + 1):

            for start in range(n - span + 1):

                end = start + span

                for split in range(start + 1, end):

                    left_cell = chart[start][split]
                    right_cell = chart[split][end]

                    for lhs, left, right, rule_probability in self.binary_rules:

                        operations += 1

                        if left not in left_cell:
                            continue

                        if right not in right_cell:
                            continue

                        probability = (
                            rule_probability
                            * left_cell[left].probability
                            * right_cell[right].probability
                        )

                        existing = chart[start][end].get(lhs)

                        if (
                            existing is None
                            or probability > existing.probability
                        ):

                            chart[start][end][lhs] = CKYCell(
                                probability=probability,
                                tree=(
                                    f"({lhs} "
                                    f"{left_cell[left].tree} "
                                    f"{right_cell[right].tree})"
                                )
                            )

        result = chart[0][n].get("S")

        if result is None:
            return None, 0.0, operations

        return (
            result.tree,
            result.probability,
            operations
        )


def run_viterbi_parser(probabilities):
    print("\n" + "=" * 70)
    print("6. VITERBI / CKY-STYLE PARSING")
    print("=" * 70)

    parser = ViterbiCKY(probabilities)

    sentences = [
        "the student needs the timetable",
        "the student checks the assignment",
        "the student needs the hostel",
    ]

    for sentence in sentences:

        words = sentence.split()

        tree, probability, operations = parser.parse(words)

        print("\nSentence:")
        print(sentence)

        if tree is None:
            print("Result: NO PARSE")
        else:
            print("Best parse:")
            print(tree)
            print(f"Probability: {probability:.10f}")

        print(f"CKY operations: {operations}")


# ============================================================
# 7. UNKNOWN / UNSEEN WORD TEST
# ============================================================

def unknown_word_test(probabilities):
    print("\n" + "=" * 70)
    print("7. UNKNOWN WORD TEST")
    print("=" * 70)

    parser = ViterbiCKY(probabilities)

    sentence = "the student needs the scholarship"

    tree, probability, operations = parser.parse(
        sentence.split()
    )

    print("\nSentence:")
    print(sentence)

    if tree is None:
        print("Result: NO PARSE")
        print(
            "Reason: 'scholarship' is not present in "
            "the learned lexical rules."
        )
    else:
        print("Unexpected parse:")
        print(tree)


# ============================================================
# 8. PROBABILITY ASSUMPTIONS
# ============================================================

def explain_assumptions():
    print("\n" + "=" * 70)
    print("8. PCFG ASSUMPTIONS AND FAILURE CASES")
    print("=" * 70)

    print("""
PCFG assumptions:
1. Each nonterminal chooses a grammar rule independently.
2. The probability of a tree is the product of its rule probabilities.
3. Rule probabilities for one nonterminal sum to 1.
4. The grammar structure determines the possible parses.
5. Viterbi parsing chooses the highest-probability parse.

Important limitations:
1. PCFG probabilities depend only on the parent nonterminal.
2. Lexical context is not directly represented.
3. Long-distance dependencies are difficult to model.
4. Unknown words cause failures without smoothing or an OOV model.
5. Ambiguity may be resolved incorrectly when structural context
   is not captured by the grammar.
""")


# ============================================================
# 9. MINI TREEBANK STATISTICS
# ============================================================

def show_treebank_statistics():
    print("\n" + "=" * 70)
    print("9. MINI TREEBANK STATISTICS")
    print("=" * 70)

    print(f"\nNumber of training trees: {len(TREEBANK)}")

    total_rules = sum(
        len(tree)
        for tree in TREEBANK
    )

    print(f"Total rule occurrences: {total_rules}")

    lhs_counts = Counter()

    for tree in TREEBANK:
        for lhs, rhs in tree:
            lhs_counts[lhs] += 1

    print("\nRule occurrences by LHS:")

    for lhs, count in sorted(lhs_counts.items()):
        print(f"  {lhs:<8} {count}")


# ============================================================
# 10. MAIN
# ============================================================

def main():

    print("=" * 70)
    print("DSA03 NLP ASSIGNMENT - TASK 9")
    print("PCFG AND PROBABILISTIC PARSING")
    print("=" * 70)

    print("\nTraining a PCFG from a 20-sentence mini-treebank...")

    probabilities = estimate_pcfg(TREEBANK)

    print(
        f"Training trees used: {len(TREEBANK)}"
    )

    show_treebank_statistics()

    display_pcfg(probabilities)

    demonstrate_tree_probability(probabilities)

    compare_ambiguous_parses(probabilities)

    show_cnf_grammar()

    run_viterbi_parser(probabilities)

    unknown_word_test(probabilities)

    explain_assumptions()

    print("\n" + "=" * 70)
    print("TASK 9 IMPLEMENTATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()