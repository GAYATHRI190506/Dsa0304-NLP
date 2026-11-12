"""
TASK 6: TOP-DOWN PARSING
DSA03 - Natural Language Processing

Covers:
1. Recursive-descent / goal-driven parsing
2. Stack / agenda trace
3. Backtracking
4. Ambiguous sentence
5. Left recursion
6. Left-recursion elimination
7. Redundant parsing
8. Worst-case complexity
"""

from time import perf_counter


# ============================================================
# GRAMMAR
# ============================================================

GRAMMAR = {
    "S": [
        ["NP", "VP"]
    ],

    "NP": [
        ["Det", "N"],
        ["Det", "N", "PP"],
        ["NP", "PP"]
    ],

    "VP": [
        ["V"],
        ["V", "NP"],
        ["V", "NP", "PP"]
    ],

    "PP": [
        ["P", "NP"]
    ],

    "Det": [
        ["the"]
    ],

    "N": [
        ["student"],
        ["timetable"],
        ["exam"],
        ["library"]
    ],

    "V": [
        ["checks"],
        ["needs"]
    ],

    "P": [
        ["for"],
        ["in"]
    ]
}

NONTERMINALS = set(GRAMMAR.keys())


def is_nonterminal(symbol):
    return symbol in NONTERMINALS


# ============================================================
# TOP-DOWN PARSER WITH BACKTRACKING
# ============================================================

class TopDownParser:

    def __init__(self, grammar):
        self.grammar = grammar
        self.steps = []
        self.backtracks = 0
        self.left_recursion_detected = False

    def parse(self, tokens):
        self.tokens = tokens
        self.steps = []
        self.backtracks = 0
        self.left_recursion_detected = False

        results = self.expand(
            "S",
            0,
            [],
            depth=0,
            active=set()
        )

        # Find a complete parse
        for tree, position in results:
            if position == len(tokens):
                return tree

        return None

    def expand(
        self,
        symbol,
        position,
        tree,
        depth,
        active
    ):
        indent = "  " * depth

        # ----------------------------------------------------
        # LEFT RECURSION PROTECTION
        # ----------------------------------------------------
        state = (symbol, position)

        if state in active and is_nonterminal(symbol):

            self.left_recursion_detected = True

            self.steps.append(
                f"{indent}LEFT RECURSION BLOCKED: "
                f"{symbol} at position {position}"
            )

            return []

        # Safety limit
        if depth > 100:

            self.steps.append(
                f"{indent}DEPTH LIMIT REACHED"
            )

            return []

        self.steps.append(
            f"{indent}STACK: [{symbol}] "
            f"POSITION: {position}"
        )

        # ----------------------------------------------------
        # TERMINAL
        # ----------------------------------------------------
        if not is_nonterminal(symbol):

            if position < len(self.tokens):

                if self.tokens[position] == symbol:

                    self.steps.append(
                        f"{indent}MATCH: '{symbol}'"
                    )

                    return [
                        (
                            (symbol, [symbol]),
                            position + 1
                        )
                    ]

            self.steps.append(
                f"{indent}FAIL: expected '{symbol}'"
            )

            return []

        # ----------------------------------------------------
        # NON-TERMINAL
        # ----------------------------------------------------

        new_active = set(active)
        new_active.add(state)

        all_results = []

        # Try every grammar rule
        for rule_no, rule in enumerate(
            self.grammar[symbol],
            1
        ):

            self.steps.append(
                f"{indent}EXPAND: "
                f"{symbol} -> {' '.join(rule)}"
            )

            partial_results = [
                ([], position)
            ]

            rule_failed = False

            for child_symbol in rule:

                next_results = []

                for children, current_position in partial_results:

                    child_results = self.expand(
                        child_symbol,
                        current_position,
                        None,
                        depth + 1,
                        new_active
                    )

                    for child_tree, next_position in child_results:

                        next_results.append(
                            (
                                children + [child_tree],
                                next_position
                            )
                        )

                partial_results = next_results

                if not partial_results:

                    rule_failed = True
                    break

            if rule_failed:

                self.backtracks += 1

                self.steps.append(
                    f"{indent}BACKTRACK: "
                    f"{symbol} -> {' '.join(rule)}"
                )

            else:

                for children, final_position in partial_results:

                    all_results.append(
                        (
                            (symbol, children),
                            final_position
                        )
                    )

        return all_results


# ============================================================
# PRINT TREE
# ============================================================

def print_tree(tree, level=0):

    if tree is None:
        return

    symbol, children = tree

    print("  " * level + symbol)

    for child in children:

        if isinstance(child, tuple):
            print_tree(child, level + 1)


# ============================================================
# TRACE
# ============================================================

def show_trace(parser, limit=100):

    print("\nSTACK / AGENDA TRACE")
    print("-" * 70)

    for i, step in enumerate(
        parser.steps[:limit],
        1
    ):

        print(f"{i:03d}. {step}")

    if len(parser.steps) > limit:

        print(
            f"... {len(parser.steps) - limit} more steps"
        )


# ============================================================
# EXAMPLE 1 - UNAMBIGUOUS
# ============================================================

def run_unambiguous_example():

    print("\n" + "=" * 70)
    print("EXAMPLE 1: UNAMBIGUOUS SENTENCE")
    print("=" * 70)

    sentence = "the student checks"

    tokens = sentence.split()

    parser = TopDownParser(GRAMMAR)

    start = perf_counter()

    tree = parser.parse(tokens)

    elapsed = perf_counter() - start

    print("Sentence:", sentence)

    if tree is not None:

        print("Result: ACCEPTED")

        print("\nParse Tree:")
        print_tree(tree)

    else:

        print("Result: REJECTED")

    print("\nParsing steps:", len(parser.steps))
    print("Backtracks:", parser.backtracks)

    print(
        "Left recursion detected:",
        parser.left_recursion_detected
    )

    print(f"Runtime: {elapsed:.6f} seconds")

    show_trace(parser)


# ============================================================
# EXAMPLE 2 - AMBIGUOUS
# ============================================================

def run_ambiguous_example():

    print("\n" + "=" * 70)
    print("EXAMPLE 2: AMBIGUOUS SENTENCE")
    print("=" * 70)

    sentence = (
        "the student checks the timetable for the exam"
    )

    tokens = sentence.split()

    print("Sentence:", sentence)

    print("\nPossible PP attachments:")

    print(
        "1. VP attachment:"
    )

    print(
        "   [the student] "
        "[checks the timetable] "
        "[for the exam]"
    )

    print(
        "\n2. NP attachment:"
    )

    print(
        "   [the student] "
        "[checks [the timetable for the exam]]"
    )

    parser = TopDownParser(GRAMMAR)

    start = perf_counter()

    tree = parser.parse(tokens)

    elapsed = perf_counter() - start

    if tree is not None:

        print("\nResult: ACCEPTED")

        print("\nFirst successful parse:")
        print_tree(tree)

    else:

        print("\nResult: REJECTED")

    print("\nParsing steps:", len(parser.steps))
    print("Backtracks:", parser.backtracks)

    print(
        "Left recursion detected:",
        parser.left_recursion_detected
    )

    print(f"Runtime: {elapsed:.6f} seconds")

    show_trace(parser)


# ============================================================
# LEFT RECURSION
# ============================================================

def demonstrate_left_recursion():

    print("\n" + "=" * 70)
    print("LEFT RECURSION")
    print("=" * 70)

    print("""
Original rule:

    NP -> NP PP

This is left-recursive because NP immediately
calls itself before consuming input.

Expansion can become:

    NP -> NP PP
       -> NP PP PP
       -> NP PP PP PP
       -> ...

A naive recursive-descent parser may repeatedly
expand NP without consuming a word.

Therefore left recursion can cause infinite recursion.
""")


# ============================================================
# LEFT RECURSION ELIMINATION
# ============================================================

def demonstrate_left_recursion_elimination():

    print("\n" + "=" * 70)
    print("LEFT-RECURSION ELIMINATION")
    print("=" * 70)

    print("""
Original:

    NP -> NP PP
    NP -> Det N

After elimination:

    NP -> Det N NP_EXT
    NP_EXT -> PP NP_EXT
    NP_EXT -> epsilon

This grammar first consumes the basic noun phrase
and then optionally consumes one or more PPs.

Examples:

    the timetable

    the timetable for the exam

    the timetable for the exam in the library
""")


# ============================================================
# MULTIPLE PP ATTACHMENT
# ============================================================

def pp_attachment_counts():

    print("\n" + "=" * 70)
    print("REDUNDANT PARSING WITH MULTIPLE PPs")
    print("=" * 70)

    print(
        "A simplified binary attachment model is used:"
    )

    print(
        "each PP can be attached to an NP or VP."
    )

    print()

    print(
        f"{'Number of PPs':<20}"
        f"{'Possible patterns':<20}"
    )

    print("-" * 40)

    for k in range(1, 7):

        patterns = 2 ** k

        print(
            f"{k:<20}"
            f"{patterns:<20}"
        )

    print("""
The number of possible attachment patterns grows
exponentially in this simplified model.

A naive top-down parser may repeatedly explore
similar sub-problems.

Chart parsing methods such as Earley reduce
this repeated work by storing parser states.
""")


# ============================================================
# COMPLEXITY
# ============================================================

def complexity_discussion():

    print("\n" + "=" * 70)
    print("WORST-CASE COMPLEXITY")
    print("=" * 70)

    print("""
Recursive-descent parsing is efficient for simple,
nearly deterministic grammars.

Approximate best-case behavior:
    O(n)

With many grammar alternatives and ambiguity,
backtracking can become exponential.

A simplified search model is:

    O(b^d)

where:

    b = average number of alternatives
    d = search depth

Earley parsing provides a cubic worst-case bound
for general CFG parsing:

    O(n^3)

Therefore chart-based parsing is more suitable
for highly ambiguous CFGs.
""")


# ============================================================
# EXAMPLE 3 - UNGRAMMATICAL
# ============================================================

def run_ungrammatical_example():

    print("\n" + "=" * 70)
    print("EXAMPLE 3: UNGRAMMATICAL SENTENCE")
    print("=" * 70)

    sentence = "student checks"

    tokens = sentence.split()

    parser = TopDownParser(GRAMMAR)

    start = perf_counter()

    tree = parser.parse(tokens)

    elapsed = perf_counter() - start

    print("Sentence:", sentence)

    if tree is None:
        print("Result: REJECTED")
    else:
        print("Result: ACCEPTED")

    print("\nParsing steps:", len(parser.steps))
    print("Backtracks:", parser.backtracks)

    print(
        "Left recursion detected:",
        parser.left_recursion_detected
    )

    print(f"Runtime: {elapsed:.6f} seconds")

    show_trace(parser, 80)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TASK 6 - TOP-DOWN PARSING")
    print("=" * 70)

    run_unambiguous_example()

    run_ambiguous_example()

    demonstrate_left_recursion()

    demonstrate_left_recursion_elimination()

    pp_attachment_counts()

    complexity_discussion()

    run_ungrammatical_example()

    print("\n" + "=" * 70)
    print("TASK 6 IMPLEMENTATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()