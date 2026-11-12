"""
TASK 7: EARLEY PARSING
DSA03 - Natural Language Processing

Covers:
1. Earley dotted rules
2. Origin positions
3. Predictor
4. Scanner
5. Completer
6. Chart states
7. Backpointers
8. Ambiguity
9. Left recursion
10. Complexity comparison
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
# EARLEY STATE
# ============================================================

class State:

    def __init__(
        self,
        lhs,
        rhs,
        dot,
        origin,
        backpointers=None
    ):

        self.lhs = lhs
        self.rhs = tuple(rhs)
        self.dot = dot
        self.origin = origin

        if backpointers is None:
            backpointers = []

        self.backpointers = backpointers

    def next_symbol(self):

        if self.dot < len(self.rhs):
            return self.rhs[self.dot]

        return None

    def is_complete(self):

        return self.dot == len(self.rhs)

    def key(self):

        return (
            self.lhs,
            self.rhs,
            self.dot,
            self.origin
        )

    def dotted_rule(self):

        before = list(self.rhs[:self.dot])

        after = list(self.rhs[self.dot:])

        parts = before + ["•"] + after

        return (
            f"{self.lhs} -> "
            + " ".join(parts)
        )

    def __str__(self):

        return (
            f"{self.dotted_rule()}, "
            f"origin={self.origin}"
        )


# ============================================================
# EARLEY PARSER
# ============================================================

class EarleyParser:

    def __init__(self, grammar):

        self.grammar = grammar

        self.chart = []

        self.operation_log = []

        self.completed_states = 0

        self.predicted_states = 0

        self.scanned_states = 0

    # --------------------------------------------------------
    # ADD STATE
    # --------------------------------------------------------

    def add_state(self, chart_index, state):

        existing = {
            s.key()
            for s in self.chart[chart_index]
        }

        if state.key() not in existing:

            self.chart[chart_index].append(state)

            return True

        return False

    # --------------------------------------------------------
    # PARSE
    # --------------------------------------------------------

    def parse(self, tokens):

        n = len(tokens)

        self.chart = [
            []
            for _ in range(n + 1)
        ]

        self.operation_log = []

        self.completed_states = 0
        self.predicted_states = 0
        self.scanned_states = 0

        # Augmented grammar
        start_state = State(
            "γ",
            ["S"],
            0,
            0
        )

        self.chart[0].append(start_state)

        # ----------------------------------------------------
        # PROCESS EACH CHART
        # ----------------------------------------------------

        for i in range(n + 1):

            index = 0

            while index < len(self.chart[i]):

                state = self.chart[i][index]

                index += 1

                next_symbol = state.next_symbol()

                # --------------------------------------------
                # COMPLETE
                # --------------------------------------------

                if state.is_complete():

                    self.completer(
                        i,
                        state
                    )

                # --------------------------------------------
                # PREDICT
                # --------------------------------------------

                elif is_nonterminal(next_symbol):

                    self.predictor(
                        i,
                        state
                    )

                # --------------------------------------------
                # SCAN
                # --------------------------------------------

                else:

                    self.scanner(
                        i,
                        state,
                        tokens
                    )

        # ----------------------------------------------------
        # ACCEPTANCE
        # ----------------------------------------------------

        final_key = (
            "γ",
            ("S",),
            1,
            0
        )

        accepted = any(
            state.key() == final_key
            for state in self.chart[n]
        )

        return accepted

    # --------------------------------------------------------
    # PREDICTOR
    # --------------------------------------------------------

    def predictor(self, index, state):

        symbol = state.next_symbol()

        self.operation_log.append(
            f"PREDICTOR at S[{index}]: "
            f"expand {symbol}"
        )

        for rule in self.grammar[symbol]:

            predicted = State(
                symbol,
                rule,
                0,
                index
            )

            if self.add_state(index, predicted):

                self.predicted_states += 1

    # --------------------------------------------------------
    # SCANNER
    # --------------------------------------------------------

    def scanner(self, index, state, tokens):

        if index >= len(tokens):

            return

        symbol = state.next_symbol()

        if symbol == tokens[index]:

            scanned = State(
                state.lhs,
                state.rhs,
                state.dot + 1,
                state.origin,
                state.backpointers + [
                    (index, tokens[index])
                ]
            )

            if self.add_state(index + 1, scanned):

                self.scanned_states += 1

                self.operation_log.append(
                    f"SCANNER: '{tokens[index]}' "
                    f"from S[{index}] to S[{index + 1}]"
                )

    # --------------------------------------------------------
    # COMPLETER
    # --------------------------------------------------------

    def completer(self, index, completed):

        self.completed_states += 1

        self.operation_log.append(
            f"COMPLETER at S[{index}]: "
            f"{completed.lhs} completed"
        )

        origin = completed.origin

        for previous in list(self.chart[origin]):

            if previous.is_complete():

                continue

            if previous.next_symbol() == completed.lhs:

                advanced = State(
                    previous.lhs,
                    previous.rhs,
                    previous.dot + 1,
                    previous.origin,
                    previous.backpointers + [
                        (
                            completed.origin,
                            completed.lhs
                        )
                    ]
                )

                self.add_state(
                    index,
                    advanced
                )

    # --------------------------------------------------------
    # PRINT CHART
    # --------------------------------------------------------

    def print_chart(self):

        print("\n" + "=" * 70)
        print("EARLEY CHART")
        print("=" * 70)

        for i, states in enumerate(self.chart):

            print(f"\nS[{i}]")
            print("-" * 60)

            if not states:

                print("  EMPTY")

            for number, state in enumerate(
                states,
                1
            ):

                print(
                    f"  {number:02d}. {state}"
                )

    # --------------------------------------------------------
    # PRINT OPERATIONS
    # --------------------------------------------------------

    def print_operations(self, limit=100):

        print("\n" + "=" * 70)
        print("EARLEY OPERATIONS")
        print("=" * 70)

        for i, operation in enumerate(
            self.operation_log[:limit],
            1
        ):

            print(
                f"{i:03d}. {operation}"
            )

        if len(self.operation_log) > limit:

            print(
                f"... {len(self.operation_log) - limit} "
                f"more operations"
            )


# ============================================================
# HAND SIMULATION
# ============================================================

def hand_simulation():

    print("\n" + "=" * 70)
    print("HAND SIMULATION OF EARLEY PARSING")
    print("=" * 70)

    sentence = (
        "the student checks the timetable"
    )

    tokens = sentence.split()

    print("Sentence:", sentence)

    print("\nTokens:")

    for i, token in enumerate(tokens):

        print(
            f"  {i}: {token}"
        )

    print("""
Initial state:

    S[0]:
    γ -> • S, origin=0

Predictor expands S:

    γ -> • S
    S  -> • NP VP

Predictor then expands NP:

    NP -> • Det N
    NP -> • Det N PP
    NP -> • NP PP

Scanner consumes:

    the
    student
    checks
    the
    timetable

Completer advances states whenever
a constituent is completed.

The final chart contains:

    γ -> S •, origin=0

which means the sentence is accepted.
""")


# ============================================================
# RUN BASIC EARLEY EXAMPLE
# ============================================================

def run_basic_example():

    print("\n" + "=" * 70)
    print("EXAMPLE 1: BASIC EARLEY PARSING")
    print("=" * 70)

    sentence = (
        "the student checks the timetable"
    )

    tokens = sentence.split()

    parser = EarleyParser(GRAMMAR)

    start = perf_counter()

    accepted = parser.parse(tokens)

    elapsed = perf_counter() - start

    print("Sentence:", sentence)

    print(
        "Result:",
        "ACCEPTED" if accepted else "REJECTED"
    )

    print("\nNumber of chart states:")

    total_states = sum(
        len(chart_states)
        for chart_states in parser.chart
    )

    print(total_states)

    print(
        "Predicted states:",
        parser.predicted_states
    )

    print(
        "Scanned states:",
        parser.scanned_states
    )

    print(
        "Completed states:",
        parser.completed_states
    )

    print(
        f"Runtime: {elapsed:.6f} seconds"
    )

    parser.print_chart()

    parser.print_operations()


# ============================================================
# AMBIGUITY
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

    print("""
Potential ambiguity:

    VP attachment:
        checks [the timetable] [for the exam]

    NP attachment:
        checks [the timetable for the exam]

Earley parsing can keep multiple compatible
states in the chart instead of committing
immediately to one parse.
""")

    parser = EarleyParser(GRAMMAR)

    start = perf_counter()

    accepted = parser.parse(tokens)

    elapsed = perf_counter() - start

    print(
        "Result:",
        "ACCEPTED" if accepted else "REJECTED"
    )

    total_states = sum(
        len(chart_states)
        for chart_states in parser.chart
    )

    print(
        "Total chart states:",
        total_states
    )

    print(
        "Predicted states:",
        parser.predicted_states
    )

    print(
        "Scanned states:",
        parser.scanned_states
    )

    print(
        "Completed states:",
        parser.completed_states
    )

    print(
        f"Runtime: {elapsed:.6f} seconds"
    )

    parser.print_chart()


# ============================================================
# LEFT RECURSION
# ============================================================

def left_recursion_example():

    print("\n" + "=" * 70)
    print("LEFT RECURSION IN EARLEY PARSING")
    print("=" * 70)

    print("""
The grammar contains:

    NP -> NP PP

This is left-recursive.

Unlike naive recursive descent,
Earley parsing can handle this rule
without infinite recursive calls.

The chart stores states together with
their origin positions.

Repeated states are not inserted again.

Therefore the parser remains finite.
""")


# ============================================================
# BACKPOINTERS
# ============================================================

def backpointer_example():

    print("\n" + "=" * 70)
    print("BACKPOINTERS AND PACKED AMBIGUITY")
    print("=" * 70)

    print("""
Earley states may store backpointer information.

Backpointers can record:

    • where a constituent started
    • which previous state produced it
    • which completed constituent was attached

For an ambiguous sentence, multiple derivations
can share the same chart states.

This compact representation is called
packed ambiguity representation.

Therefore the parser does not need to store
every complete tree separately during recognition.
""")


# ============================================================
# COMPLEXITY COMPARISON
# ============================================================

def complexity_comparison():

    print("\n" + "=" * 70)
    print("COMPLEXITY COMPARISON")
    print("=" * 70)

    print("""
Top-down recursive descent:

    Simple deterministic case:
        approximately O(n)

    Highly ambiguous grammar:
        may become exponential
        because of backtracking.

Earley parser:

    Best case:
        O(n)

    Typical unambiguous CFG:
        O(n^2)

    Worst case:
        O(n^3)

Earley therefore provides a more predictable
worst-case bound for general CFG parsing.
""")


# ============================================================
# UNGRAMMATICAL SENTENCE
# ============================================================

def run_ungrammatical_example():

    print("\n" + "=" * 70)
    print("EXAMPLE 3: UNGRAMMATICAL SENTENCE")
    print("=" * 70)

    sentence = (
        "student checks"
    )

    tokens = sentence.split()

    parser = EarleyParser(GRAMMAR)

    start = perf_counter()

    accepted = parser.parse(tokens)

    elapsed = perf_counter() - start

    print("Sentence:", sentence)

    print(
        "Result:",
        "ACCEPTED" if accepted else "REJECTED"
    )

    total_states = sum(
        len(chart_states)
        for chart_states in parser.chart
    )

    print(
        "Total chart states:",
        total_states
    )

    print(
        f"Runtime: {elapsed:.6f} seconds"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TASK 7 - EARLEY PARSING")
    print("=" * 70)

    hand_simulation()

    run_basic_example()

    run_ambiguous_example()

    left_recursion_example()

    backpointer_example()

    complexity_comparison()

    run_ungrammatical_example()

    print("\n" + "=" * 70)
    print("TASK 7 IMPLEMENTATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()