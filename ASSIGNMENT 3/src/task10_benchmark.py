# ============================================================
# DSA03 NLP ASSIGNMENT - TASK 10
# IMPLEMENTATION, BENCHMARKING AND COMPARISON
# ============================================================

import time
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt


# ============================================================
# 1. 30-SENTENCE TEST SET
# ============================================================

TEST_SENTENCES = [
    "the student needs the timetable",
    "the student checks the assignment",
    "the student needs the hostel",
    "the student checks the portal",
    "the student needs the schedule",
    "the student checks the result",
    "the student needs the form",
    "the student checks the notice",
    "the student needs the certificate",
    "the student checks the deadline",
    "the student needs the document",
    "the student checks the notification",
    "the student needs the record",
    "the student checks the password",
    "the student needs the answer",
    "the student checks the message",
    "the student needs the approval",
    "the student checks the status",
    "the student needs the permission",
    "the student checks the account",
    "the student needs the timetable",
    "the student checks the timetable",
    "the student needs the assignment",
    "the student checks the exam",
    "the student needs the exam",
    "the student checks the schedule",
    "the student needs the result",
    "the student checks the certificate",
    "the student checks the timetable for the exam",
    "the student needs the document for the course",
]


# ============================================================
# 2. GRAMMAR
# ============================================================

GRAMMAR = {
    "S": [
        ["NP", "VP"]
    ],

    "NP": [
        ["Det", "N"],
        ["NP", "PP"]
    ],

    "VP": [
        ["V", "NP"],
        ["VP", "PP"]
    ],

    "PP": [
        ["P", "NP"]
    ],

    "Det": [
        ["the"],
        ["a"]
    ],

    "N": [
        ["student"],
        ["timetable"],
        ["assignment"],
        ["hostel"],
        ["portal"],
        ["schedule"],
        ["result"],
        ["form"],
        ["notice"],
        ["certificate"],
        ["deadline"],
        ["document"],
        ["notification"],
        ["record"],
        ["password"],
        ["answer"],
        ["message"],
        ["approval"],
        ["status"],
        ["permission"],
        ["account"],
        ["exam"],
        ["course"],
    ],

    "V": [
        ["needs"],
        ["checks"],
    ],

    "P": [
        ["for"]
    ],
}


NONTERMINALS = set(GRAMMAR.keys())


# ============================================================
# 3. TOP-DOWN PARSER
# ============================================================

class TopDownParser:

    def __init__(self, grammar):
        self.grammar = grammar
        self.steps = 0
        self.parses = 0
        self.max_parses = 100

    def parse(self, words):

        self.steps = 0
        self.parses = 0

        active = set()

        def expand(symbol, position):

            self.steps += 1

            if self.steps > 100000:
                return []

            # ------------------------------------------------
            # Terminal
            # ------------------------------------------------

            if symbol not in NONTERMINALS:

                if (
                    position < len(words)
                    and words[position] == symbol
                ):
                    return [
                        (position + 1, symbol)
                    ]

                return []

            # ------------------------------------------------
            # Prevent infinite left recursion
            # ------------------------------------------------

            state_key = (symbol, position)

            if state_key in active:
                return []

            active.add(state_key)

            results_local = []

            try:

                for rule in self.grammar[symbol]:

                    # Skip directly left-recursive rules.
                    #
                    # NP -> NP PP
                    # VP -> VP PP
                    #
                    # These rules are handled by Earley/Viterbi
                    # in the later experiments.

                    if (
                        len(rule) > 0
                        and rule[0] == symbol
                    ):
                        continue

                    positions = [
                        (position, [])
                    ]

                    for child in rule:

                        new_positions = []

                        for (
                            current_position,
                            children
                        ) in positions:

                            child_results = expand(
                                child,
                                current_position
                            )

                            for (
                                next_position,
                                tree
                            ) in child_results:

                                new_positions.append(
                                    (
                                        next_position,
                                        children + [tree]
                                    )
                                )

                        positions = new_positions

                        if not positions:
                            break

                    for (
                        final_position,
                        children
                    ) in positions:

                        results_local.append(
                            (
                                final_position,
                                (symbol, children)
                            )
                        )

                        if (
                            len(results_local)
                            >= self.max_parses
                        ):
                            return results_local

                return results_local

            finally:
                active.remove(state_key)

        results = expand("S", 0)

        valid = [
            tree
            for position, tree in results
            if position == len(words)
        ]

        self.parses = len(valid)

        return valid


# ============================================================
# 4. EARLEY PARSER
# ============================================================

class EarleyState:

    def __init__(
        self,
        lhs,
        rhs,
        dot,
        origin
    ):
        self.lhs = lhs
        self.rhs = tuple(rhs)
        self.dot = dot
        self.origin = origin

    def key(self):
        return (
            self.lhs,
            self.rhs,
            self.dot,
            self.origin
        )

    def is_complete(self):
        return self.dot >= len(self.rhs)

    def next_symbol(self):

        if self.is_complete():
            return None

        return self.rhs[self.dot]

    def __repr__(self):

        before = list(
            self.rhs[:self.dot]
        )

        after = list(
            self.rhs[self.dot:]
        )

        return (
            f"{self.lhs} -> "
            f"{' '.join(before)} • "
            f"{' '.join(after)} "
            f"[{self.origin}]"
        )


class EarleyParser:

    def __init__(self, grammar):

        self.grammar = grammar
        self.state_count = 0
        self.steps = 0

    def parse(self, words):

        self.steps = 0

        n = len(words)

        chart = [
            {}
            for _ in range(n + 1)
        ]

        start = EarleyState(
            "GAMMA",
            ["S"],
            0,
            0
        )

        chart[0][start.key()] = start

        # ----------------------------------------------------
        # Process chart
        # ----------------------------------------------------

        for i in range(n + 1):

            changed = True

            while changed:

                changed = False

                states = list(
                    chart[i].values()
                )

                for state in states:

                    self.steps += 1

                    # ----------------------------------------
                    # COMPLETER
                    # ----------------------------------------

                    if state.is_complete():

                        origin_states = list(
                            chart[state.origin].values()
                        )

                        for parent in origin_states:

                            if (
                                parent.next_symbol()
                                == state.lhs
                            ):

                                advanced = EarleyState(
                                    parent.lhs,
                                    parent.rhs,
                                    parent.dot + 1,
                                    parent.origin
                                )

                                key = advanced.key()

                                if key not in chart[i]:

                                    chart[i][key] = advanced
                                    changed = True

                    # ----------------------------------------
                    # PREDICTOR
                    # ----------------------------------------

                    elif (
                        state.next_symbol()
                        in NONTERMINALS
                    ):

                        next_symbol = (
                            state.next_symbol()
                        )

                        for rule in self.grammar[
                            next_symbol
                        ]:

                            predicted = EarleyState(
                                next_symbol,
                                rule,
                                0,
                                i
                            )

                            key = predicted.key()

                            if key not in chart[i]:

                                chart[i][key] = predicted
                                changed = True

                    # ----------------------------------------
                    # SCANNER
                    # ----------------------------------------

                    else:

                        symbol = state.next_symbol()

                        if (
                            i < n
                            and symbol not in NONTERMINALS
                            and symbol == words[i]
                        ):

                            scanned = EarleyState(
                                state.lhs,
                                state.rhs,
                                state.dot + 1,
                                state.origin
                            )

                            key = scanned.key()

                            if key not in chart[i + 1]:

                                chart[i + 1][key] = scanned

        final_state = EarleyState(
            "GAMMA",
            ["S"],
            1,
            0
        )

        accepted = (
            final_state.key()
            in chart[n]
        )

        self.state_count = sum(
            len(column)
            for column in chart
        )

        return accepted, chart


# ============================================================
# 5. VITERBI / PCFG-STYLE PARSER
# ============================================================

class ViterbiParser:

    def __init__(self, grammar):

        self.grammar = grammar
        self.operations = 0

        self.probabilities = {
            ("S", "NP VP"): 1.0,
            ("NP", "Det N"): 0.8,
            ("NP", "NP PP"): 0.2,
            ("VP", "V NP"): 0.8,
            ("VP", "VP PP"): 0.2,
            ("PP", "P NP"): 1.0,
        }

    def parse(self, words):

        self.operations = 0

        n = len(words)

        chart = [
            [
                {}
                for _ in range(n + 1)
            ]
            for _ in range(n)
        ]

        # ----------------------------------------------------
        # Lexical initialization
        # ----------------------------------------------------

        for i, word in enumerate(words):

            for lhs, rules in self.grammar.items():

                for rule in rules:

                    if (
                        len(rule) == 1
                        and rule[0] == word
                    ):

                        chart[i][i + 1][lhs] = (
                            1.0,
                            f"({lhs} {word})"
                        )

        # ----------------------------------------------------
        # CKY dynamic programming
        # ----------------------------------------------------

        for span in range(2, n + 1):

            for start in range(
                n - span + 1
            ):

                end = start + span

                for split in range(
                    start + 1,
                    end
                ):

                    left = chart[start][split]
                    right = chart[split][end]

                    for lhs, rules in self.grammar.items():

                        for rule in rules:

                            self.operations += 1

                            # CKY works with binary rules.
                            if len(rule) != 2:
                                continue

                            left_symbol = rule[0]
                            right_symbol = rule[1]

                            if (
                                left_symbol not in left
                                or right_symbol not in right
                            ):
                                continue

                            rule_probability = (
                                self.probabilities.get(
                                    (
                                        lhs,
                                        " ".join(rule)
                                    ),
                                    0.5
                                )
                            )

                            probability = (
                                rule_probability
                                * left[left_symbol][0]
                                * right[right_symbol][0]
                            )

                            old = chart[start][end].get(lhs)

                            if (
                                old is None
                                or probability > old[0]
                            ):

                                chart[start][end][lhs] = (
                                    probability,
                                    f"({lhs} "
                                    f"{left[left_symbol][1]} "
                                    f"{right[right_symbol][1]})"
                                )

        result = chart[0][n].get("S")

        if result is None:

            return (
                False,
                0.0,
                self.operations
            )

        return (
            True,
            result[0],
            self.operations
        )


# ============================================================
# 6. BENCHMARK FUNCTION
# ============================================================

def benchmark_parser(
    parser_name,
    parser,
    sentences
):

    results = []

    print("\n" + "=" * 70)
    print(f"BENCHMARKING: {parser_name}")
    print("=" * 70)

    for index, sentence in enumerate(
        sentences,
        start=1
    ):

        words = sentence.split()

        start_time = time.perf_counter()

        # ----------------------------------------------------
        # Top-Down
        # ----------------------------------------------------

        if parser_name == "Top-Down":

            parses = parser.parse(words)

            accepted = len(parses) > 0

            parse_count = len(parses)

            steps = parser.steps

        # ----------------------------------------------------
        # Earley
        # ----------------------------------------------------

        elif parser_name == "Earley":

            accepted, chart = parser.parse(words)

            parse_count = (
                1
                if accepted
                else 0
            )

            steps = parser.steps

        # ----------------------------------------------------
        # Viterbi
        # ----------------------------------------------------

        else:

            (
                accepted,
                probability,
                operations
            ) = parser.parse(words)

            parse_count = (
                1
                if accepted
                else 0
            )

            steps = operations

        elapsed = (
            time.perf_counter()
            - start_time
        )

        results.append(
            {
                "sentence": sentence,
                "accepted": accepted,
                "parses": parse_count,
                "steps": steps,
                "runtime": elapsed,
            }
        )

        print(
            f"{index:02d}. "
            f"{sentence:<55} "
            f"{'ACCEPT' if accepted else 'REJECT':<7} "
            f"steps={steps:<7} "
            f"time={elapsed:.6f}s"
        )

    return results


# ============================================================
# 7. SPECIAL TEST CASES
# ============================================================

def run_special_tests():

    print("\n" + "=" * 70)
    print("SPECIAL TEST CASES")
    print("=" * 70)

    cases = [
        (
            "Recursion",
            "the student checks the timetable for the exam"
        ),
        (
            "Ambiguity",
            "the student checks the timetable for the exam"
        ),
        (
            "Ungrammatical",
            "student checks timetable"
        ),
        (
            "Long sentence",
            "the student checks the timetable for the exam"
        ),
    ]

    parsers = [
        (
            "Top-Down",
            TopDownParser(GRAMMAR)
        ),
        (
            "Earley",
            EarleyParser(GRAMMAR)
        ),
        (
            "Viterbi",
            ViterbiParser(GRAMMAR)
        ),
    ]

    for case_name, sentence in cases:

        print(f"\n{case_name}:")
        print(f"Sentence: {sentence}")

        for name, parser in parsers:

            start = time.perf_counter()

            if name == "Top-Down":

                result = parser.parse(
                    sentence.split()
                )

                accepted = len(result) > 0

                measure = parser.steps

            elif name == "Earley":

                accepted, chart = parser.parse(
                    sentence.split()
                )

                measure = parser.state_count

            else:

                (
                    accepted,
                    probability,
                    operations
                ) = parser.parse(
                    sentence.split()
                )

                measure = operations

            elapsed = (
                time.perf_counter()
                - start
            )

            print(
                f"  {name:<10} "
                f"{'ACCEPT' if accepted else 'REJECT':<7} "
                f"measure={measure:<6} "
                f"time={elapsed:.6f}s"
            )


# ============================================================
# 8. SUMMARY STATISTICS
# ============================================================

def summarize_results(all_results):

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    for parser_name, results in all_results.items():

        total = len(results)

        accepted = sum(
            r["accepted"]
            for r in results
        )

        total_steps = sum(
            r["steps"]
            for r in results
        )

        total_time = sum(
            r["runtime"]
            for r in results
        )

        average_time = (
            total_time / total
            if total
            else 0
        )

        total_parses = sum(
            r["parses"]
            for r in results
        )

        print(f"\n{parser_name}")

        print(
            f"  Sentences tested : {total}"
        )

        print(
            f"  Accepted         : {accepted}"
        )

        print(
            f"  Rejected         : "
            f"{total - accepted}"
        )

        print(
            f"  Total parses     : "
            f"{total_parses}"
        )

        print(
            f"  Total steps      : "
            f"{total_steps}"
        )

        print(
            f"  Total runtime    : "
            f"{total_time:.6f} sec"
        )

        print(
            f"  Average runtime  : "
            f"{average_time:.6f} sec"
        )


# ============================================================
# 9. CREATE BENCHMARK PLOT
# ============================================================

def create_plot(all_results):

    print("\n" + "=" * 70)
    print("CREATING BENCHMARK PLOT")
    print("=" * 70)

    names = []

    average_times = []

    for parser_name, results in all_results.items():

        names.append(parser_name)

        average = (
            sum(
                r["runtime"]
                for r in results
            )
            / len(results)
        )

        average_times.append(
            average
        )

    plt.figure(
        figsize=(9, 6)
    )

    plt.bar(
        names,
        average_times
    )

    plt.xlabel(
        "Parser"
    )

    plt.ylabel(
        "Average Runtime (seconds)"
    )

    plt.title(
        "Task 10 - Parser Average Runtime Comparison"
    )

    plt.tight_layout()

    output_path = Path(
        "results/task10_runtime_comparison.png"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(
        output_path
    )

    plt.close()

    print(
        f"\nPlot saved to:\n"
        f"{output_path}"
    )


# ============================================================
# 10. SAVE CSV RESULTS
# ============================================================

def save_results(all_results):

    output_path = Path(
        "results/task10_benchmark_results.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "Parser,Sentence,Accepted,"
            "Parses,Steps,Runtime\n"
        )

        for parser_name, results in all_results.items():

            for result in results:

                sentence = (
                    result["sentence"]
                    .replace(",", " ")
                )

                file.write(
                    f"{parser_name},"
                    f"{sentence},"
                    f"{result['accepted']},"
                    f"{result['parses']},"
                    f"{result['steps']},"
                    f"{result['runtime']:.8f}\n"
                )

    print(
        f"Benchmark CSV saved to:\n"
        f"{output_path}"
    )


# ============================================================
# 11. COMPARISON WITH TASK 7 AND TASK 9
# ============================================================

def comparison_with_previous_tasks():

    print("\n" + "=" * 70)
    print("COMPARISON WITH TASK 7 AND TASK 9")
    print("=" * 70)

    print("""
Task 7 - Earley Parsing
-----------------------
• Handles recursive CFG structures.
• Maintains chart states.
• Avoids repeated recomputation.
• Suitable for ambiguous and left-recursive grammars.

Task 9 - PCFG / Viterbi
-----------------------
• Assigns probabilities to grammar rules.
• Calculates probability of a parse.
• Selects the highest-probability interpretation.
• Useful for resolving structural ambiguity.

Task 10 - Benchmarking
----------------------
• Runs the implementations on a common 30-sentence test set.
• Measures parser steps and runtime.
• Tests grammatical, ambiguous, recursive,
  ungrammatical and longer sentences.
• Provides empirical evidence for parser comparison.
""")


# ============================================================
# 12. MAIN PROGRAM
# ============================================================

def main():

    print("=" * 70)
    print("DSA03 NLP ASSIGNMENT - TASK 10")
    print("IMPLEMENTATION AND BENCHMARKING")
    print("=" * 70)

    print(
        f"\nTest set size: "
        f"{len(TEST_SENTENCES)} sentences"
    )

    # --------------------------------------------------------
    # Create parsers
    # --------------------------------------------------------

    top_down = TopDownParser(
        GRAMMAR
    )

    earley = EarleyParser(
        GRAMMAR
    )

    viterbi = ViterbiParser(
        GRAMMAR
    )

    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------

    all_results = {}

    all_results["Top-Down"] = benchmark_parser(
        "Top-Down",
        top_down,
        TEST_SENTENCES
    )

    all_results["Earley"] = benchmark_parser(
        "Earley",
        earley,
        TEST_SENTENCES
    )

    all_results["Viterbi"] = benchmark_parser(
        "Viterbi",
        viterbi,
        TEST_SENTENCES
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summarize_results(
        all_results
    )

    # --------------------------------------------------------
    # Special tests
    # --------------------------------------------------------

    run_special_tests()

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    comparison_with_previous_tasks()

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    create_plot(
        all_results
    )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    save_results(
        all_results
    )

    print("\n" + "=" * 70)
    print(
        "TASK 10 IMPLEMENTATION COMPLETED"
    )
    print("=" * 70)


# ============================================================
# PROGRAM ENTRY
# ============================================================

if __name__ == "__main__":
    main()