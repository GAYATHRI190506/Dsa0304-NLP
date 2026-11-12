# ============================================================
# DSA03 NLP ASSIGNMENT - TASK 11
# COMPARISON AND RECOMMENDATION
# ============================================================

from pathlib import Path
import csv


# ============================================================
# 1. TASK 10 RESULT FILE
# ============================================================

RESULT_FILE = Path(
    "results/task10_benchmark_results.csv"
)


# ============================================================
# 2. READ TASK 10 RESULTS
# ============================================================

def read_task10_results():

    if not RESULT_FILE.exists():

        print(
            "ERROR: Task 10 result file was not found."
        )

        print(
            f"Expected file:\n{RESULT_FILE}"
        )

        return {}

    data = {}

    with open(
        RESULT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            parser = row["Parser"]

            if parser not in data:
                data[parser] = []

            data[parser].append(
                {
                    "sentence": row["Sentence"],
                    "accepted": (
                        row["Accepted"].lower()
                        == "true"
                    ),
                    "parses": int(
                        row["Parses"]
                    ),
                    "steps": int(
                        row["Steps"]
                    ),
                    "runtime": float(
                        row["Runtime"]
                    ),
                }
            )

    return data


# ============================================================
# 3. CALCULATE STATISTICS
# ============================================================

def calculate_statistics(data):

    statistics = {}

    for parser, results in data.items():

        total = len(results)

        accepted = sum(
            result["accepted"]
            for result in results
        )

        rejected = (
            total - accepted
        )

        total_steps = sum(
            result["steps"]
            for result in results
        )

        total_runtime = sum(
            result["runtime"]
            for result in results
        )

        average_steps = (
            total_steps / total
            if total > 0
            else 0
        )

        average_runtime = (
            total_runtime / total
            if total > 0
            else 0
        )

        acceptance_rate = (
            accepted / total * 100
            if total > 0
            else 0
        )

        statistics[parser] = {
            "total": total,
            "accepted": accepted,
            "rejected": rejected,
            "total_steps": total_steps,
            "average_steps": average_steps,
            "total_runtime": total_runtime,
            "average_runtime": average_runtime,
            "acceptance_rate": acceptance_rate,
        }

    return statistics


# ============================================================
# 4. PRINT EMPIRICAL COMPARISON
# ============================================================

def print_empirical_comparison(statistics):

    print("\n" + "=" * 80)
    print("TASK 11 - EMPIRICAL COMPARISON")
    print("=" * 80)

    print(
        f"{'Parser':<15}"
        f"{'Accepted':<12}"
        f"{'Rejected':<12}"
        f"{'Avg Steps':<15}"
        f"{'Avg Time(s)':<15}"
        f"{'Coverage':<12}"
    )

    print("-" * 80)

    for parser, stats in statistics.items():

        print(
            f"{parser:<15}"
            f"{stats['accepted']:<12}"
            f"{stats['rejected']:<12}"
            f"{stats['average_steps']:<15.2f}"
            f"{stats['average_runtime']:<15.6f}"
            f"{stats['acceptance_rate']:<12.2f}%"
        )


# ============================================================
# 5. QUALITATIVE COMPARISON
# ============================================================

def qualitative_comparison():

    print("\n" + "=" * 80)
    print("QUALITATIVE COMPARISON")
    print("=" * 80)

    comparison = {

        "Correctness / Coverage": {
            "Top-Down":
                "Good for simple CFG sentences, "
                "but direct left recursion must be "
                "controlled or eliminated.",

            "Earley":
                "Strong CFG coverage and can handle "
                "left recursion and ambiguous structures.",

            "Viterbi":
                "Provides the best parse under a "
                "probabilistic model, but the CKY-style "
                "implementation requires suitable grammar form."
        },

        "Ambiguity Handling": {
            "Top-Down":
                "Can generate multiple parses, but "
                "ambiguity can increase search and backtracking.",

            "Earley":
                "Naturally represents multiple analyses "
                "using chart states and packed-style structures.",

            "Viterbi":
                "Ranks competing parses using probabilities "
                "and selects the highest-scoring analysis."
        },

        "Runtime / Space": {
            "Top-Down":
                "Simple and lightweight for small "
                "unambiguous grammars, but recursive "
                "search can become expensive.",

            "Earley":
                "Uses chart parsing to avoid repeated "
                "work. Worst-case CFG complexity is O(n^3).",

            "Viterbi":
                "Dynamic programming avoids repeated "
                "subproblems and is efficient for "
                "CNF-style grammars."
        },

        "Ungrammatical Input": {
            "Top-Down":
                "Rejects input when no complete derivation "
                "is found, but recursive search must be controlled.",

            "Earley":
                "Rejects input when the completed start state "
                "is absent from the final chart.",

            "Viterbi":
                "Rejects input when no valid complete "
                "probabilistic parse is available."
        },

        "Extension / Maintenance": {
            "Top-Down":
                "Easy to understand and modify for small grammars.",

            "Earley":
                "More complex implementation but highly "
                "flexible for CFG extensions.",

            "Viterbi":
                "Good when probabilities and ranking are "
                "required, but grammar conversion and "
                "probability estimation add complexity."
        },
    }

    for criterion, parser_data in comparison.items():

        print(
            f"\n--- {criterion} ---"
        )

        for parser, explanation in parser_data.items():

            print(
                f"\n{parser}:"
            )

            print(
                f"  {explanation}"
            )


# ============================================================
# 6. ADVANTAGES AND LIMITATIONS
# ============================================================

def advantages_and_limitations():

    print("\n" + "=" * 80)
    print("ADVANTAGES AND LIMITATIONS")
    print("=" * 80)

    data = {

        "Top-Down": {
            "advantages": [
                "Simple recursive implementation.",
                "Easy to understand and debug.",
                "Useful for small deterministic grammars.",
                "Can produce explicit parse trees."
            ],

            "limitations": [
                "Left recursion can cause infinite recursion.",
                "Backtracking can increase runtime.",
                "Ambiguity may produce many alternative parses.",
                "Performance decreases for complex grammars."
            ]
        },

        "Earley": {
            "advantages": [
                "Handles general context-free grammars.",
                "Supports left-recursive grammar rules.",
                "Works well with ambiguous structures.",
                "Uses chart states to avoid repeated work."
            ],

            "limitations": [
                "Implementation is more complex.",
                "Can require more memory than simple parsing.",
                "Worst-case complexity is O(n^3)."
            ]
        },

        "Viterbi": {
            "advantages": [
                "Ranks parses using probabilities.",
                "Selects the most probable interpretation.",
                "Dynamic programming reduces repeated computation.",
                "Useful for ambiguity resolution."
            ],

            "limitations": [
                "Requires a probabilistic grammar.",
                "Probabilities depend on training data.",
                "CKY-style parsing normally requires CNF/binary rules.",
                "Rare or unseen constructions may receive zero probability."
            ]
        }
    }

    for parser, information in data.items():

        print(
            f"\n{'-' * 20} {parser} {'-' * 20}"
        )

        print("\nAdvantages:")

        for item in information["advantages"]:
            print(
                f"  + {item}"
            )

        print("\nLimitations:")

        for item in information["limitations"]:
            print(
                f"  - {item}"
            )


# ============================================================
# 7. RECOMMENDATION
# ============================================================

def recommendation():

    print("\n" + "=" * 80)
    print("RECOMMENDATION FOR UNIVERSITY HELPDESK")
    print("=" * 80)

    print("""
Recommended architecture:

1. Earley Parser
   Use Earley parsing as the main syntactic parser.

   Reason:
   - Handles general CFGs.
   - Supports recursion.
   - Handles ambiguous structures.
   - Provides chart states that can be inspected.
   - Suitable for complex university-helpdesk queries.

2. Feature Structures
   Combine Earley parsing with feature structures
   for agreement, number, person, case and
   subcategorization constraints.

3. PCFG / Viterbi
   Use probabilistic ranking after generating
   candidate parses.

   This is useful when a sentence has multiple
   structurally valid interpretations.

4. Top-Down Parsing
   Keep Top-Down parsing mainly as a simple
   baseline/reference parser.

Overall recommendation:

    Earley + Feature Structures + PCFG Ranking

This combination provides a balance between:

    grammatical coverage
    ambiguity handling
    linguistic constraints
    robustness
    probabilistic ranking
    maintainability
""")


# ============================================================
# 8. DECISION MATRIX
# ============================================================

def decision_matrix():

    print("\n" + "=" * 80)
    print("DECISION MATRIX")
    print("=" * 80)

    headers = [
        "Criterion",
        "Top-Down",
        "Earley",
        "Viterbi"
    ]

    rows = [
        (
            "CFG Coverage",
            "Medium",
            "Excellent",
            "Good"
        ),

        (
            "Left Recursion",
            "Weak",
            "Excellent",
            "Limited"
        ),

        (
            "Ambiguity",
            "Good",
            "Excellent",
            "Excellent"
        ),

        (
            "Probability Ranking",
            "No",
            "No",
            "Yes"
        ),

        (
            "Ungrammatical Input",
            "Good",
            "Excellent",
            "Good"
        ),

        (
            "Implementation Simplicity",
            "Excellent",
            "Medium",
            "Medium"
        ),

        (
            "Complex Grammar",
            "Limited",
            "Excellent",
            "Good"
        ),

        (
            "Helpdesk Suitability",
            "Medium",
            "Excellent",
            "Excellent"
        )
    ]

    print(
        f"{headers[0]:<30}"
        f"{headers[1]:<15}"
        f"{headers[2]:<15}"
        f"{headers[3]:<15}"
    )

    print("-" * 75)

    for row in rows:

        print(
            f"{row[0]:<30}"
            f"{row[1]:<15}"
            f"{row[2]:<15}"
            f"{row[3]:<15}"
        )


# ============================================================
# 9. SAVE REPORT
# ============================================================

def save_report(statistics):

    output_path = Path(
        "results/task11_comparison_report.txt"
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
            "DSA03 NLP ASSIGNMENT - TASK 11\n"
        )

        file.write(
            "COMPARISON AND RECOMMENDATION\n\n"
        )

        file.write(
            "EMPIRICAL RESULTS\n"
        )

        file.write(
            "=================\n\n"
        )

        for parser, stats in statistics.items():

            file.write(
                f"{parser}\n"
            )

            file.write(
                f"Sentences tested: "
                f"{stats['total']}\n"
            )

            file.write(
                f"Accepted: "
                f"{stats['accepted']}\n"
            )

            file.write(
                f"Rejected: "
                f"{stats['rejected']}\n"
            )

            file.write(
                f"Average steps: "
                f"{stats['average_steps']:.2f}\n"
            )

            file.write(
                f"Average runtime: "
                f"{stats['average_runtime']:.6f} sec\n"
            )

            file.write(
                f"Acceptance rate: "
                f"{stats['acceptance_rate']:.2f}%\n\n"
            )

        file.write(
            "RECOMMENDATION\n"
        )

        file.write(
            "==============\n\n"
        )

        file.write(
            "The recommended architecture for the "
            "university helpdesk is Earley parsing "
            "combined with feature structures and "
            "PCFG/Viterbi ranking.\n"
        )

        file.write(
            "\nEarley provides broad CFG coverage, "
            "recursion handling and ambiguity support. "
            "Feature structures add linguistic constraints "
            "such as agreement and subcategorization. "
            "PCFG/Viterbi ranking can then select the "
            "most probable interpretation among "
            "multiple valid parses.\n"
        )

    print(
        f"\nTask 11 report saved to:\n"
        f"{output_path}"
    )


# ============================================================
# 10. MAIN
# ============================================================

def main():

    print("=" * 80)
    print("DSA03 NLP ASSIGNMENT - TASK 11")
    print("COMPARISON AND RECOMMENDATION")
    print("=" * 80)

    # --------------------------------------------------------
    # Read Task 10 results
    # --------------------------------------------------------

    data = read_task10_results()

    if not data:

        print(
            "\nTask 10 results could not be loaded."
        )

        print(
            "Please run Task 10 first."
        )

        return

    print(
        f"\nLoaded Task 10 results for:"
    )

    for parser in data:

        print(
            f"  - {parser}: "
            f"{len(data[parser])} sentences"
        )

    # --------------------------------------------------------
    # Calculate statistics
    # --------------------------------------------------------

    statistics = calculate_statistics(
        data
    )

    # --------------------------------------------------------
    # Empirical comparison
    # --------------------------------------------------------

    print_empirical_comparison(
        statistics
    )

    # --------------------------------------------------------
    # Qualitative comparison
    # --------------------------------------------------------

    qualitative_comparison()

    # --------------------------------------------------------
    # Advantages and limitations
    # --------------------------------------------------------

    advantages_and_limitations()

    # --------------------------------------------------------
    # Decision matrix
    # --------------------------------------------------------

    decision_matrix()

    # --------------------------------------------------------
    # Recommendation
    # --------------------------------------------------------

    recommendation()

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    save_report(
        statistics
    )

    print("\n" + "=" * 80)
    print(
        "TASK 11 IMPLEMENTATION COMPLETED"
    )
    print("=" * 80)


# ============================================================
# PROGRAM ENTRY
# ============================================================

if __name__ == "__main__":
    main()