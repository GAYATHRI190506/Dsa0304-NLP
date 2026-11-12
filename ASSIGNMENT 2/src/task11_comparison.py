from pathlib import Path
import csv


# ============================================================
# TASK 11 - MODEL AND TAGGER COMPARISON & RECOMMENDATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------
# MEASURED RESULTS FROM TASKS 3, 4, 5, 7, 8, 9 AND 10
# ------------------------------------------------------------

models = [
    {
        "name": "MLE Bigram",
        "type": "Language Model",
        "performance": "Perplexity",
        "score": 6.2825,
        "runtime": "Not benchmarked",
        "complexity": "Low",
        "manual_effort": "Low",
        "smoothing": "No",
    },
    {
        "name": "Add-k (k=0.01)",
        "type": "Language Model",
        "performance": "Perplexity",
        "score": 5.2027,
        "runtime": "Not benchmarked",
        "complexity": "Low",
        "manual_effort": "Low",
        "smoothing": "Yes",
    },
    {
        "name": "Good-Turing",
        "type": "Language Model",
        "performance": "Perplexity",
        "score": 5.6818,
        "runtime": "Not benchmarked",
        "complexity": "Medium",
        "manual_effort": "Low",
        "smoothing": "Yes",
    },
    {
        "name": "Katz Backoff",
        "type": "Language Model",
        "performance": "Perplexity",
        "score": 3.5081,
        "runtime": "Not benchmarked",
        "complexity": "Medium",
        "manual_effort": "Low",
        "smoothing": "Yes",
    },
    {
        "name": "Stupid Backoff",
        "type": "Language Model",
        "performance": "Perplexity",
        "score": 3.1869,
        "runtime": "Not benchmarked",
        "complexity": "Low",
        "manual_effort": "Low",
        "smoothing": "Yes",
    },
    {
        "name": "Linear Interpolation",
        "type": "Language Model",
        "performance": "Perplexity",
        "score": 3.3953,
        "runtime": "Not benchmarked",
        "complexity": "Medium",
        "manual_effort": "Low",
        "smoothing": "Yes",
    },
    {
        "name": "Deleted Interpolation",
        "type": "Language Model",
        "performance": "Perplexity",
        "score": 3.5179,
        "runtime": "Not benchmarked",
        "complexity": "Medium",
        "manual_effort": "Low",
        "smoothing": "Yes",
    },
]


taggers = [
    {
        "name": "Rule-Based Tagger",
        "accuracy": 100.00,
        "runtime": 0.1031,
        "complexity": "Low",
        "manual_effort": "Medium",
        "main_strength": "Fast and interpretable",
    },
    {
        "name": "HMM Viterbi",
        "accuracy": 97.61,
        "runtime": 1.7012,
        "complexity": "Medium",
        "manual_effort": "High",
        "main_strength": "Statistical sequence modelling",
    },
    {
        "name": "Brill Tagger",
        "accuracy": 100.00,
        "runtime": 5.71,
        "complexity": "Medium",
        "manual_effort": "Medium",
        "main_strength": "Transformation-based correction",
    },
]


# ------------------------------------------------------------
# HELPER FUNCTION
# ------------------------------------------------------------

def write_line(file, text=""):
    file.write(text + "\n")


# ------------------------------------------------------------
# SAVE CSV COMPARISON
# ------------------------------------------------------------

csv_path = RESULTS_DIR / "task11_model_comparison.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    writer.writerow([
        "Model",
        "Type",
        "Performance Measure",
        "Score",
        "Runtime",
        "Complexity",
        "Manual Effort",
        "Smoothing"
    ])

    for model in models:
        writer.writerow([
            model["name"],
            model["type"],
            model["performance"],
            model["score"],
            model["runtime"],
            model["complexity"],
            model["manual_effort"],
            model["smoothing"]
        ])

    for tagger in taggers:
        writer.writerow([
            tagger["name"],
            "POS Tagger",
            "Accuracy",
            tagger["accuracy"],
            f"{tagger['runtime']:.4f} sec",
            tagger["complexity"],
            tagger["manual_effort"],
            "N/A"
        ])


# ------------------------------------------------------------
# TASK 11 REPORT
# ------------------------------------------------------------

report_path = RESULTS_DIR / "task11_results.txt"

with open(report_path, "w", encoding="utf-8") as f:

    write_line(f, "=" * 75)
    write_line(f, "TASK 11 - MODEL AND TAGGER COMPARISON")
    write_line(f, "=" * 75)
    write_line(f)

    # --------------------------------------------------------
    # LANGUAGE MODEL COMPARISON
    # --------------------------------------------------------

    write_line(f, "1. LANGUAGE MODEL COMPARISON")
    write_line(f, "-" * 75)

    write_line(
        f,
        f"{'Model':25} {'Perplexity':12} {'Complexity':12} {'Manual Effort':15}"
    )
    write_line(f, "-" * 75)

    for model in models:
        write_line(
            f,
            f"{model['name']:25} "
            f"{model['score']:<12.4f} "
            f"{model['complexity']:<12} "
            f"{model['manual_effort']:<15}"
        )

    write_line(f)

    # --------------------------------------------------------
    # BEST LANGUAGE MODEL
    # --------------------------------------------------------

    best_lm = min(models, key=lambda x: x["score"])

    write_line(f, "BEST LANGUAGE MODEL")
    write_line(f, "-" * 75)
    write_line(f, f"Model      : {best_lm['name']}")
    write_line(f, f"Perplexity : {best_lm['score']:.4f}")
    write_line(
        f,
        "Reason     : It achieved the lowest measured perplexity among "
        "the compared language models."
    )
    write_line(f)

    # --------------------------------------------------------
    # POS TAGGER COMPARISON
    # --------------------------------------------------------

    write_line(f, "2. POS TAGGER COMPARISON")
    write_line(f, "-" * 75)

    write_line(
        f,
        f"{'Tagger':25} {'Accuracy':12} {'Runtime(s)':12} "
        f"{'Complexity':12} {'Manual Effort':15}"
    )
    write_line(f, "-" * 75)

    for tagger in taggers:
        write_line(
            f,
            f"{tagger['name']:25} "
            f"{tagger['accuracy']:<12.2f} "
            f"{tagger['runtime']:<12.4f} "
            f"{tagger['complexity']:<12} "
            f"{tagger['manual_effort']:<15}"
        )

    write_line(f)

    # --------------------------------------------------------
    # BEST TAGGER
    # --------------------------------------------------------

    best_tagger = max(taggers, key=lambda x: x["accuracy"])

    write_line(f, "BEST POS TAGGER BY MEASURED ACCURACY")
    write_line(f, "-" * 75)
    write_line(f, f"Model    : {best_tagger['name']}")
    write_line(f, f"Accuracy : {best_tagger['accuracy']:.2f}%")
    write_line(
        f,
        "Reason   : It achieved the highest measured test accuracy."
    )
    write_line(f)

    # --------------------------------------------------------
    # RUNTIME COMPARISON
    # --------------------------------------------------------

    write_line(f, "3. RUNTIME COMPARISON")
    write_line(f, "-" * 75)

    write_line(
        f,
        "Rule-based tagger runtime : 0.1031 seconds"
    )
    write_line(
        f,
        "HMM Viterbi runtime       : 1.7012 seconds"
    )
    write_line(
        f,
        "Brill training time       : 5.71 seconds"
    )
    write_line(f)

    write_line(
        f,
        "The rule-based tagger was the fastest measured tagger."
    )
    write_line(f)

    # --------------------------------------------------------
    # MANUAL EFFORT
    # --------------------------------------------------------

    write_line(f, "4. MANUAL EFFORT")
    write_line(f, "-" * 75)

    write_line(
        f,
        "Task 6 used a 22-tag custom tagset."
    )
    write_line(
        f,
        "3,000 tokens were annotated."
    )
    write_line(
        f,
        "500 tokens were double-tagged for agreement analysis."
    )
    write_line(
        f,
        "Agreement = 96.40%"
    )
    write_line(
        f,
        "Cohen's kappa = 0.9597"
    )
    write_line(f)

    write_line(
        f,
        "The tagging design therefore required a non-trivial manual "
        "annotation effort before statistical taggers could be evaluated."
    )
    write_line(f)

    # --------------------------------------------------------
    # STRENGTHS AND WEAKNESSES
    # --------------------------------------------------------

    write_line(f, "5. STRENGTHS AND WEAKNESSES")
    write_line(f, "-" * 75)

    write_line(f, "MLE Bigram:")
    write_line(
        f,
        "  Strength: Simple and easy to calculate."
    )
    write_line(
        f,
        "  Weakness : High sparsity and zero-probability problem."
    )
    write_line(f)

    write_line(f, "Add-k:")
    write_line(
        f,
        "  Strength: Simple smoothing and removes zero probabilities."
    )
    write_line(
        f,
        "  Weakness : Performance depends on the selected k value."
    )
    write_line(f)

    write_line(f, "Good-Turing:")
    write_line(
        f,
        "  Strength: Adjusts probabilities using frequency-of-frequency "
        "information."
    )
    write_line(
        f,
        "  Weakness : More complicated than Add-k."
    )
    write_line(f)

    write_line(f, "Katz Backoff:")
    write_line(
        f,
        "  Strength: Uses lower-order models when higher-order evidence "
        "is unavailable."
    )
    write_line(
        f,
        "  Weakness : More implementation complexity."
    )
    write_line(f)

    write_line(f, "Stupid Backoff:")
    write_line(
        f,
        "  Strength: Lowest measured perplexity and simple implementation."
    )
    write_line(
        f,
        "  Weakness : Uses a fixed backoff factor rather than a normalized "
        "probability model."
    )
    write_line(f)

    write_line(f, "Rule-Based Tagger:")
    write_line(
        f,
        "  Strength: Very fast, interpretable and achieved 100% test "
        "accuracy on this corpus."
    )
    write_line(
        f,
        "  Weakness : Requires manually designed lexical dictionaries and "
        "rules."
    )
    write_line(f)

    write_line(f, "HMM Viterbi:")
    write_line(
        f,
        "  Strength: Models dependencies between neighbouring tags."
    )
    write_line(
        f,
        "  Weakness : Slower than the rule-based approach and requires "
        "statistical training."
    )
    write_line(f)

    write_line(f, "Brill Tagger:")
    write_line(
        f,
        "  Strength: Learns transformations that correct initial tagging "
        "errors."
    )
    write_line(
        f,
        "  Weakness : Training is slower than the rule-based tagger and "
        "the initial tagger was already highly accurate on this corpus."
    )
    write_line(f)

    # --------------------------------------------------------
    # FINAL RECOMMENDATION
    # --------------------------------------------------------

    write_line(f, "6. FINAL RECOMMENDATION")
    write_line(f, "-" * 75)

    write_line(
        f,
        "Recommended language model: Stupid Backoff"
    )
    write_line(
        f,
        "Measured perplexity: 3.1869"
    )
    write_line(
        f,
        "It achieved the lowest perplexity among the Task 4 models."
    )
    write_line(f)

    write_line(
        f,
        "Recommended POS tagger for this public-transport corpus: "
        "Rule-Based Tagger"
    )
    write_line(
        f,
        "Measured test accuracy: 100.00%"
    )
    write_line(
        f,
        "Measured runtime: 0.1031 seconds"
    )
    write_line(
        f,
        "It provides the best combination of measured accuracy, speed "
        "and interpretability for this controlled corpus."
    )
    write_line(f)

    write_line(
        f,
        "HMM Viterbi is a useful alternative when sequence-based "
        "statistical modelling is preferred."
    )
    write_line(
        f,
        "Brill is useful as a transformation-based correction approach, "
        "but its improvement was small because the initial tagger was "
        "already highly accurate."
    )
    write_line(f)

    # --------------------------------------------------------
    # LIMITATION
    # --------------------------------------------------------

    write_line(f, "7. LIMITATION")
    write_line(f, "-" * 75)

    write_line(
        f,
        "The corpus is synthetic and highly regular. Therefore, the "
        "measured 100% rule-based and Brill accuracy should not be "
        "interpreted as evidence that the same performance will occur "
        "on unrestricted real-world speech."
    )

    write_line(
        f,
        "The Day-4 challenge in Task 12 is used to test robustness "
        "under more difficult conditions such as unseen vocabulary "
        "and code-mixed speech."
    )

    write_line(f)
    write_line(f, "=" * 75)
    write_line(f, "TASK 11 COMPLETED")
    write_line(f, "=" * 75)


print("=" * 75)
print("TASK 11 - MODEL AND TAGGER COMPARISON")
print("=" * 75)

print()
print("LANGUAGE MODEL COMPARISON")
print("-" * 75)

for model in models:
    print(
        f"{model['name']:25} "
        f"Perplexity = {model['score']:.4f}"
    )

print()
print("BEST LANGUAGE MODEL")
print("-" * 75)
print(f"Stupid Backoff")
print(f"Perplexity = {best_lm['score']:.4f}")

print()
print("POS TAGGER COMPARISON")
print("-" * 75)

for tagger in taggers:
    print(
        f"{tagger['name']:25} "
        f"Accuracy = {tagger['accuracy']:.2f}%   "
        f"Time = {tagger['runtime']:.4f}s"
    )

print()
print("RECOMMENDED POS TAGGER")
print("-" * 75)
print("Rule-Based Tagger")
print("Accuracy = 100.00%")
print("Runtime = 0.1031 seconds")

print()
print("FILES CREATED")
print("-" * 75)
print(f"1. {csv_path.relative_to(BASE_DIR)}")
print(f"2. {report_path.relative_to(BASE_DIR)}")

print()
print("=" * 75)
print("TASK 11 COMPLETED SUCCESSFULLY")
print("=" * 75)