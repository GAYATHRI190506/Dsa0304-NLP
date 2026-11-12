# ============================================================
# DSA03 NLP ASSIGNMENT - TASK 12
# ROBUSTNESS TO CONVERSATIONAL AND ADVERSARIAL INPUT
# ============================================================

from pathlib import Path
import re


# ============================================================
# ORIGINAL GRAMMAR
# ============================================================

ORIGINAL_GRAMMAR = {
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
        ["department"],
        ["faculty"],
        ["teacher"],
        ["class"],
        ["fee"],
        ["fees"],
        ["scholarship"],
        ["registration"],
        ["library"],
        ["card"],
        ["mark"],
        ["marks"],
    ],

    "V": [
        ["need"],
        ["needs"],
        ["check"],
        ["checks"],
        ["want"],
        ["wants"],
        ["get"],
        ["gets"],
        ["provide"],
        ["provides"],
        ["show"],
        ["shows"],
    ],

    "P": [
        ["for"],
        ["of"],
        ["from"],
        ["to"],
        ["with"],
    ],
}


# ============================================================
# ADAPTED / ROBUST GRAMMAR
# ============================================================

ADAPTED_GRAMMAR = {

    "S": [
        ["NP", "VP"],
        ["S", "Conj", "S"],
        ["Question"],
        ["S", "Clause"]
    ],

    "Question": [
        ["Aux", "NP", "VP"],
        ["Wh", "Aux", "NP", "VP"],
        ["Wh", "VP"]
    ],

    "Clause": [
        ["RelClause"],
        ["PP"]
    ],

    "RelClause": [
        ["Rel", "VP"],
        ["Rel", "NP", "VP"]
    ],

    "NP": [
        ["Det", "N"],
        ["Pronoun"],
        ["NP", "PP"],
        ["NP", "RelClause"],
        ["NP", "Adj"]
    ],

    "VP": [
        ["V", "NP"],
        ["V"],
        ["VP", "PP"],
        ["VP", "Conj", "VP"],
        ["V", "Clause"]
    ],

    "PP": [
        ["P", "NP"]
    ],

    "Adj": [
        ["available"],
        ["pending"],
        ["current"],
        ["new"],
        ["old"],
    ],

    "Det": [
        ["the"],
        ["a"],
        ["an"],
        ["my"],
        ["your"],
        ["this"],
        ["that"],
    ],

    "Aux": [
        ["do"],
        ["does"],
        ["can"],
        ["could"],
        ["will"],
        ["would"],
        ["is"],
        ["are"],
    ],

    "Wh": [
        ["what"],
        ["which"],
        ["where"],
        ["when"],
        ["who"],
        ["how"],
    ],

    "Rel": [
        ["that"],
        ["which"],
        ["who"],
    ],

    "Pronoun": [
        ["i"],
        ["you"],
        ["we"],
        ["they"],
        ["he"],
        ["she"],
        ["it"],
    ],

    "Conj": [
        ["and"],
        ["or"],
        ["but"],
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
        ["department"],
        ["faculty"],
        ["teacher"],
        ["class"],
        ["fee"],
        ["fees"],
        ["scholarship"],
        ["registration"],
        ["library"],
        ["card"],
        ["mark"],
        ["marks"],
        ["semester"],
        ["subject"],
        ["attendance"],
        ["result"],
        ["grade"],
        ["application"],
        ["certificate"],
        ["project"],
        ["lab"],
        ["room"],
    ],

    "V": [
        ["need"],
        ["needs"],
        ["check"],
        ["checks"],
        ["want"],
        ["wants"],
        ["get"],
        ["gets"],
        ["provide"],
        ["provides"],
        ["show"],
        ["shows"],
        ["know"],
        ["knows"],
        ["tell"],
        ["tells"],
        ["submit"],
        ["submits"],
        ["download"],
        ["downloads"],
        ["verify"],
        ["verifies"],
        ["upload"],
        ["uploads"],
    ],
    
}


# ============================================================
# TEST MESSAGES
# ============================================================

MESSAGES = [

    # Normal conversational
    "I need the timetable",
    "I need the exam schedule",
    "can you show the result",
    "where is the assignment",
    "how can I get the certificate",

    # Missing punctuation
    "can you tell me the exam date",
    "where can I get the timetable",
    "please show my result",

    # Informal / incomplete
    "exam timetable please",
    "hostel room needed",
    "need assignment deadline",
    "result please",
    "certificate needed",

    # Coordination
    "I need the timetable and the exam schedule",
    "I need the result and the certificate",
    "can you show the timetable or the schedule",

    # Embedded / multi-clause
    "I want to know when the exam starts",
    "can you tell me where the department is",
    "I need to know whether the result is available",

    # Relative clause
    "I need the certificate that the department provides",
    "show the timetable which the department published",

    # PP attachment
    "I need the timetable for the exam",
    "I need the document for the course",

    # Missing article
    "student needs timetable",
    "student checks result",

    # Ungrammatical / adversarial
    "the student need the timetable",
    "the timetable student need",
    "need the the timetable",
    "where timetable exam",
    "can you the result show",

    # Long conversational messages
    "I am a student and I need the timetable for the exam",
    "I need the result and I also want the certificate",
    "can you tell me where I can download the assignment",
    "I want the timetable that the department published",
]


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(sentence):

    sentence = sentence.lower()

    sentence = re.sub(
        r"[^\w\s]",
        "",
        sentence
    )

    return sentence.split()


# ============================================================
# SIMPLE CFG PARSER
# ============================================================

class RobustParser:

    def __init__(self, grammar):

        self.grammar = grammar
        self.nonterminals = set(
            grammar.keys()
        )

    def parse(self, words):

        n = len(words)

        memo = {}
        active = set()

        def expand(
            symbol,
            position
        ):

            key = (
                symbol,
                position
            )

            if key in memo:
                return memo[key]

            if key in active:
                return []

            active.add(key)

            results = []

            # ------------------------------------------------
            # Terminal
            # ------------------------------------------------

            if symbol not in self.nonterminals:

                if (
                    position < n
                    and words[position] == symbol
                ):

                    results.append(
                        position + 1
                    )

                active.remove(key)

                memo[key] = results

                return results

            # ------------------------------------------------
            # Non-terminal
            # ------------------------------------------------

            for rule in self.grammar[symbol]:

                positions = [position]

                for child in rule:

                    next_positions = []

                    for current in positions:

                        child_results = expand(
                            child,
                            current
                        )

                        next_positions.extend(
                            child_results
                        )

                    positions = next_positions

                    if not positions:
                        break

                results.extend(
                    positions
                )

            active.remove(key)

            # Remove duplicate positions
            results = list(
                dict.fromkeys(results)
            )

            memo[key] = results

            return results

        final_positions = expand(
            "S",
            0
        )

        return n in final_positions


# ============================================================
# HEURISTIC CONVERSATIONAL ANALYZER
# ============================================================

def conversational_features(sentence):

    words = tokenize(sentence)

    features = {
        "word_count": len(words),
        "question": (
            sentence.strip().lower().startswith(
                (
                    "can ",
                    "could ",
                    "where ",
                    "what ",
                    "when ",
                    "which ",
                    "who ",
                    "how ",
                    "do ",
                    "does ",
                )
            )
        ),
        "coordination": any(
            word in words
            for word in [
                "and",
                "or",
                "but"
            ]
        ),
        "relative_clause": any(
            word in words
            for word in [
                "that",
                "which",
                "who"
            ]
        ),
        "missing_article": (
            len(words) > 0
            and words[0] in [
                "student",
                "result",
                "timetable",
                "certificate",
                "assignment",
                "hostel",
            ]
        ),
        "incomplete": (
            len(words) <= 4
        ),
        "embedded_clause": (
            "when" in words
            or "where" in words
            or "whether" in words
        ),
        "multi_clause": (
            "and" in words
            or "but" in words
        ),
    }

    return features


# ============================================================
# COVERAGE TEST
# ============================================================

def evaluate_coverage():

    print("\n" + "=" * 80)
    print("TASK 12 - CONVERSATIONAL ROBUSTNESS TEST")
    print("=" * 80)

    original_parser = RobustParser(
        ORIGINAL_GRAMMAR
    )

    adapted_parser = RobustParser(
        ADAPTED_GRAMMAR
    )

    results = []

    original_accepted = 0
    adapted_accepted = 0

    print(
        f"\nTotal conversational messages: "
        f"{len(MESSAGES)}"
    )

    print("\n" + "-" * 80)

    for index, message in enumerate(
        MESSAGES,
        start=1
    ):

        words = tokenize(message)

        original = original_parser.parse(
            words
        )

        adapted = adapted_parser.parse(
            words
        )

        features = conversational_features(
            message
        )

        if original:
            original_accepted += 1

        if adapted:
            adapted_accepted += 1

        results.append(
            {
                "id": index,
                "message": message,
                "original": original,
                "adapted": adapted,
                "features": features,
            }
        )

        print(
            f"{index:02d}. "
            f"{message:<60}"
        )

        print(
            f"    Original: "
            f"{'ACCEPT' if original else 'REJECT':<8}"
            f" | Adapted: "
            f"{'ACCEPT' if adapted else 'REJECT'}"
        )

    original_coverage = (
        original_accepted
        / len(MESSAGES)
        * 100
    )

    adapted_coverage = (
        adapted_accepted
        / len(MESSAGES)
        * 100
    )

    print("\n" + "=" * 80)
    print("COVERAGE RESULTS")
    print("=" * 80)

    print(
        f"\nOriginal grammar accepted : "
        f"{original_accepted}/{len(MESSAGES)} "
        f"({original_coverage:.2f}%)"
    )

    print(
        f"Original grammar uncovered: "
        f"{len(MESSAGES) - original_accepted}/"
        f"{len(MESSAGES)} "
        f"({100 - original_coverage:.2f}%)"
    )

    print(
        f"\nAdapted grammar accepted  : "
        f"{adapted_accepted}/{len(MESSAGES)} "
        f"({adapted_coverage:.2f}%)"
    )

    print(
        f"Adapted grammar uncovered : "
        f"{len(MESSAGES) - adapted_accepted}/"
        f"{len(MESSAGES)} "
        f"({100 - adapted_coverage:.2f}%)"
    )

    return results


# ============================================================
# CATEGORY ANALYSIS
# ============================================================

def category_analysis(results):

    print("\n" + "=" * 80)
    print("CATEGORY-BASED ANALYSIS")
    print("=" * 80)

    categories = {
        "Questions": [],
        "Coordination": [],
        "Relative Clauses": [],
        "Embedded Clauses": [],
        "Incomplete / Short": [],
        "Missing Articles": [],
        "Multi-Clause": [],
    }

    for result in results:

        features = result["features"]

        if features["question"]:
            categories[
                "Questions"
            ].append(result)

        if features["coordination"]:
            categories[
                "Coordination"
            ].append(result)

        if features["relative_clause"]:
            categories[
                "Relative Clauses"
            ].append(result)

        if features["embedded_clause"]:
            categories[
                "Embedded Clauses"
            ].append(result)

        if features["incomplete"]:
            categories[
                "Incomplete / Short"
            ].append(result)

        if features["missing_article"]:
            categories[
                "Missing Articles"
            ].append(result)

        if features["multi_clause"]:
            categories[
                "Multi-Clause"
            ].append(result)

    for category, items in categories.items():

        if not items:
            continue

        original = sum(
            item["original"]
            for item in items
        )

        adapted = sum(
            item["adapted"]
            for item in items
        )

        print(
            f"\n{category}"
        )

        print(
            f"  Messages: {len(items)}"
        )

        print(
            f"  Original: "
            f"{original}/{len(items)}"
        )

        print(
            f"  Adapted : "
            f"{adapted}/{len(items)}"
        )


# ============================================================
# MISSING CONSTRUCTIONS
# ============================================================

def missing_construction_analysis():

    print("\n" + "=" * 80)
    print("MISSING CONSTRUCTION ANALYSIS")
    print("=" * 80)

    missing = [

        (
            "Informal fragments",
            "Messages such as 'result please' "
            "do not contain a complete S -> NP VP structure."
        ),

        (
            "Auxiliary questions",
            "Questions such as 'can you show the result' "
            "require auxiliary and inversion rules."
        ),

        (
            "Wh questions",
            "Questions beginning with what, where, when "
            "and how require dedicated question structures."
        ),

        (
            "Coordination",
            "Messages containing and, or or but "
            "require coordination rules."
        ),

        (
            "Relative clauses",
            "Relative clauses require rules connecting "
            "a noun phrase with an embedded clause."
        ),

        (
            "Embedded clauses",
            "Requests such as 'tell me where...' "
            "contain clauses embedded inside another clause."
        ),

        (
            "Missing punctuation",
            "Conversational messages often omit punctuation, "
            "so token-level parsing should not depend on punctuation."
        ),

        (
            "Spelling variation",
            "Real users may type abbreviations, spelling errors "
            "and informal expressions."
        ),
    ]

    for name, explanation in missing:

        print(
            f"\n{name}:"
        )

        print(
            f"  {explanation}"
        )


# ============================================================
# IMPACT ANALYSIS
# ============================================================

def impact_analysis():

    print("\n" + "=" * 80)
    print("IMPACT OF MISSING CONSTRUCTIONS")
    print("=" * 80)

    print("""
The original grammar was designed mainly for simple
declarative university-helpdesk sentences.

Therefore, conversational input can expose several
coverage problems.

1. User experience
   A valid user request may be rejected even when
   its intended meaning is clear.

2. Reduced recall
   Missing constructions reduce the number of
   real-world queries that can be parsed.

3. Ambiguity
   Informal messages can have multiple possible
   interpretations.

4. Multi-clause requests
   A single user message may contain several
   requests or embedded clauses.

5. Missing punctuation
   Users frequently omit question marks and commas
   in chat interfaces.

6. Robustness
   A practical university helpdesk needs to tolerate
   incomplete and conversational language.
""")


# ============================================================
# REDESIGN STRATEGY
# ============================================================

def redesign_strategy():

    print("\n" + "=" * 80)
    print("ADAPTED GRAMMAR / REDESIGN STRATEGY")
    print("=" * 80)

    print("""
The original CFG can be redesigned using the following
layers.

Layer 1 - Normalization
----------------------
Convert text to lowercase, remove unnecessary punctuation
and tokenize the user message.

Layer 2 - Conversational constructions
--------------------------------------
Add rules for:

    • yes/no questions
    • wh questions
    • imperatives
    • short fragments
    • coordination
    • embedded clauses
    • relative clauses

Layer 3 - Feature constraints
-----------------------------
Use number, person, case and subcategorization features
to prevent linguistically invalid parses.

Layer 4 - Earley parsing
------------------------
Use Earley parsing for the resulting CFG because it
supports recursive and ambiguous structures.

Layer 5 - Probabilistic ranking
------------------------------
Use PCFG/Viterbi ranking to choose the most likely
interpretation when several parses are possible.

Layer 6 - Semantic/helpdesk layer
---------------------------------
Map the parsed request to intents such as:

    • timetable
    • examination
    • result
    • certificate
    • assignment
    • hostel
    • fees
    • scholarship
    • registration
    • library

This produces a more practical university-helpdesk NLP
pipeline.
""")


# ============================================================
# EVIDENCE-BASED DEFENCE
# ============================================================

def evidence_based_defence(results):

    print("\n" + "=" * 80)
    print("EVIDENCE-BASED DEFENCE")
    print("=" * 80)

    original = sum(
        result["original"]
        for result in results
    )

    adapted = sum(
        result["adapted"]
        for result in results
    )

    total = len(results)

    original_rate = (
        original / total * 100
    )

    adapted_rate = (
        adapted / total * 100
    )

    improvement = (
        adapted_rate
        - original_rate
    )

    print(
        f"""
Test-set evidence:

Total conversational messages : {total}

Original grammar coverage     : {original_rate:.2f}%

Adapted grammar coverage      : {adapted_rate:.2f}%

Coverage improvement          : {improvement:.2f} percentage points
"""
    )

    print("""
The evidence demonstrates that a grammar designed only
for simple declarative sentences is insufficient for a
conversational university-helpdesk environment.

The adapted design improves coverage by adding question,
coordination, relative-clause, embedded-clause and
conversational structures.

However, a grammar alone should not be expected to handle
all user language. A practical system should combine
normalization, feature constraints, syntactic parsing,
probabilistic ranking and semantic intent detection.
""")


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(results):

    output_path = Path(
        "results/task12_robustness_report.txt"
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
            "DSA03 NLP ASSIGNMENT - TASK 12\n"
        )

        file.write(
            "ROBUSTNESS TO CONVERSATIONAL AND "
            "ADVERSARIAL INPUT\n\n"
        )

        total = len(results)

        original = sum(
            result["original"]
            for result in results
        )

        adapted = sum(
            result["adapted"]
            for result in results
        )

        file.write(
            f"Total messages: {total}\n"
        )

        file.write(
            f"Original accepted: {original}\n"
        )

        file.write(
            f"Original coverage: "
            f"{original / total * 100:.2f}%\n"
        )

        file.write(
            f"Adapted accepted: {adapted}\n"
        )

        file.write(
            f"Adapted coverage: "
            f"{adapted / total * 100:.2f}%\n\n"
        )

        file.write(
            "MESSAGE RESULTS\n"
        )

        file.write(
            "===============\n\n"
        )

        for result in results:

            file.write(
                f"{result['id']}. "
                f"{result['message']}\n"
            )

            file.write(
                f"   Original: "
                f"{'ACCEPT' if result['original'] else 'REJECT'}\n"
            )

            file.write(
                f"   Adapted : "
                f"{'ACCEPT' if result['adapted'] else 'REJECT'}\n\n"
            )

        file.write(
            "CONCLUSION\n"
        )

        file.write(
            "==========\n\n"
        )

        file.write(
            "The original CFG does not adequately cover "
            "conversational university-helpdesk queries. "
            "The adapted grammar provides broader coverage "
            "by supporting questions, coordination, embedded "
            "clauses, relative clauses and conversational "
            "forms. For a practical system, Earley parsing "
            "with feature structures and probabilistic "
            "ranking is recommended.\n"
        )

    print(
        f"\nTask 12 report saved to:\n"
        f"{output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("DSA03 NLP ASSIGNMENT - TASK 12")
    print("ROBUSTNESS TO CONVERSATIONAL AND ADVERSARIAL INPUT")
    print("=" * 80)

    # --------------------------------------------------------
    # Evaluate original and adapted grammars
    # --------------------------------------------------------

    results = evaluate_coverage()

    # --------------------------------------------------------
    # Category analysis
    # --------------------------------------------------------

    category_analysis(
        results
    )

    # --------------------------------------------------------
    # Missing constructions
    # --------------------------------------------------------

    missing_construction_analysis()

    # --------------------------------------------------------
    # Impact
    # --------------------------------------------------------

    impact_analysis()

    # --------------------------------------------------------
    # Redesign
    # --------------------------------------------------------

    redesign_strategy()

    # --------------------------------------------------------
    # Evidence
    # --------------------------------------------------------

    evidence_based_defence(
        results
    )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    save_results(
        results
    )

    print("\n" + "=" * 80)
    print(
        "TASK 12 IMPLEMENTATION COMPLETED"
    )
    print("=" * 80)


# ============================================================
# PROGRAM ENTRY
# ============================================================

if __name__ == "__main__":
    main()