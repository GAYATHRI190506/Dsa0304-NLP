# ============================================================
# DSA03 NLP ASSIGNMENT 3
# Task 5 - Verb Subcategorization
# ============================================================

# ------------------------------------------------------------
# SUBCATEGORIZATION FRAMES
# ------------------------------------------------------------

frames = {
    "INTRANSITIVE": {
        "description": "Verb requires no object.",
        "pattern": ["V"],
        "examples": [
            "crash",
            "arrive",
            "sleep"
        ]
    },

    "TRANSITIVE": {
        "description": "Verb requires one noun phrase object.",
        "pattern": ["V", "NP"],
        "examples": [
            "check",
            "submit",
            "download",
            "view"
        ]
    },

    "DITRANSITIVE": {
        "description": "Verb takes two complements, typically NP + NP.",
        "pattern": ["V", "NP", "NP"],
        "examples": [
            "give",
            "send",
            "provide"
        ]
    },

    "SENTENTIAL_COMPLEMENT": {
        "description": "Verb takes an embedded sentence.",
        "pattern": ["V", "S"],
        "examples": [
            "know",
            "explain",
            "confirm"
        ]
    },

    "PP_COMPLEMENT": {
        "description": "Verb requires a prepositional phrase.",
        "pattern": ["V", "PP"],
        "examples": [
            "apply",
            "depend",
            "belong"
        ]
    },

    "NP_PP": {
        "description": "Verb takes an NP followed by a PP.",
        "pattern": ["V", "NP", "PP"],
        "examples": [
            "provide",
            "send",
            "give"
        ]
    }
}


# ------------------------------------------------------------
# VERB LEXICON
# ------------------------------------------------------------

verb_frames = {

    "check": ["TRANSITIVE"],
    "submit": ["TRANSITIVE"],
    "download": ["TRANSITIVE"],
    "view": ["TRANSITIVE"],

    "give": ["DITRANSITIVE", "NP_PP"],
    "send": ["DITRANSITIVE", "NP_PP"],
    "provide": ["DITRANSITIVE", "NP_PP"],

    "know": ["SENTENTIAL_COMPLEMENT", "TRANSITIVE"],
    "explain": ["SENTENTIAL_COMPLEMENT", "TRANSITIVE"],
    "confirm": ["SENTENTIAL_COMPLEMENT", "TRANSITIVE"],

    "apply": ["PP_COMPLEMENT"],
    "depend": ["PP_COMPLEMENT"],
    "belong": ["PP_COMPLEMENT"],

    "crash": ["INTRANSITIVE"],
    "arrive": ["INTRANSITIVE"],
    "sleep": ["INTRANSITIVE"]
}


# ------------------------------------------------------------
# COMPLEMENT VS ADJUNCT
# ------------------------------------------------------------

complement_examples = [
    ("submit", "the assignment"),
    ("check", "the timetable"),
    ("apply", "for the hostel"),
    ("provide", "placement information")
]

adjunct_examples = [
    ("check", "the timetable quickly"),
    ("submit", "the assignment today"),
    ("view", "the result online"),
    ("download", "the form yesterday")
]


# ------------------------------------------------------------
# VALIDATE FRAME
# ------------------------------------------------------------

def validate_frame(verb, frame_name):

    if verb not in verb_frames:
        return False

    return frame_name in verb_frames[verb]


# ------------------------------------------------------------
# SENTENCE VALIDATION
# ------------------------------------------------------------

sentence_tests = [

    # Correct
    ("The student checks the timetable.", True, "TRANSITIVE"),
    ("The student submits the assignment.", True, "TRANSITIVE"),
    ("The university provides the student information.", True, "DITRANSITIVE"),
    ("The student knows that the exam starts.", True, "SENTENTIAL_COMPLEMENT"),
    ("The student applies for the hostel.", True, "PP_COMPLEMENT"),
    ("The portal crashed.", True, "INTRANSITIVE"),

    # Incorrect / under-complemented
    ("Submit.", False, "TRANSITIVE"),
    ("Check.", False, "TRANSITIVE"),
    ("Provide.", False, "DITRANSITIVE"),

    # Incorrect / over-complemented
    ("The portal crashed the form.", False, "INTRANSITIVE"),
    ("The student checks.", False, "TRANSITIVE"),
    ("The student applies the hostel.", False, "PP_COMPLEMENT")
]


# ------------------------------------------------------------
# DISPLAY FRAMES
# ------------------------------------------------------------

def display_frames():

    print("\n" + "=" * 75)
    print("VERB SUBCATEGORIZATION FRAMES")
    print("=" * 75)

    for name, information in frames.items():

        print(f"\n{name}")
        print("-" * 75)

        print(f"Description : {information['description']}")
        print(f"Pattern     : {' '.join(information['pattern'])}")
        print(
            f"Examples    : "
            f"{', '.join(information['examples'])}"
        )


# ------------------------------------------------------------
# MULTIPLE FRAME VERBS
# ------------------------------------------------------------

def display_multiple_frames():

    print("\n" + "=" * 75)
    print("VERBS WITH MULTIPLE FRAMES")
    print("=" * 75)

    for verb, verb_frame_list in verb_frames.items():

        if len(verb_frame_list) > 1:

            print(
                f"{verb:<10} -> "
                f"{', '.join(verb_frame_list)}"
            )


# ------------------------------------------------------------
# COMPLEMENT / ADJUNCT TEST
# ------------------------------------------------------------

def display_complement_adjunct():

    print("\n" + "=" * 75)
    print("COMPLEMENT VS ADJUNCT")
    print("=" * 75)

    print("\nComplements:")
    for verb, phrase in complement_examples:
        print(f"  {verb:<10} + {phrase}")

    print("\nAdjuncts:")
    for verb, phrase in adjunct_examples:
        print(f"  {verb:<10} + {phrase}")

    print("""
Complement tests:
1. A complement is selected by the verb.
2. Removing a required complement may make the sentence
   incomplete or ungrammatical.
3. The type of complement is constrained by the verb.

Adjunct tests:
1. Adjuncts provide optional information.
2. They can usually be removed without making the sentence
   structurally incomplete.
3. They are generally more freely repeatable and mobile.
""")


# ------------------------------------------------------------
# TEST SENTENCES
# ------------------------------------------------------------

def run_sentence_tests():

    print("\n" + "=" * 75)
    print("SUBCATEGORIZATION TEST SENTENCES")
    print("=" * 75)

    correct = 0

    for sentence, expected, frame in sentence_tests:

        # Simple heuristic for demonstration.
        words = sentence.lower().replace(".", "").split()

        if frame == "TRANSITIVE":
            predicted = len(words) >= 4

        elif frame == "DITRANSITIVE":
            predicted = len(words) >= 5

        elif frame == "SENTENTIAL_COMPLEMENT":
            predicted = "that" in words

        elif frame == "PP_COMPLEMENT":
            predicted = "for" in words

        elif frame == "INTRANSITIVE":
            predicted = len(words) <= 3

        else:
            predicted = False

        status = "PASS" if predicted == expected else "FAIL"

        if status == "PASS":
            correct += 1

        print(
            f"{sentence:<48} "
            f"Expected={str(expected):<5} "
            f"Predicted={str(predicted):<5} "
            f"{status}"
        )

    print("-" * 75)
    print(f"Correct: {correct}/{len(sentence_tests)}")


# ------------------------------------------------------------
# PARSING COST DISCUSSION
# ------------------------------------------------------------

def parsing_cost():

    print("\n" + "=" * 75)
    print("PARSING COST AND AMBIGUITY")
    print("=" * 75)

    print("""
A verb may license more than one subcategorization frame.

For example:

    provide + NP + NP
    provide + NP + PP

Therefore, when the parser encounters 'provide', it may need
to consider multiple possible structures.

This increases the number of candidate parses.

A subcategorization lexicon reduces this ambiguity because
the parser can reject structures that do not satisfy the
selected verb's valence requirements.

This improves grammaticality checking and reduces unnecessary
parse trees.
""")


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    print("=" * 75)
    print("TASK 5 - VERB SUBCATEGORIZATION")
    print("=" * 75)

    display_frames()

    display_multiple_frames()

    display_complement_adjunct()

    run_sentence_tests()

    parsing_cost()

    print("\n" + "=" * 75)
    print("TASK 5 IMPLEMENTATION COMPLETED")
    print("=" * 75)


if __name__ == "__main__":
    main()