# ============================================================
# DSA03 NLP ASSIGNMENT 3
# Task 4 - Agreement
# ============================================================

# ------------------------------------------------------------
# AGREEMENT-AWARE CFG
# ------------------------------------------------------------

agreement_rules = [

    # Subject-verb agreement
    "S -> NP_SG VP_SG",
    "S -> NP_PL VP_PL",

    # Determiner-noun agreement
    "NP_SG -> DET_SG N_SG",
    "NP_PL -> DET_PL N_PL",

    # Singular / plural noun phrases
    "NP_SG -> PRO_SG",
    "NP_PL -> PRO_PL",

    # Verb phrases
    "VP_SG -> V_SG",
    "VP_SG -> V_SG NP_SG",
    "VP_SG -> V_SG NP_PL",

    "VP_PL -> V_PL",
    "VP_PL -> V_PL NP_SG",
    "VP_PL -> V_PL NP_PL",

    # Pronoun case
    "NP_NOM -> PRO_NOM",
    "NP_ACC -> PRO_ACC",
]


# ------------------------------------------------------------
# FEATURE-BASED REPRESENTATION
# ------------------------------------------------------------

features = {
    "student": {
        "CAT": "N",
        "NUM": "SG"
    },

    "students": {
        "CAT": "N",
        "NUM": "PL"
    },

    "the": {
        "CAT": "DET",
        "NUM": "ANY"
    },

    "a": {
        "CAT": "DET",
        "NUM": "SG"
    },

    "these": {
        "CAT": "DET",
        "NUM": "PL"
    },

    "this": {
        "CAT": "DET",
        "NUM": "SG"
    },

    "needs": {
        "CAT": "V",
        "NUM": "SG"
    },

    "need": {
        "CAT": "V",
        "NUM": "PL"
    },

    "does": {
        "CAT": "AUX",
        "NUM": "SG"
    },

    "do": {
        "CAT": "AUX",
        "NUM": "PL"
    },

    "he": {
        "CAT": "PRON",
        "NUM": "SG",
        "PER": "3",
        "CASE": "NOM"
    },

    "she": {
        "CAT": "PRON",
        "NUM": "SG",
        "PER": "3",
        "CASE": "NOM"
    },

    "they": {
        "CAT": "PRON",
        "NUM": "PL",
        "PER": "3",
        "CASE": "NOM"
    },

    "we": {
        "CAT": "PRON",
        "NUM": "PL",
        "PER": "1",
        "CASE": "NOM"
    },

    "him": {
        "CAT": "PRON",
        "NUM": "SG",
        "PER": "3",
        "CASE": "ACC"
    },

    "her": {
        "CAT": "PRON",
        "NUM": "SG",
        "PER": "3",
        "CASE": "ACC"
    },

    "them": {
        "CAT": "PRON",
        "NUM": "PL",
        "PER": "3",
        "CASE": "ACC"
    }
}


# ------------------------------------------------------------
# UNIFICATION
# ------------------------------------------------------------

def unify(feature1, feature2):
    """
    Unify two feature structures.

    If the same feature has conflicting values,
    unification fails.
    """

    result = dict(feature1)

    for key, value in feature2.items():

        if key in result:

            if result[key] != value:
                return None

        else:
            result[key] = value

    return result


# ------------------------------------------------------------
# AGREEMENT CHECK
# ------------------------------------------------------------

def check_subject_verb(subject, verb):

    if subject not in features or verb not in features:
        return False

    subject_features = features[subject]
    verb_features = features[verb]

    if subject_features.get("CAT") == "PRON":
        subject_num = subject_features.get("NUM")
    elif subject_features.get("CAT") == "N":
        subject_num = subject_features.get("NUM")
    else:
        return False

    verb_num = verb_features.get("NUM")

    return subject_num == verb_num


# ------------------------------------------------------------
# DETERMINER-NOUN AGREEMENT
# ------------------------------------------------------------

def check_determiner_noun(det, noun):

    if det not in features or noun not in features:
        return False

    det_num = features[det].get("NUM")
    noun_num = features[noun].get("NUM")

    if det_num == "ANY":
        return True

    return det_num == noun_num


# ------------------------------------------------------------
# PRONOUN CASE
# ------------------------------------------------------------

def check_pronoun_case(pronoun, required_case):

    if pronoun not in features:
        return False

    return features[pronoun].get("CASE") == required_case


# ------------------------------------------------------------
# TEST SENTENCES
# ------------------------------------------------------------

grammatical = [
    "The student needs the timetable.",
    "The students need the timetable.",
    "This student needs the timetable.",
    "These students need the timetable.",
    "He needs the assignment.",
    "They need the assignment."
]


ungrammatical = [
    "The student need the timetable.",
    "The students needs the timetable.",
    "These student need the timetable.",
    "This students needs the timetable.",
    "He need the assignment.",
    "They needs the assignment."
]


# ------------------------------------------------------------
# SIMPLE SENTENCE VALIDATION
# ------------------------------------------------------------

def validate_sentence(sentence):

    words = sentence.lower().replace(".", "").split()

    if len(words) < 4:
        return False

    det = words[0]
    noun = words[1]
    verb = words[2]

    # Determiner-noun agreement
    det_ok = check_determiner_noun(det, noun)

    # Subject-verb agreement
    subject_ok = check_subject_verb(noun, verb)

    return det_ok and subject_ok


# ------------------------------------------------------------
# DISPLAY TEST RESULTS
# ------------------------------------------------------------

def test_sentences(sentences, expected):

    print("\n" + "-" * 75)
    print(f"EXPECTED: {expected}")
    print("-" * 75)

    correct = 0

    for sentence in sentences:

        result = validate_sentence(sentence)

        predicted = "Grammatical" if result else "Ungrammatical"

        status = "PASS" if (
            (expected == "Grammatical" and result)
            or
            (expected == "Ungrammatical" and not result)
        ) else "FAIL"

        if status == "PASS":
            correct += 1

        print(
            f"{sentence:<45} "
            f"{predicted:<15} {status}"
        )

    print("-" * 75)
    print(f"Correct: {correct}/{len(sentences)}")


# ------------------------------------------------------------
# UNIFICATION TESTS
# ------------------------------------------------------------

def unification_tests():

    print("\n" + "=" * 75)
    print("UNIFICATION TESTS")
    print("=" * 75)

    tests = [

        (
            {"NUM": "SG"},
            {"NUM": "SG"}
        ),

        (
            {"NUM": "PL", "PER": "3"},
            {"NUM": "PL"}
        ),

        (
            {"NUM": "SG"},
            {"NUM": "PL"}
        )
    ]

    for i, (f1, f2) in enumerate(tests, 1):

        result = unify(f1, f2)

        print(f"\nTest {i}")
        print(f"Feature 1: {f1}")
        print(f"Feature 2: {f2}")
        print(f"Result   : {result}")

        if result is None:
            print("Status   : FAIL - Feature conflict")
        else:
            print("Status   : SUCCESS")


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    print("=" * 75)
    print("TASK 4 - AGREEMENT")
    print("=" * 75)

    # --------------------------------------------------------
    # RULE COUNT
    # --------------------------------------------------------

    print("\nAGREEMENT-AWARE CFG")
    print("-" * 75)

    for rule in agreement_rules:
        print(rule)

    print(f"\nAgreement-aware rules: {len(agreement_rules)}")

    # --------------------------------------------------------
    # GRAMMATICAL SENTENCES
    # --------------------------------------------------------

    test_sentences(
        grammatical,
        "Grammatical"
    )

    # --------------------------------------------------------
    # UNGRAMMATICAL SENTENCES
    # --------------------------------------------------------

    test_sentences(
        ungrammatical,
        "Ungrammatical"
    )

    # --------------------------------------------------------
    # UNIFICATION
    # --------------------------------------------------------

    unification_tests()

    # --------------------------------------------------------
    # PRONOUN CASE
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("PRONOUN CASE TESTS")
    print("=" * 75)

    case_tests = [
        ("he", "NOM"),
        ("they", "NOM"),
        ("him", "ACC"),
        ("them", "ACC"),
        ("him", "NOM")
    ]

    for pronoun, required_case in case_tests:

        result = check_pronoun_case(
            pronoun,
            required_case
        )

        print(
            f"{pronoun:<8} "
            f"required={required_case:<4} "
            f"result={'PASS' if result else 'FAIL'}"
        )

    # --------------------------------------------------------
    # FEATURE-BASED DISCUSSION
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("FEATURE-BASED APPROACH")
    print("=" * 75)

    print("""
The agreement-aware CFG uses separate nonterminals such as
NP_SG, NP_PL, VP_SG and VP_PL.

This enforces agreement directly through grammar rules.

However, multiplying nonterminals becomes expensive when more
features are added, such as:

    Number
    Person
    Case
    Gender

A feature-based grammar stores these properties as attributes
inside feature structures.

For example:

    [CAT: NP
     NUM: SG
     PER: 3]

Feature unification checks whether two structures are compatible.

Therefore, feature structures reduce rule duplication and make
the grammar easier to maintain.
""")

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print("=" * 75)
    print("TASK 4 IMPLEMENTATION COMPLETED")
    print("=" * 75)


if __name__ == "__main__":
    main()