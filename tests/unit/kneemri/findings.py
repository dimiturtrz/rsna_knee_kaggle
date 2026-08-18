from kneemri.findings import TARGETS


def test_targets():
    """The twelve findings, in the ONE order the submission columns and every member's head share.

    Asserted whole rather than by length, because the failure this guards is a REORDER: a name moved
    within the tuple leaves every count, every shape and every gate agreeing with itself, and turns a
    member fitted under the old order into one that is wrong on every finding when it is served.
    """
    assert TARGETS == (
        "ACL",
        "MCL",
        "Medial Meniscus",
        "Lateral Meniscus",
        "Medial OA",
        "Lateral OA",
        "PF OA",
        "Effusion",
        "Synovitis",
        "Baker's",
        "Contusion",
        "Fracture",
    )
    assert len(set(TARGETS)) == len(TARGETS), "a duplicated name would make two columns one"
