
"""Phase 11.4G-4 proposal validator tests."""

from src.recovery.decomposition_proposal_validator import (
    DecompositionProposalValidator,
)


VALIDATOR = DecompositionProposalValidator()

QUESTION = (
    "Which university did the author of "
    "The Silver Atlas attend?"
)


def valid_proposal():
    return {
        "relationships": [
            {
                "subject": "The Silver Atlas",
                "predicate": "authored_by",
                "object": "?author",
            },
            {
                "subject": "?author",
                "predicate": "attended",
                "object": "?university",
            },
        ],
        "answer_variable": "?university",
    }


def test_valid_two_hop():
    result = VALIDATOR.validate(QUESTION, valid_proposal())
    assert result.is_valid
    assert len(result.requirements.relationships) == 2


def test_correct_answer_variable():
    result = VALIDATOR.validate(QUESTION, valid_proposal())
    assert result.requirements.answer_variable == "?university"


def test_single_hop():
    proposal = {
        "relationships": [
            {
                "subject": "The Silver Atlas",
                "predicate": "authored_by",
                "object": "?author",
            }
        ],
        "answer_variable": "?author",
    }
    assert VALIDATOR.validate(QUESTION, proposal).is_valid


def test_unknown_predicate():
    proposal = valid_proposal()
    proposal["relationships"][1]["predicate"] = "graduated_from"
    assert not VALIDATOR.validate(QUESTION, proposal).is_valid


def test_disconnected_relationships():
    proposal = valid_proposal()
    proposal["relationships"][1]["subject"] = "Daniel Reed"
    assert not VALIDATOR.validate(QUESTION, proposal).is_valid


def test_missing_answer_variable():
    proposal = valid_proposal()
    proposal["answer_variable"] = "?city"
    assert not VALIDATOR.validate(QUESTION, proposal).is_valid


def test_malformed_variable():
    proposal = valid_proposal()
    proposal["answer_variable"] = "?University Name"
    assert not VALIDATOR.validate(QUESTION, proposal).is_valid


def test_extra_field():
    proposal = valid_proposal()
    proposal["confidence"] = 0.99
    assert not VALIDATOR.validate(QUESTION, proposal).is_valid


def test_missing_field():
    proposal = valid_proposal()
    del proposal["answer_variable"]
    assert not VALIDATOR.validate(QUESTION, proposal).is_valid


def test_empty_relationships():
    proposal = valid_proposal()
    proposal["relationships"] = []
    assert not VALIDATOR.validate(QUESTION, proposal).is_valid


def test_too_many_relationships():
    proposal = valid_proposal()
    proposal["relationships"] *= 3
    assert not VALIDATOR.validate(QUESTION, proposal).is_valid


def test_invalid_json_shape():
    assert not VALIDATOR.validate(QUESTION, []).is_valid


def test_invalid_relationship_type():
    proposal = valid_proposal()
    proposal["relationships"][0] = "not an object"
    assert not VALIDATOR.validate(QUESTION, proposal).is_valid


def test_invalid_question():
    try:
        VALIDATOR.validate("", valid_proposal())
    except ValueError:
        return
    raise AssertionError("Expected ValueError.")


def test_structurally_valid_but_semantically_wrong():
    proposal = valid_proposal()
    proposal["relationships"][0]["subject"] = "Another Book"

    # A wrong book title is not detectable through structural
    # validation alone. This limitation must be measured.
    result = VALIDATOR.validate(QUESTION, proposal)
    assert result.is_valid


def test_invalid_variable_in_relationship():
    proposal = valid_proposal()
    proposal["relationships"][0]["object"] = "?Author Name"
    assert not VALIDATOR.validate(QUESTION, proposal).is_valid


def test_unconnected_third_relationship():
    proposal = valid_proposal()
    proposal["relationships"].append(
        {
            "subject": "Unrelated Book",
            "predicate": "authored_by",
            "object": "?other_author",
        }
    )
    assert not VALIDATOR.validate(QUESTION, proposal).is_valid


def main():
    tests = [
        test_valid_two_hop,
        test_correct_answer_variable,
        test_single_hop,
        test_unknown_predicate,
        test_disconnected_relationships,
        test_missing_answer_variable,
        test_malformed_variable,
        test_extra_field,
        test_missing_field,
        test_empty_relationships,
        test_too_many_relationships,
        test_invalid_json_shape,
        test_invalid_relationship_type,
        test_invalid_question,
        test_structurally_valid_but_semantically_wrong,
        test_invalid_variable_in_relationship,
        test_unconnected_third_relationship,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print(
        "\nPHASE 11.4G-4 PROPOSAL VALIDATOR TESTS PASSED"
    )
    print(f"Tests passed: {len(tests)}/{len(tests)}")


if __name__ == "__main__":
    main()
