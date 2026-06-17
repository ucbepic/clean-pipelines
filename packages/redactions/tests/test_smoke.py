"""Import-shape smoke tests. No API calls."""


def test_package_imports():
    import prap_redactions  # noqa: F401


def test_run_callable_exported():
    from prap_redactions import run

    assert callable(run)


def test_schemas_load():
    from prap_redactions.schemas import (
        AddressFinding,
        AddressResponse,
        FormattedOutput,
        FormattedPerson,
        NameExtractionResponse,
        PersonExtraction,
        PersonVerification,
        RunResult,
        VerificationResponse,
    )

    # Construct each with required fields; pydantic ValidationError would
    # surface here if the schemas were misdefined.
    PersonExtraction(
        name="x", person_type="victim", reasoning="r", confidence="high", context="c"
    )
    NameExtractionResponse()
    PersonVerification(name="x", should_be_included=True, reasoning="r")
    VerificationResponse(verified_persons=[])
    FormattedPerson(name="x", person_type=["victim"], confidence="high", reasoning="r")
    FormattedOutput()
    AddressFinding(name="x", address_found=False, reasoning="r")
    AddressResponse(addresses=[])
    RunResult(
        n_cases_attempted=0,
        n_cases_processed=0,
        n_cases_errored=0,
        n_files_with_redactions=0,
        output_path="/tmp/out.csv",
    )
