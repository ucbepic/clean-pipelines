"""Import-only smoke tests. No network calls."""


def test_package_imports():
    import prap_involved_agency  # noqa: F401


def test_run_importable():
    from prap_involved_agency import run  # noqa: F401

    assert callable(run)


def test_schemas_load():
    from prap_involved_agency.schemas import (
        Agency,
        AgencyExtraction,
        AgencyNameMatch,
        PrimaryCitationAnalysis,
        RunResult,
        SingleAgencyVerification,
        ValidatorCitationResult,
    )

    rr = RunResult(n_cases=0, n_agencies_extracted=0, output_path="/tmp/x.csv")
    assert rr.n_cases == 0
    assert rr.n_agencies_extracted == 0

    a = Agency(
        agency_name="X PD",
        evidence=["officer fired weapon"],
        role_description="responded",
    )
    assert a.has_dual_role is False

    # Ensure all citation/eval schemas instantiate
    PrimaryCitationAnalysis(has_citation=False, reasoning="r", quote="q", confidence="HIGH")
    ValidatorCitationResult(
        final_decision=False,
        validator_reasoning="r",
        verified_quote="q",
        evidence_strength="WEAK",
    )
    AgencyNameMatch(is_match=False, reasoning="r", confidence="HIGH")

    # AgencyExtraction with required fields
    AgencyExtraction(
        incident_type="Unclear",
        confidence_level="LOW",
        extraction_reasoning="r",
    )

    # SingleAgencyVerification with required fields
    # (corrected_agency_type is required but Optional[str])
    SingleAgencyVerification(
        verification_status="REJECTED",
        corrected_agency_type=None,
        verified_agency_name="X",
        verified_evidence=[],
        verified_role_description="",
        has_dual_role=False,
        confidence_level="LOW",
        verification_reasoning="r",
        recommendation="EXCLUDE",
    )


def test_prompts_load():
    from importlib import resources

    expected = [
        "extract_agencies.txt",
        "verify_agency.txt",
        "primary_citation.txt",
        "validator_citation.txt",
        "filter_summaries.txt",
        "extract_clean_indices.txt",
        "compare_agency_names.txt",
    ]
    for name in expected:
        body = (
            resources.files("prap_involved_agency.prompts")
            .joinpath(name)
            .read_text(encoding="utf-8")
        )
        assert len(body) > 100, f"prompt {name} unexpectedly short"
