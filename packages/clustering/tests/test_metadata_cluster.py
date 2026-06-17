"""
Test suite for metadata clustering pipeline.

Following TDD principles:
1. Write failing test first
2. Watch it fail
3. Write minimal code to pass
4. Refactor
"""



# Test parse_feature_value() - Parses CSV string representations
class TestParseFeatureValue:
    """Tests for parsing feature values from CSV strings."""

    def test_parse_empty_list_string(self):
        """Should return None for empty list string '[]'"""
        from prap_clustering.metadata_pipeline.clustering.cluster import parse_feature_value

        result = parse_feature_value('[]')
        assert result is None

    def test_parse_none_value(self):
        """Should return None for None input"""
        from prap_clustering.metadata_pipeline.clustering.cluster import parse_feature_value

        result = parse_feature_value(None)
        assert result is None

    def test_parse_string_list_with_single_value(self):
        """Should parse string list with single value: \"['2023-01-15']\" """
        from prap_clustering.metadata_pipeline.clustering.cluster import parse_feature_value

        result = parse_feature_value("['2023-01-15']")
        assert result == ['2023-01-15']

    def test_parse_string_list_with_multiple_values(self):
        """Should parse string list with multiple values"""
        from prap_clustering.metadata_pipeline.clustering.cluster import parse_feature_value

        result = parse_feature_value("['IAD-2023-01', 'IAD-2023-02']")
        assert result == ['IAD-2023-01', 'IAD-2023-02']

    def test_parse_comma_separated_string(self):
        """Should parse comma-separated string: 'val1, val2'"""
        from prap_clustering.metadata_pipeline.clustering.cluster import parse_feature_value

        result = parse_feature_value('val1, val2')
        assert result == ['val1', 'val2']

    def test_parse_empty_string(self):
        """Should return None for empty string"""
        from prap_clustering.metadata_pipeline.clustering.cluster import parse_feature_value

        result = parse_feature_value('')
        assert result is None

    def test_parse_string_none(self):
        """Should return None for string 'None'"""
        from prap_clustering.metadata_pipeline.clustering.cluster import parse_feature_value

        result = parse_feature_value('None')
        assert result is None


# Test calculate_edge_weight() - Core clustering business logic
class TestCalculateEdgeWeight:
    """Tests for the core clustering decision logic."""

    def test_shared_case_id_returns_1_0(self):
        """Rule 1: Any shared case ID should cluster (return 1.0)"""
        from prap_clustering.metadata_pipeline.clustering.cluster import (
            calculate_edge_weight_metadata,
        )

        doc1 = {
            'extracted_case_ids_fp': "['IAD-2023-01']",
            'extracted_case_ids_fn': None,
            'extracted_dates_fp': None,
            'extracted_dates_fn': None,
            'extracted_names_fp': None,
            'extracted_names_fn': None,
        }
        doc2 = {
            'extracted_case_ids_fp': "['IAD-2023-01']",
            'extracted_case_ids_fn': None,
            'extracted_dates_fp': None,
            'extracted_dates_fn': None,
            'extracted_names_fp': None,
            'extracted_names_fn': None,
        }

        weight = calculate_edge_weight_metadata(doc1, doc2, 'filepath_only')
        assert weight == 1.0

    def test_no_shared_case_id_returns_0_0(self):
        """Different case IDs with no other overlap should not cluster"""
        from prap_clustering.metadata_pipeline.clustering.cluster import (
            calculate_edge_weight_metadata,
        )

        doc1 = {
            'extracted_case_ids_fp': "['IAD-2023-01']",
            'extracted_case_ids_fn': None,
            'extracted_dates_fp': None,
            'extracted_dates_fn': None,
            'extracted_names_fp': None,
            'extracted_names_fn': None,
        }
        doc2 = {
            'extracted_case_ids_fp': "['IAD-2023-99']",
            'extracted_case_ids_fn': None,
            'extracted_dates_fp': None,
            'extracted_dates_fn': None,
            'extracted_names_fp': None,
            'extracted_names_fn': None,
        }

        weight = calculate_edge_weight_metadata(doc1, doc2, 'filepath_only')
        assert weight == 0.0

    def test_same_date_plus_2_names_returns_1_0(self):
        """Rule 2: Same date (within 30 days) + 2+ shared names should cluster"""
        from prap_clustering.metadata_pipeline.clustering.cluster import (
            calculate_edge_weight_metadata,
        )

        doc1 = {
            'extracted_case_ids_fp': None,
            'extracted_case_ids_fn': None,
            'extracted_dates_fp': "['2023-01-15']",
            'extracted_dates_fn': None,
            'extracted_names_fp': "['John Smith', 'Jane Doe']",
            'extracted_names_fn': None,
        }
        doc2 = {
            'extracted_case_ids_fp': None,
            'extracted_case_ids_fn': None,
            'extracted_dates_fp': "['2023-01-20']",  # Within 30 days
            'extracted_dates_fn': None,
            'extracted_names_fp': "['John Smith', 'Jane Doe']",
            'extracted_names_fn': None,
        }

        weight = calculate_edge_weight_metadata(doc1, doc2, 'filepath_only')
        assert weight == 1.0

    def test_same_date_only_1_name_returns_0_0(self):
        """Same date but only 1 shared name should not cluster"""
        from prap_clustering.metadata_pipeline.clustering.cluster import (
            calculate_edge_weight_metadata,
        )

        doc1 = {
            'extracted_case_ids_fp': None,
            'extracted_case_ids_fn': None,
            'extracted_dates_fp': "['2023-01-15']",
            'extracted_dates_fn': None,
            'extracted_names_fp': "['John Smith']",
            'extracted_names_fn': None,
        }
        doc2 = {
            'extracted_case_ids_fp': None,
            'extracted_case_ids_fn': None,
            'extracted_dates_fp': "['2023-01-20']",
            'extracted_dates_fn': None,
            'extracted_names_fp': "['John Smith']",
            'extracted_names_fn': None,
        }

        weight = calculate_edge_weight_metadata(doc1, doc2, 'filepath_only')
        assert weight == 0.0

    def test_2_shared_names_no_date_returns_1_0(self):
        """Rule 3: 2+ shared names (exact match) should cluster even without dates"""
        from prap_clustering.metadata_pipeline.clustering.cluster import (
            calculate_edge_weight_metadata,
        )

        doc1 = {
            'extracted_case_ids_fp': None,
            'extracted_case_ids_fn': None,
            'extracted_dates_fp': None,
            'extracted_dates_fn': None,
            'extracted_names_fp': "['John Smith', 'Jane Doe']",
            'extracted_names_fn': None,
        }
        doc2 = {
            'extracted_case_ids_fp': None,
            'extracted_case_ids_fn': None,
            'extracted_dates_fp': None,
            'extracted_dates_fn': None,
            'extracted_names_fp': "['John Smith', 'Jane Doe']",
            'extracted_names_fn': None,
        }

        weight = calculate_edge_weight_metadata(doc1, doc2, 'filepath_only')
        assert weight == 1.0

    def test_no_overlap_returns_0_0(self):
        """Rule 4: No overlap should not cluster"""
        from prap_clustering.metadata_pipeline.clustering.cluster import (
            calculate_edge_weight_metadata,
        )

        doc1 = {
            'extracted_case_ids_fp': None,
            'extracted_case_ids_fn': None,
            'extracted_dates_fp': "['2023-01-15']",
            'extracted_dates_fn': None,
            'extracted_names_fp': "['John Smith']",
            'extracted_names_fn': None,
        }
        doc2 = {
            'extracted_case_ids_fp': None,
            'extracted_case_ids_fn': None,
            'extracted_dates_fp': "['2023-06-20']",
            'extracted_dates_fn': None,
            'extracted_names_fp': "['Jane Doe']",
            'extracted_names_fn': None,
        }

        weight = calculate_edge_weight_metadata(doc1, doc2, 'filepath_only')
        assert weight == 0.0

    def test_combined_mode_uses_both_sources(self):
        """In combined mode, should use union of filepath and filename features"""
        from prap_clustering.metadata_pipeline.clustering.cluster import (
            calculate_edge_weight_metadata,
        )

        doc1 = {
            'extracted_case_ids_fp': "['IAD-2023-01']",
            'extracted_case_ids_fn': None,
            'extracted_dates_fp': None,
            'extracted_dates_fn': None,
            'extracted_names_fp': None,
            'extracted_names_fn': None,
        }
        doc2 = {
            'extracted_case_ids_fp': None,
            'extracted_case_ids_fn': "['IAD-2023-01']",  # In filename instead
            'extracted_dates_fp': None,
            'extracted_dates_fn': None,
            'extracted_names_fp': None,
            'extracted_names_fn': None,
        }

        weight = calculate_edge_weight_metadata(doc1, doc2, 'combined')
        assert weight == 1.0


# Test normalization edge cases
class TestNormalization:
    """Tests for ID, name, and date normalization."""

    def test_case_id_with_different_separators_should_match(self):
        """IAD_552 and IAD-552 should be treated as same ID"""
        from prap_clustering.metadata_pipeline.clustering.cluster import (
            calculate_edge_weight_metadata,
        )

        doc1 = {
            'extracted_case_ids_fp': "['IAD_552']",
            'extracted_case_ids_fn': None,
            'extracted_dates_fp': None,
            'extracted_dates_fn': None,
            'extracted_names_fp': None,
            'extracted_names_fn': None,
        }
        doc2 = {
            'extracted_case_ids_fp': "['IAD-552']",
            'extracted_case_ids_fn': None,
            'extracted_dates_fp': None,
            'extracted_dates_fn': None,
            'extracted_names_fp': None,
            'extracted_names_fn': None,
        }

        weight = calculate_edge_weight_metadata(doc1, doc2, 'filepath_only')
        assert weight == 1.0

    def test_case_id_with_different_case_should_match(self):
        """iad-2023-01 and IAD-2023-01 should match (case insensitive)"""
        from prap_clustering.metadata_pipeline.clustering.cluster import (
            calculate_edge_weight_metadata,
        )

        doc1 = {
            'extracted_case_ids_fp': "['iad-2023-01']",
            'extracted_case_ids_fn': None,
            'extracted_dates_fp': None,
            'extracted_dates_fn': None,
            'extracted_names_fp': None,
            'extracted_names_fn': None,
        }
        doc2 = {
            'extracted_case_ids_fp': "['IAD-2023-01']",
            'extracted_case_ids_fn': None,
            'extracted_dates_fp': None,
            'extracted_dates_fn': None,
            'extracted_names_fp': None,
            'extracted_names_fn': None,
        }

        weight = calculate_edge_weight_metadata(doc1, doc2, 'filepath_only')
        assert weight == 1.0

    def test_names_with_different_case_should_match(self):
        """john smith and John Smith should match"""
        from prap_clustering.metadata_pipeline.clustering.cluster import (
            calculate_edge_weight_metadata,
        )

        doc1 = {
            'extracted_case_ids_fp': None,
            'extracted_case_ids_fn': None,
            'extracted_dates_fp': None,
            'extracted_dates_fn': None,
            'extracted_names_fp': "['john smith', 'jane doe']",
            'extracted_names_fn': None,
        }
        doc2 = {
            'extracted_case_ids_fp': None,
            'extracted_case_ids_fn': None,
            'extracted_dates_fp': None,
            'extracted_dates_fn': None,
            'extracted_names_fp': "['John Smith', 'Jane Doe']",
            'extracted_names_fn': None,
        }

        weight = calculate_edge_weight_metadata(doc1, doc2, 'filepath_only')
        assert weight == 1.0

    def test_names_with_titles_should_match_without_titles(self):
        """'Officer John Smith' and 'John Smith' should match"""
        from prap_clustering.metadata_pipeline.clustering.cluster import (
            calculate_edge_weight_metadata,
        )

        doc1 = {
            'extracted_case_ids_fp': None,
            'extracted_case_ids_fn': None,
            'extracted_dates_fp': None,
            'extracted_dates_fn': None,
            'extracted_names_fp': "['Officer John Smith', 'Det. Jane Doe']",
            'extracted_names_fn': None,
        }
        doc2 = {
            'extracted_case_ids_fp': None,
            'extracted_case_ids_fn': None,
            'extracted_dates_fp': None,
            'extracted_dates_fn': None,
            'extracted_names_fp': "['John Smith', 'Jane Doe']",
            'extracted_names_fn': None,
        }

        weight = calculate_edge_weight_metadata(doc1, doc2, 'filepath_only')
        assert weight == 1.0

    def test_dates_different_formats_same_day_should_match(self):
        """2023-01-15 and 01/15/2023 should be same date"""
        from prap_clustering.metadata_pipeline.clustering.cluster import (
            calculate_edge_weight_metadata,
        )

        doc1 = {
            'extracted_case_ids_fp': None,
            'extracted_case_ids_fn': None,
            'extracted_dates_fp': "['2023-01-15']",
            'extracted_dates_fn': None,
            'extracted_names_fp': "['John Smith', 'Jane Doe']",
            'extracted_names_fn': None,
        }
        doc2 = {
            'extracted_case_ids_fp': None,
            'extracted_case_ids_fn': None,
            'extracted_dates_fp': "['01/15/2023']",
            'extracted_dates_fn': None,
            'extracted_names_fp': "['John Smith', 'Jane Doe']",
            'extracted_names_fn': None,
        }

        weight = calculate_edge_weight_metadata(doc1, doc2, 'filepath_only')
        assert weight == 1.0
