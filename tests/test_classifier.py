import pytest
import numpy as np
from eurocoins.classifier import EuroCoinClassifier


class TestEuroCoinClassifier:
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.classifier = EuroCoinClassifier()

    def test_estimate_pixel_to_mm_ratio_empty_list(self):
        """Test with empty pixel_diameters list should return 1.0."""
        result = self.classifier.estimate_pixel_to_mm_ratio([])
        assert result == 1.0

    def test_estimate_pixel_to_mm_ratio_single_value(self):
        """Test with single pixel diameter value."""
        pixel_diameters = [100.0]
        expected_ratio = 21.75 / 100.0  # 21.75mm / 100px = 0.2175
        result = self.classifier.estimate_pixel_to_mm_ratio(pixel_diameters)
        assert result == pytest.approx(expected_ratio, rel=1e-6)

    def test_estimate_pixel_to_mm_ratio_multiple_values_odd(self):
        """Test with odd number of values - median is middle value."""
        pixel_diameters = [60.0, 80.0, 100.0, 120.0, 140.0]  # median = 100.0
        expected_ratio = 21.75 / 100.0  # 0.2175
        result = self.classifier.estimate_pixel_to_mm_ratio(pixel_diameters)
        assert result == pytest.approx(expected_ratio, rel=1e-6)

    def test_estimate_pixel_to_mm_ratio_multiple_values_even(self):
        """Test with even number of values - median is average of middle two."""
        pixel_diameters = [60.0, 80.0, 120.0, 140.0]  # median = (80 + 120) / 2 = 100.0
        expected_ratio = 21.75 / 100.0  # 0.2175
        result = self.classifier.estimate_pixel_to_mm_ratio(pixel_diameters)
        assert result == pytest.approx(expected_ratio, rel=1e-6)

    def test_estimate_pixel_to_mm_ratio_unsorted_values(self):
        """Test with unsorted values - numpy.median should handle this."""
        pixel_diameters = [140.0, 60.0, 100.0, 80.0, 120.0]  # median = 100.0
        expected_ratio = 21.75 / 100.0  # 0.2175
        result = self.classifier.estimate_pixel_to_mm_ratio(pixel_diameters)
        assert result == pytest.approx(expected_ratio, rel=1e-6)

    def test_estimate_pixel_to_mm_ratio_realistic_scenario(self):
        """Test with realistic pixel diameter values for euro coins."""
        # Simulate detected diameters: small coins ~50px, medium coins ~70px, large coins ~90px
        pixel_diameters = [50.0, 56.0, 64.0, 70.0, 76.0, 84.0, 90.0]  # median = 70.0
        expected_ratio = 21.75 / 70.0  # 0.3107
        result = self.classifier.estimate_pixel_to_mm_ratio(pixel_diameters)
        assert result == pytest.approx(expected_ratio, rel=1e-6)

    def test_estimate_pixel_to_mm_ratio_very_small_values(self):
        """Test with very small pixel diameter values."""
        pixel_diameters = [2.0, 4.0, 6.0]  # median = 4.0
        expected_ratio = 21.75 / 4.0  # 5.4375
        result = self.classifier.estimate_pixel_to_mm_ratio(pixel_diameters)
        assert result == pytest.approx(expected_ratio, rel=1e-6)

    def test_estimate_pixel_to_mm_ratio_very_large_values(self):
        """Test with very large pixel diameter values."""
        pixel_diameters = [200.0, 400.0, 600.0]  # median = 400.0
        expected_ratio = 21.75 / 400.0  # 0.054375
        result = self.classifier.estimate_pixel_to_mm_ratio(pixel_diameters)
        assert result == pytest.approx(expected_ratio, rel=1e-6)

    def test_estimate_pixel_to_mm_ratio_identical_values(self):
        """Test with all identical values."""
        pixel_diameters = [80.0, 80.0, 80.0, 80.0]  # median = 80.0
        expected_ratio = 21.75 / 80.0  # 0.271875
        result = self.classifier.estimate_pixel_to_mm_ratio(pixel_diameters)
        assert result == pytest.approx(expected_ratio, rel=1e-6)

    def test_estimate_pixel_to_mm_ratio_return_type(self):
        """Test that the method returns a Python float, not numpy type."""
        pixel_diameters = [60.0, 80.0, 100.0]
        result = self.classifier.estimate_pixel_to_mm_ratio(pixel_diameters)
        assert isinstance(result, float)
        assert not isinstance(result, np.floating)

    def test_estimate_pixel_to_mm_ratio_with_floats_and_ints(self):
        """Test with mixed integer and float values."""
        pixel_diameters = [60, 80.0, 100, 120.0, 140]  # median = 100.0
        expected_ratio = 21.75 / 100.0  # 0.2175
        result = self.classifier.estimate_pixel_to_mm_ratio(pixel_diameters)
        assert result == pytest.approx(expected_ratio, rel=1e-6)

    def test_official_diameters_setup(self):
        """Test that official diameters are correctly initialized."""
        expected_diameters = {
            '1c': 16.25,
            '2c': 18.75,
            '5c': 21.25,
            '10c': 19.75,
            '20c': 22.25,
            '50c': 24.25,
            '1': 23.25,
            '2': 25.75
        }
        assert self.classifier.official_diameters == expected_diameters

    def test_diameter_median_calculation(self):
        """Test that diameter median calculation is correct."""
        # Official diameters: 16.25, 18.75, 19.75, 21.25, 22.25, 23.25, 24.25, 25.75
        # Sorted: 16.25, 18.75, 19.75, 21.25, 22.25, 23.25, 24.25, 25.75
        # Median = (21.25 + 22.25) / 2 = 21.75
        expected_median = 21.75
        actual_median = float(np.median(list(self.classifier.official_diameters.values())))
        assert actual_median == pytest.approx(expected_median, rel=1e-6)

    def test_initial_state(self):
        """Test that classifier is initialized with correct default values."""
        assert self.classifier.collected_diameters == []
        assert self.classifier.pixel_to_mm_ratio is None
