import numpy as np
from typing import List, Tuple

class EuroCoinClassifier:
    def __init__(self) -> None:
        # https://economy-finance.ec.europa.eu/euro/euro-coins-and-notes/euro-coins/common-sides-euro-coins_en
        self.official_diameters = {
            '1c': 16.25,
            '2c': 18.75,
            '5c': 21.25,
            '10c': 19.75,
            '20c': 22.25,
            '50c': 24.25,
            '1': 23.25,
            '2': 25.75
        }

        self.collected_diameters = []
        self.pixel_to_mm_ratio = None

    def estimate_pixel_to_mm_ratio(self, pixel_diameters: List[float]) -> float:
        """
        Estimate pixel-to-mm conversion using statistical matching.
        Assumes we have a good distribution of different coin sizes.
        """
        if not pixel_diameters:
            return 1.0

        # Use median diameter as reference point
        median_pixel_diameter = float(np.median(pixel_diameters))

        # Assume median corresponds to median of official euro coin diameters
        estimated_mm_diameter = float(np.median(list(self.official_diameters.values())))

        return estimated_mm_diameter / median_pixel_diameter

    def classify_by_nearest_neighbor(self, diameter_mm: float) -> Tuple[str, float]:
        """
        Classify coin using nearest neighbor to official diameters.
        Returns (denomination, confidence_score)
        """
        official_values = np.array(list(self.official_diameters.values()))
        official_keys = list(self.official_diameters.keys())

        # Find closest match
        distances = np.abs(official_values - diameter_mm)
        closest_idx = np.argmin(distances)

        denomination = official_keys[closest_idx]
        confidence = max(0, 1 - (distances[closest_idx] / diameter_mm))  # Simple confidence metric

        return denomination, confidence
