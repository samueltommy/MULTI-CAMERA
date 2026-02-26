import numpy as np

class WeightPredictor:
    """
    Morphometric Regression Service (Process 4.0)
    Adapted from yolo_multiclass_weight.py calibration data.
    """
    def __init__(self, standard_width=1280, standard_height=720):
        self.standard_width = standard_width
        self.standard_height = standard_height
        
        # Expanded calibration data (real_weight: predicted_weight)
        self.calibration_data = {
            0.100: 0.05,  0.150: 0.076, 0.289: 0.15,  0.378: 0.19,  
            0.345: 0.18,  0.405: 0.21,  0.420: 0.25,  0.450: 0.26,  
            0.483: 0.29,  0.540: 0.30,  0.600: 0.302, 0.650: 0.304,   
            0.715: 0.306, 0.790: 0.309, 0.82: 0.313,  0.85: 0.316
        }
        
        # Compute polynomial coefficients for calibration
        real_weights = np.array(list(self.calibration_data.keys()))
        pred_weights = np.array(list(self.calibration_data.values()))
        self.poly_coeffs = np.polyfit(pred_weights, real_weights, 2)
        
        # Base factor for chicken
        self.chicken_factor = 0.78

        # Setup Region Grid (3x4)
        self.regions = []
        grid_rows, grid_cols = 3, 4
        col_width = self.standard_width // grid_cols
        row_height = self.standard_height // grid_rows

        base_factors = [
            [1.25, 1.20, 1.20, 1.25], # Top Row (Far)
            [1.10, 1.00, 1.00, 1.10], # Middle Row (Base accuracy)
            [0.95, 0.90, 0.90, 0.95]  # Bottom Row (Close)
        ]

        for i in range(grid_rows):
            for j in range(grid_cols):
                x1 = j * col_width
                y1 = i * row_height
                x2 = (j + 1) * col_width
                y2 = (i + 1) * row_height
                self.regions.append({
                    'coords': (x1, y1, x2, y2),
                    'factor': base_factors[i][j]
                })

    def get_region_factor(self, centroid):
        cx, cy = centroid
        for region in self.regions:
            x1, y1, x2, y2 = region['coords']
            if x1 <= cx < x2 and y1 <= cy < y2:
                return region['factor']
        return 1.0

    def calculate_weight(self, top_box, frame_shape=None):
        """Calculate calibrated weight based on area and centroid."""
        if not top_box:
            return 0.0
            
        x1, y1, x2, y2 = top_box
        area = (x2 - x1) * (y2 - y1)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        if area < 100:
            return 0.0
            
        # Determine scale if frame is not standard size
        if frame_shape:
            h, w = frame_shape[:2]
            frame_scale = min(self.standard_width / w, self.standard_height / h)
        else:
            frame_scale = 1.0
            
        normalized_area = area / (frame_scale * frame_scale)
        base_weight = 1.920 * (normalized_area / 71667)
        scaled_weight = base_weight * self.chicken_factor

        # Apply calibration polynomial
        a, b, c = self.poly_coeffs
        calibrated_weight = a * scaled_weight**2 + b * scaled_weight + c
        
        # Apply region-based correction
        region_factor = self.get_region_factor((cx, cy))
        final_weight = calibrated_weight * region_factor
        
        return max(0.0, final_weight) # Clamp to zero to prevent negative weights

weight_predictor = WeightPredictor()