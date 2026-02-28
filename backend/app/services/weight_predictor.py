import numpy as np

class WeightPredictor:
    def __init__(self, standard_width=1280, standard_height=720):
        self.standard_width = standard_width
        self.standard_height = standard_height
        
        # 1. Kalibrasi dari yolo_multiclass_weight.py
        self.calibration_data = {
            0.100: 0.05,  0.150: 0.076, 0.289: 0.15,  0.378: 0.19,  
            0.345: 0.18,  0.405: 0.21,  0.420: 0.25,  0.450: 0.26,  
            0.483: 0.29,  0.540: 0.30,  0.600: 0.302, 0.650: 0.304,   
            0.715: 0.306, 0.790: 0.309, 0.82: 0.313,  0.85: 0.316
        }
        real_weights = np.array(list(self.calibration_data.keys()))
        pred_weights = np.array(list(self.calibration_data.values()))
        self.poly_coeffs = np.polyfit(pred_weights, real_weights, 2)
        
        self.weight_factors = {
            'chicken': 0.78,
            'chicken drumstick': 0.5,
            'chicken neck': 0.53,
            'chicken wing': 0.4
        }

        # 2. Setup Region Grid (3x4) persis seperti kode lama
        self.regions = []
        grid_rows, grid_cols = 3, 4
        col_width, row_height = self.standard_width // grid_cols, self.standard_height // grid_rows
        base_factors = [
            [1.25, 1.20, 1.20, 1.25], 
            [1.10, 1.00, 1.00, 1.10], 
            [0.95, 0.90, 0.90, 0.95]  
        ]
        for i in range(grid_rows):
            for j in range(grid_cols):
                self.regions.append({
                    'coords': (j * col_width, i * row_height, (j + 1) * col_width, (i + 1) * row_height),
                    'factor': base_factors[i][j]
                })

        # 3. Static Obstacle Masks (Tiang & Tempat Makan)
        self.obstacle_masks = [
            {'name': 'Pole', 'coords': (250, 435, 530, 1200)},
            {'name': 'Feeder', 'coords': (730, 170, 910, 320)}
        ]

    def is_in_obstacle_zone(self, centroid):
        cx, cy = centroid
        for mask in self.obstacle_masks:
            x1, y1, x2, y2 = mask['coords']
            if x1 <= cx < x2 and y1 <= cy < y2:
                return True
        return False

    def get_region_factor(self, centroid):
        cx, cy = centroid
        for region in self.regions:
            x1, y1, x2, y2 = region['coords']
            if x1 <= cx < x2 and y1 <= cy < y2:
                return region['factor']
        return 1.0

    def predict_from_area(self, top_box, class_name, frame_shape):
        """Algoritma 2D: Persis sama dengan yolo_multiclass_weight.py untuk objek TANPA fusion."""
        x1, y1, x2, y2 = top_box
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        
        # Abaikan jika masuk obstacle zone
        if self.is_in_obstacle_zone((cx, cy)):
            return -1.0 
            
        area = (x2 - x1) * (y2 - y1)
        if area < 100: return 0.0
            
        frame_scale = min(self.standard_width / frame_shape[1], self.standard_height / frame_shape[0]) if frame_shape else 1.0
        normalized_area = area / (frame_scale * frame_scale)
        base_weight = 1.920 * (normalized_area / 71667)
        
        factor = self.weight_factors.get(class_name, 0.78)
        scaled_weight = base_weight * factor
        
        a, b, c = self.poly_coeffs
        calibrated_weight = a * scaled_weight**2 + b * scaled_weight + c
        
        final_weight = calibrated_weight * self.get_region_factor((cx, cy))
        return max(0.0, final_weight)

    def predict_from_volume(self, top_box, side_box, class_name, frame_shape):
        """Algoritma 3D: Untuk ayam yang TER-FUSION oleh kedua kamera."""
        x1_t, y1_t, x2_t, y2_t = top_box
        x1_s, y1_s, x2_s, y2_s = side_box
        cx, cy = (x1_t + x2_t) / 2.0, (y1_t + y2_t) / 2.0
        
        if self.is_in_obstacle_zone((cx, cy)):
            return -1.0
            
        width = x2_t - x1_t
        length = y2_t - y1_t
        height = y2_s - y1_s  # Mengambil tinggi dari kamera samping
        
        volume_idx = width * length * height
        if volume_idx < 1000: return 0.0
        
        frame_scale = min(self.standard_width / frame_shape[1], self.standard_height / frame_shape[0]) if frame_shape else 1.0
        normalized_vol = volume_idx / (frame_scale**3)
        
        # Sesuaikan pembagi ini (71667 * 150) dengan data riil volume Anda nanti
        base_weight = 1.920 * (normalized_vol / (71667 * 150)) 
        
        factor = self.weight_factors.get(class_name, 0.78)
        scaled_weight = base_weight * factor
        
        a, b, c = self.poly_coeffs
        calibrated_weight = a * scaled_weight**2 + b * scaled_weight + c
        
        final_weight = calibrated_weight * self.get_region_factor((cx, cy))
        return max(0.0, final_weight)

weight_predictor = WeightPredictor()