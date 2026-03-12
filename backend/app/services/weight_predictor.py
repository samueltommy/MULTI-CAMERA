import numpy as np

class WeightPredictor:
    def __init__(self, standard_width=1280, standard_height=720):
        self.standard_width = standard_width
        self.standard_height = standard_height
        
        self.cm_per_pixel = 0.05
        
        self.weight_factors = {
            'chicken': 1.0,
            'chicken drumstick': 0.64,
            'chicken neck': 0.68,
            'chicken wing': 0.51
        }

        self.regions = []
        grid_rows, grid_cols = 3, 4
        col_width, row_height = self.standard_width // grid_cols, self.standard_height // grid_rows
        base_factors = [
            [1.05, 1.02, 1.02, 1.05], 
            [1.00, 1.00, 1.00, 1.00], 
            [0.98, 0.95, 0.95, 0.98]  
        ]
        for i in range(grid_rows):
            for j in range(grid_cols):
                self.regions.append({
                    'coords': (j * col_width, i * row_height, (j + 1) * col_width, (i + 1) * row_height),
                    'factor': base_factors[i][j]
                })

        self.obstacle_masks = [
            {'name': 'Pole', 'coords': (250, 435, 530, 1200)},
            {'name': 'Feeder', 'coords': (730, 170, 910, 320)}
        ]

        # ==========================================================
        # TABEL STANDAR CIOMAS & DYNAMIC OUTLIER LIMITS
        # ==========================================================
        self.ciomas_standard = {
            0: 0.042, 1: 0.056, 2: 0.073, 3: 0.094, 4: 0.118, 5: 0.145, 6: 0.176,
            7: 0.210, 8: 0.247, 9: 0.288, 10: 0.332, 11: 0.379, 12: 0.429, 13: 0.483,
            14: 0.540, 15: 0.600, 16: 0.663, 17: 0.729, 18: 0.798, 19: 0.870, 20: 0.945,
            21: 1.024, 22: 1.105, 23: 1.189, 24: 1.276, 25: 1.365, 26: 1.457, 27: 1.552,
            28: 1.649, 29: 1.747, 30: 1.846, 31: 1.945, 32: 2.045, 33: 2.146, 34: 2.247,
            35: 2.348
        }
        
        # Batas Default (Akan ditimpa oleh pipeline sesuai umur)
        self.min_valid_kg = 0.050
        self.max_valid_kg = 3.000

    def update_age_limits(self, age_days: int):
        """Dipanggil oleh pipeline saat sesi dimulai untuk mensetting batas wajar"""
        target_bw = self.ciomas_standard.get(age_days)
        if target_bw is None:
            if age_days > 35:
                target_bw = 2.348 + ((age_days - 35) * 0.1)
            else:
                target_bw = 0.05

        # DYNAMIC FILTER:
        # Kita buat sangat longgar agar tidak membuang ayam asli,
        # tapi cukup ketat untuk membuang kotak deteksi yang error.
        # Min = 30% dari standar, Max = 180% dari standar
        self.min_valid_kg = max(0.030, target_bw * 0.3)
        self.max_valid_kg = max(0.800, target_bw * 1.8) # Minimal set ke 800g agar aman
        
        print(f"[Weight Predictor] Age: {age_days} Days | Target: {target_bw:.3f} kg")
        print(f"[Weight Predictor] OUTLIER Filter set to: {self.min_valid_kg:.3f} kg - {self.max_valid_kg:.3f} kg")

    def set_scale_ratio(self, ratio):
        if ratio and ratio > 0:
            self.cm_per_pixel = ratio
            print(f"Weight Predictor now using scale: {ratio:.4f} cm/pixel")

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

    def predict_from_area(self, top_det, class_name, frame_shape):
        """Algoritma 2D Area"""
        try:
            x1, y1, x2, y2 = top_det['box']
            cx, cy = top_det['center']
            
            if self.is_in_obstacle_zone((cx, cy)):
                return -1.0
            
            area_px = float((x2 - x1) * (y2 - y1))
            cm_per_pixel = self.cm_per_pixel if self.cm_per_pixel is not None else 0.05
            area_cm2 = area_px * (cm_per_pixel ** 2)
            
            if area_cm2 < 0.5: return 0.0
            
            DENSITY_2D = 3.5
            base_weight_kg = ((area_cm2 ** 1.5) * DENSITY_2D) / 1000.0
            
            factor = self.weight_factors.get(class_name, 1.0)
            final_weight = base_weight_kg * factor * self.get_region_factor((cx, cy))
            
            # --- CEK OUTLIER DINAMIS ---
            if final_weight < self.min_valid_kg or final_weight > self.max_valid_kg:
                return 0.0  # Buang outlier

            # Hapus # di bawah ini jika ingin melihat log 2D
            # print(f"[DEBUG 2D] ID: {top_det.get('track_id')} | Final: {final_weight:.3f} kg")
            return final_weight
        except Exception as e:
            return 0.0

    def predict_from_volume(self, top_det, side_det, class_name, frame_shape):
        """Algoritma 3D Volume"""
        try:
            x1_t, y1_t, x2_t, y2_t = top_det['box']
            x1_s, y1_s, x2_s, y2_s = side_det['box']
            cx, cy = top_det['center']
            
            if self.is_in_obstacle_zone((cx, cy)):
                return -1.0
                
            base_area_px = float((x2_t - x1_t) * (y2_t - y1_t))
            height_px = float(y2_s - y1_s)
            
            volume_px3 = base_area_px * height_px
            cm_per_pixel = self.cm_per_pixel if self.cm_per_pixel is not None else 0.05
            volume_cm3 = volume_px3 * (cm_per_pixel ** 3)
            
            if volume_cm3 < 1.0: return 0.0
            
            DENSITY_G_PER_CM3 = 2.7 
            base_weight_kg = (volume_cm3 * DENSITY_G_PER_CM3) / 1000.0
            
            factor = self.weight_factors.get(class_name, 1.0)
            final_weight = base_weight_kg * factor * self.get_region_factor((cx, cy))
            
            # --- CEK OUTLIER DINAMIS ---
            if final_weight < self.min_valid_kg or final_weight > self.max_valid_kg:
                # Print sesekali agar kita tahu kalau ada yg dibuang
                # print(f"[DEBUG 3D] ID: {top_det.get('track_id')} | OUTLIER DIBUANG: {final_weight:.3f} kg")
                return 0.0 
            
            print(f"[DEBUG 3D] ID: {top_det.get('track_id')} | cm3: {volume_cm3:.2f} | Final: {final_weight:.3f} kg")
            return final_weight
        except Exception as e:
            return 0.0

weight_predictor = WeightPredictor()