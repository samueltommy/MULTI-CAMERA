import numpy as np

class WeightPredictor:
    def __init__(self, standard_width=1280, standard_height=720):
        self.standard_width = standard_width
        self.standard_height = standard_height
        self.cm_per_pixel = None
        
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

    def set_scale_ratio(self, ratio):
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
        """Algoritma 2D: Persis sama dengan yolo_multiclass_weight.py untuk objek TANPA fusion."""
        x1, y1, x2, y2 = top_det['box']
        cx, cy = top_det['center']
        
        # Abaikan jika masuk obstacle zone
        if self.is_in_obstacle_zone((cx, cy)):
            return -1.0
        
        if top_det.get('mask_area') and top_det['mask_area'] > 0:
            area_px = top_det['mask_area'] # Ekstraksi lekuk tubuh asli
        else:
            area_px = (x2 - x1) * (y2 - y1) # Fallback pakai Bounding Box

        if area_px < 100: return 0.0 # Abaikan jika area terlalu kecil (noise)
            
        if self.cm_per_pixel is None:
            # Fallback jika belum kalibrasi ArUco (Pakai rasio layar standar)
            frame_scale = min(self.standard_width / frame_shape[1], self.standard_height / frame_shape[0]) if frame_shape else 1.0
            normalized_area = area_px / (frame_scale * frame_scale)
            base_weight = 1.920 * (normalized_area / 71667.0)
        else:
            # KALIBRASI ARUCO AKTIF: Area piksel dikali kuadrat dari resolusi spasial
            area_cm2 = area_px * (self.cm_per_pixel ** 2)
            
            if area_cm2 < 10: return 0.0 
            
            # Pembagi 400.0 cm^2 adalah nilai asumsi luas ayam ukuran dewasa
            # (Anda bisa menyesuaikan (tuning) angka 400.0 ini nanti saat pengujian)
            normalized_area = area_cm2 / 400.0 
            base_weight = 1.920 * normalized_area
            
        # 3. Hitung Berat Final
        factor = self.weight_factors.get(class_name, 0.78)
        scaled_weight = base_weight * factor
        
        a, b, c = self.poly_coeffs
        calibrated_weight = a * scaled_weight**2 + b * scaled_weight + c
        
        final_weight = calibrated_weight * self.get_region_factor((cx, cy))
        return max(0.0, final_weight)

    def predict_from_volume(self, top_det, side_det, class_name, frame_shape):
        """Algoritma 3D: Untuk ayam yang TER-FUSION oleh kedua kamera."""
        x1_t, y1_t, x2_t, y2_t = top_det['box']
        x1_s, y1_s, x2_s, y2_s = side_det['box']
        cx, cy = top_det['center']
        
        if self.is_in_obstacle_zone((cx, cy)):
            return -1.0
            
        # 1. Dapatkan Luas Alas dalam Piksel (Dari Top Camera)
        if top_det.get('mask_area') and top_det['mask_area'] > 0:
            base_area_px = top_det['mask_area']
        else:
            base_area_px = (x2_t - x1_t) * (y2_t - y1_t)
            
        # 2. Dapatkan Tinggi dari Side Camera (Kita pakai Bounding Box Height karena 
        # tinggi ayam merepresentasikan postur vertikal penuhnya)
        height_px = y2_s - y1_s
        
        # Volume 3D dalam satuan piksel
        volume_px3 = base_area_px * height_px
        
        if volume_px3 < 1000: return 0.0
            
        # 3. Hitung Normalisasi Volume
        if self.cm_per_pixel is None:
            # Fallback tanpa ArUco
            frame_scale = min(self.standard_width / frame_shape[1], self.standard_height / frame_shape[0]) if frame_shape else 1.0
            normalized_vol = volume_px3 / (frame_scale ** 3)
            base_weight = 1.920 * (normalized_vol / (71667 * 150))
        else:
            # KALIBRASI ARUCO AKTIF: Volume piksel dikali pangkat 3 dari resolusi spasial
            volume_cm3 = volume_px3 * (self.cm_per_pixel ** 3)
            
            if volume_cm3 < 100: return 0.0
            
            # Pembagi 5000.0 cm^3 adalah asumsi volume spasial tubuh ayam dewasa
            # (Anda bisa melakukan tuning angka 5000.0 ini agar hasilnya presisi)
            normalized_vol = volume_cm3 / 5000.0
            base_weight = 1.920 * normalized_vol
            
        # 4. Hitung Berat Final
        factor = self.weight_factors.get(class_name, 0.78)
        scaled_weight = base_weight * factor
        
        a, b, c = self.poly_coeffs
        calibrated_weight = a * scaled_weight**2 + b * scaled_weight + c
        
        final_weight = calibrated_weight * self.get_region_factor((cx, cy))
        return max(0.0, final_weight)

weight_predictor = WeightPredictor()