import cv2
import numpy as np

def generate_marker_20cm(id=1):
    # 1. Pengaturan Resolusi (300 DPI untuk hasil cetak standar dan tajam)
    DPI = 300
    cm_to_inch = 2.54
    
    # 2. Hitung ukuran piksel untuk kotak hitam (20 cm)
    marker_size_cm = 20.0
    marker_size_px = int((marker_size_cm / cm_to_inch) * DPI) # Sekitar 2362 piksel
    
    # 3. Load dictionary ArUco (Wajib 4x4 untuk tesis Anda)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    
    # 4. Generate Marker Hitam
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, id, marker_size_px)
    
    # 5. WAJIB: Tambahkan Quiet Zone (Margin Putih)
    # Tambahkan 2 cm margin putih di setiap sisi agar algoritma bisa mendeteksinya
    margin_cm = 2.0
    margin_px = int((margin_cm / cm_to_inch) * DPI)
    
    # Buat canvas putih yang lebih besar
    canvas_size = marker_size_px + (2 * margin_px)
    canvas = np.ones((canvas_size, canvas_size), dtype=np.uint8) * 255
    
    # Tempelkan marker hitam tepat di tengah canvas putih
    canvas[margin_px:margin_px+marker_size_px, margin_px:margin_px+marker_size_px] = marker_img
    
    # 6. Simpan Gambar
    filename = f"marker_id{id}_20x20cm.png"
    cv2.imwrite(filename, canvas)
    
    print(f"✅ Berhasil menyimpan: {filename}")

if __name__ == "__main__":
    print("="*50)
    print("🚀 GENERATING ARUCO MARKERS UNTUK TESIS")
    print("="*50)
    
    # Buat 4 marker untuk 4 sudut lantai kandang
    for i in range(1, 5):
        generate_marker_20cm(id=i)
        
    print("="*50)
    print("🖨️ CARA PRINT YANG BENAR (SANGAT PENTING):")
    print("1. Buka Microsoft Word (Pilih ukuran kertas A4/F4).")
    print("2. Masukkan (Insert) salah satu gambar marker ke dalam Word.")
    print("3. Klik kanan pada gambar -> Pilih 'Size and Position'.")
    print("4. Atur ukurannya (Absolute Width & Height) menjadi PERSIS 24 cm.")
    print("   (Penjelasan: 20 cm untuk kotak hitam + 4 cm untuk margin kiri-kanan).")
    print("5. Print menggunakan kertas HVS biasa tebal (Jangan kertas Glossy/Foto yang memantulkan cahaya!).")
    print("6. Tempelkan kertas tersebut ke atas kardus datar sebelum diletakkan di kandang.")
    print("="*50)