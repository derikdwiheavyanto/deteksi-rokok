# Deteksi Rokok (Cigarette Detection)

## Deskripsi Proyek
Proyek ini adalah sistem deteksi rokok menggunakan model YOLO (You Only Look Once) untuk mendeteksi objek rokok dalam waktu nyata melalui webcam atau dari file video. Sistem ini menggunakan library Ultralytics YOLOv8 untuk melakukan deteksi objek dengan akurasi tinggi.

## Fitur
- Deteksi rokok dalam waktu nyata melalui webcam
- Pemrosesan file video untuk deteksi rokok
- Visualisasi bounding box dan label pada objek terdeteksi
- Output video yang dianotasi untuk file video

## Persyaratan Sistem
- Python 3.8 atau lebih baru
- Webcam (untuk mode real-time)
- File video input (untuk mode file video)

## Instalasi
1. Clone atau download repository ini.
2. Install dependencies dengan menjalankan:
   ```
   pip install -r requirement.txt
   ```
3. Pastikan model YOLO (`best.pt`) berada di folder `weight/`.

## Penggunaan

### Deteksi Real-Time dengan Webcam
Jalankan script `inference_yolo_webcam.py`:
```
python inference_yolo_webcam.py
```
- Script akan membuka webcam dan menampilkan deteksi rokok secara real-time.
- Tekan 'q' untuk keluar dari aplikasi.

### Deteksi dari File Video
Jalankan script `inference_yolo_file_video.py`:
```
python inference_yolo_file_video.py
```
- Pastikan file video bernama `test.mp4` berada di direktori yang sama.
- Output akan disimpan sebagai `output.mp4` dengan anotasi deteksi.

## Konfigurasi
- Model: `weight/best.pt` (YOLOv8 trained model)
- Confidence threshold: 0.5
- Image size: 640x640
- IoU threshold: 0.5 (untuk NMS)

## Dependencies
- ultralytics==8.4.13
- opencv-python==4.13.0.92
- torch==2.7.0+cu118
- torchvision==0.22.0+cu118
- numpy==2.2.6
- matplotlib==3.10.8
- pillow==12.1.0
- requests==2.32.5
- tqdm==4.67.3
- scipy==1.15.3
- pyyaml==6.0.3

## Struktur Proyek
```
deteksi rokok/
├── inference_yolo_webcam.py    # Script deteksi real-time webcam
├── inference_yolo_file_video.py # Script deteksi dari file video
├── requirement.txt             # File dependencies
├── readme.md                   # Dokumentasi proyek
├── weight/                     # Folder model YOLO
│   └── best.pt                 # Model terlatih
├── runs/                       # Folder output training (jika ada)
└── .gitignore                  # File gitignore
```

## Catatan
- Pastikan CUDA terinstall jika menggunakan GPU untuk akselerasi.
- Model `best.pt` harus dilatih terlebih dahulu menggunakan dataset rokok.
- Untuk training model, gunakan Ultralytics YOLOv8 dengan dataset yang sesuai.

## Lisensi
Proyek ini untuk tujuan gabut saja.
