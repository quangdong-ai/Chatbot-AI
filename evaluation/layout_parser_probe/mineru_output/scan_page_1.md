# CHUONG 4: TRIN KHAI H THÓNG VÀ ĐÁNH GIÁ THUCNGHIÊM

## 4.1. Môi trưòng và cu hình thc nghim

## 4.1.1. Cu hình phn cng ti nút biên

H thông đưc th nghim trên mt máy ch đc lp đóng vai trò là trung tâm d liu thu nh, mô phông li môi trưòng trm x lý biên ca lc lưng cnh sát giao thông ti các nút giao thông lón. Vic la chn cu hình đưc tính toán k lưõng da trên bài toán đim nghn d liu gia b x lý trung tâm CPU và b x lý đ ha GPU.

Bng 4.1. Cu hinh phn cng máy tính thc nghim

```
Linh kin   Thông s k thut chi tit   Vai trò trong h thông SmartTraffic
  CPU      Intel Core i5-12500H   Đm nhim x lý luồng s kin bt đng
           - 12 Cores (4 P-Cores, 8 E- b cůa FastAPI, vn hành thut toán tim
          Cores)                  kim Vector FAISS CPU, và tin x lý
           - 16 Threads, Xung nhip co khung hinh Video H.264 trưc khi đy
           bån 2.50 GHz           vào GPU.
  GPU      NVIDIA GeForce RTX     Phân h tính toán và suy lun AI trung
           3050 Ti Laptop GPU     tâm đåm nhim quá trình ni suy cho mô
           4 GB VRAM              hinh YOLOv8 và b gii mã quang hc
           H trą CUDA & Tensor    OCR. Vói 4 GB VRAM, h thông đóng
          Cores                   vai trò như mt Trm x lý biên đt ti
                                  ngã tu.
  RAM      16 GB (Tc đ 3200 MT/s) Cung cp b nh đ lưu tr b đm
                                  Redis, duy trì kho In-memory Vector
                                  Database (FAISS), và b nhó đm khung
                                  hinh Video.
 Luu trã   512 GB SSD NVMe        Đm bo tc đ truy xut siêu tc khi
                                  np trong s mng no-ron và lưu tr tm
                                  thi các nh ct bng chng.
```

Phân tích v giói hn phn cng: cu hinh thc nghim mô phng chính xác mt trm x lý biên đưc lp đt trc tip ti các tů điu khin ngã tư đưòng ph, thay vì s