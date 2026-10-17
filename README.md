# Ensemble Learning: Bagging, AdaBoost, Stacking (NumPy From Scratch)

Dự án cài đặt thuần (from scratch bằng NumPy) và đánh giá các phương pháp học kết hợp (Ensemble Learning) trên tập dữ liệu **Breast Cancer Wisconsin**, có đối chiếu với `scikit-learn`.

## 1. Cấu trúc thư mục

```text
.
├── base_models.py      # Các mô hình cơ sở tự cài: Stump, Tree, KNN, LogReg
├── bagging.py          # Thuật toán Bagging (Bootstrap + Majority Voting)
├── adaboost.py         # Thuật toán AdaBoost (SAMME nhị phân + Decision Stump)
├── stacking.py         # Thuật toán Stacking (K-Fold Out-of-fold meta-features)
├── experiments.py      # Pipeline thực nghiệm, đo đạc và trực quan hóa kết quả
├── main.py             # Script thực thi chính
├── requirements.txt    # Thư viện phụ thuộc (numpy, matplotlib, scikit-learn)
├── .gitignore          # Cấu hình bỏ qua file rác, virtualenv
└── figs/               # Biểu đồ và đồ thị kết quả thực nghiệm
    ├── acc_vs_T.png        # Độ chính xác theo số lượng mô hình (T)
    ├── boundary.png        # Ranh giới phân lớp 2D của các mô hình
    ├── compare.png         # Biểu đồ so sánh Test Accuracy
    ├── compare_repeated.png# Kết quả lặp 10 lần ngẫu nhiên (Mean ± Std)
    └── confusion.png       # Ma trận nhầm lẫn (Confusion Matrix)
```

## 2. Cài đặt & Thực thi

Cài đặt thư viện:
```bash
pip install -r requirements.txt
```

Chạy toàn bộ thực nghiệm:
```bash
python main.py
```

## 3. Các mô hình cài đặt

- **Base Models**:
  - `Stump`: Decision tree 1 tầng tối thiểu hóa sai số có trọng số.
  - `Tree`: Cây quyết định đầy đủ dựa trên Binary Entropy và Information Gain.
  - `KNN`: K-Nearest Neighbors với chuẩn hóa Z-score.
  - `LogReg`: Logistic Regression tối ưu bằng Gradient Descent có L2 regularization.
- **Ensemble Models**:
  - `Bagging`: Lấy mẫu bootstrap có hoàn lại, kết hợp $T$ cây quyết định sâu qua biểu quyết đa số.
  - `AdaBoost`: Cập nhật trọng số mẫu theo cấp số nhân, tăng trọng số cho các mẫu dự đoán sai.
  - `Stacking`: Tầng 1 gồm `Tree(4)`, `KNN(5)`, `LogReg` tạo meta-features qua 5-fold out-of-fold; Tầng 2 dùng `LogReg` để tổng hợp.

## 4. Tóm tắt kết quả thực nghiệm

Kết quả trung bình trên 10 lần chia Train/Test ngẫu nhiên (70/30):

| Mô hình | Test Accuracy (TB $\pm$ Std) |
|---|:---:|
| **Stump** (1 tầng) | $0.8959 \pm 0.0173$ |
| **Cây sâu** (depth=12) | $0.9256 \pm 0.0116$ |
| **KNN** ($k=5$) | $0.9628 \pm 0.0108$ |
| **LogReg** | $0.9750 \pm 0.0154$ |
| **Bagging** ($T=25$) | $0.9552 \pm 0.0104$ |
| **AdaBoost** ($T=50$) | $0.9605 \pm 0.0137$ |
| **Stacking** | $\mathbf{0.9756 \pm 0.0103}$ |
