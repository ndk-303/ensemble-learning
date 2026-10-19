# Ensemble Learning: Bagging, AdaBoost, Stacking

Dự án tập trung trình bày và cài đặt chuyên sâu các giải thuật **Học kết hợp (Ensemble Learning)** từ nền tảng toán học, kết hợp tối ưu hóa các thành phần phụ trợ (mô hình cơ sở, tiền xử lý, đo đạc) bằng thư viện `scikit-learn`. Đánh giá thực nghiệm trên tập dữ liệu chuẩn **Breast Cancer Wisconsin** và đối chiếu với `scikit-learn`.

---

## 1. Cấu trúc thư mục

```text
.
├── bagging.py          # Cài đặt Bagging: Bootstrap sampling + Majority voting
├── adaboost.py         # Cài đặt AdaBoost: Trọng số mẫu lũy thừa + SAMME nhị phân
├── stacking.py         # Cài đặt Stacking: K-Fold Out-of-Fold (OOF) chống rò rỉ dữ liệu
├── experiments.py      # Pipeline thực nghiệm, đối chiếu với sklearn & vẽ biểu đồ
├── main.py             # Điểm khởi chạy toàn bộ thí nghiệm
├── requirements.txt    # Thư viện phụ thuộc (numpy, matplotlib, scikit-learn)
├── .gitignore          # Cấu hình bỏ qua file rác, virtualenv
└── figs/               # Biểu đồ và trực quan hóa kết quả
    ├── acc_vs_T.png        # Độ chính xác theo số lượng mô hình T (1..100)
    ├── boundary.png        # Ranh giới phân lớp 2D của các mô hình
    ├── compare.png         # Biểu đồ so sánh Test Accuracy
    ├── compare_repeated.png# Kết quả lặp 10 lần ngẫu nhiên (Mean ± Std)
    └── confusion.png       # Ma trận nhầm lẫn (Confusion Matrix)
```

---

## 2. Điểm nhấn kỹ thuật Ensemble Learning

Các kỹ thuật học kết hợp được thiết kế và cài đặt rõ ràng từng bước toán học:

1. **Bagging ([`bagging.py`](file:///D:/Khoa%2015/TON%20DUC%20THANG/ML/MidTerm/bagging.py)):**
   * **Bootstrap Sampling:** Lấy ngẫu nhiên $n$ mẫu có hoàn lại từ tập huấn luyện ban đầu.
   * **Đa dạng hóa:** Huấn luyện $T$ cây quyết định sâu độc lập trên các tập con mẫu khác nhau nhằm triệt tiêu phương sai (variance).
   * **Majority Voting:** Tổng hợp kết quả bằng biểu quyết đa số qua hàm dấu $\text{sign}(\sum h_t(x))$.
   * **Staged Predict:** Cung cấp generator dự đoán tích lũy để theo dõi sự ổn định khi tăng dần số lượng cây $T$.

2. **AdaBoost ([`adaboost.py`](file:///D:/Khoa%2015/TON%20DUC%20THANG/ML/MidTerm/adaboost.py)):**
   * **Phân phối trọng số mẫu:** Khởi tạo $w_i = 1/n$.
   * **Cập nhật lũy thừa:** Tăng trọng số các mẫu bị phân lớp sai, giảm trọng số mẫu đúng: $w_i \leftarrow w_i \exp(-\alpha_t y_i h_t(x_i))$.
   * **Độ tin cậy của mô hình:** $\alpha_t = \frac{1}{2}\ln\left(\frac{1 - \epsilon_t}{\epsilon_t}\right)$ với sai số có trọng số $\epsilon_t$.
   * **Dự đoán kết hợp:** Tuyến tính theo hệ số tin cậy: $H(x) = \text{sign}\left(\sum_{t=1}^T \alpha_t h_t(x)\right)$.

3. **Stacking ([`stacking.py`](file:///D:/Khoa%2015/TON%20DUC%20THANG/ML/MidTerm/stacking.py)):**
   * **Chống rò rỉ dữ liệu (Out-of-fold):** Sử dụng K-Fold CV ($k=5$) trên tập huấn luyện để tạo ma trận đặc trưng mới (meta-features $Z$). Mô hình ở fold $k$ chỉ dự đoán trên tập dữ liệu chưa từng thấy ở fold đó.
   * **Đa dạng hóa cấu trúc:** Tầng 1 kết hợp 3 họ mô hình khác biệt (Cây quyết định, K-Nearest Neighbors, Hồi quy Logistic).
   * **Meta Learner:** Huấn luyện mô hình tầng 2 (Logistic Regression) để tìm ra trọng số kết hợp tối ưu giữa các mô hình tầng 1.

---

## 3. Cài đặt & Chạy thực nghiệm

Cài đặt thư viện:
```bash
pip install -r requirements.txt
```

Chạy toàn bộ thí nghiệm và lặp 10 lần:
```bash
python main.py
```

---

## 4. Tóm tắt kết quả thực nghiệm

### Đối chiếu với Scikit-Learn trên tập kiểm thử (1 lần chia 70/30)

| Mô hình | Tự cài đặt | Scikit-Learn |
|---|:---:|:---:|
| **KNN ($k=5$)** | $0.9649$ | $0.9649$ |
| **Logistic Regression** | $0.9708$ | $0.9708$ |
| **Bagging ($T=50$)** | $0.9649$ | $0.9591$ |
| **AdaBoost ($T=100$)** | $0.9766$ | $0.9766$ |
| **Stacking (Tree+KNN+LR $\to$ LR)** | $0.9766$ | $0.9825$ |

*Trọng số meta Logistic Regression học được cho 3 mô hình [Tree, KNN, LR]:* `[1.782, 2.582, 3.794]`.

### Độ ổn định qua 10 lần phân chia ngẫu nhiên (Mean $\pm$ Std)

| Mô hình | Test Accuracy (TB $\pm$ Std) |
|---|:---:|
| **Stump** (1 tầng) | $0.8813 \pm 0.0228$ |
| **Cây sâu** (depth=12) | $0.9298 \pm 0.0203$ |
| **KNN** ($k=5$) | $0.9620 \pm 0.0131$ |
| **LogReg** | $0.9696 \pm 0.0107$ |
| **Bagging** ($T=25$) | $0.9491 \pm 0.0172$ |
| **AdaBoost** ($T=50$) | $0.9520 \pm 0.0140$ |
| **Stacking** | $\mathbf{0.9661 \pm 0.0122}$ |
