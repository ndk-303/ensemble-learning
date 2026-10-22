# Ensemble Learning: Bagging, AdaBoost, Stacking (Handwritten Digits 0-9)

Dự án tập trung trình bày và cài đặt chuyên sâu các giải thuật **Học kết hợp (Ensemble Learning)** từ nền tảng toán học, kết hợp tối ưu hóa các thành phần phụ trợ (mô hình cơ sở, tiền xử lý, đo đạc) bằng thư viện `scikit-learn`. Đánh giá thực nghiệm trên tập dữ liệu chuẩn nhận diện chữ số viết tay **Scikit-Learn Digits** (1797 mẫu, ảnh 8x8, 64 đặc trưng, 10 lớp `0..9`) và đối chiếu trực tiếp với `scikit-learn`.

---

## 1. Cấu trúc thư mục

```text
.
├── bagging.py          # Cài đặt Bagging: Bootstrap sampling + Majority voting (Đa lớp)
├── adaboost.py         # Cài đặt AdaBoost: Trọng số mẫu lũy thừa + Thuật toán SAMME (Đa lớp)
├── stacking.py         # Cài đặt Stacking: K-Fold Out-of-Fold (OOF) chống rò rỉ dữ liệu (Đa lớp)
├── experiments.py      # Pipeline thực nghiệm Digits, đối chiếu với sklearn & vẽ biểu đồ
├── main.py             # Điểm khởi chạy toàn bộ thí nghiệm
├── requirements.txt    # Thư viện phụ thuộc (numpy, matplotlib, scikit-learn)
├── .gitignore          # Cấu hình bỏ qua file rác, virtualenv
└── figs/               # Biểu đồ và trực quan hóa kết quả
    ├── acc_vs_T.png        # Độ chính xác theo số lượng mô hình T (1..100)
    ├── boundary.png        # Ranh giới phân lớp 2D (PCA) của các mô hình trên 10 lớp số
    ├── compare.png         # Biểu đồ so sánh Test Accuracy trên Digits
    ├── compare_repeated.png# Kết quả lặp 10 lần ngẫu nhiên (Mean ± Std)
    └── confusion.png       # Ma trận nhầm lẫn 10x10 & mẫu ảnh chữ số 8x8
```

---

## 2. Điểm nhấn kỹ thuật Ensemble Learning cho bài toán Đa lớp (Multi-class)

Các kỹ thuật học kết hợp được thiết kế và mở rộng cho bài toán 10 lớp:

1. **Bagging ([`bagging.py`](file:///D:/Khoa%2015/TON%20DUC%20THANG/ML/MidTerm/bagging.py)):**
   * **Bootstrap Sampling:** Lấy ngẫu nhiên $n$ mẫu có hoàn lại từ tập huấn luyện ban đầu.
   * **Đa dạng hóa:** Huấn luyện $T$ cây quyết định sâu độc lập trên các tập con mẫu khác nhau nhằm triệt tiêu phương sai (variance).
   * **Majority Voting:** Tổng hợp kết quả đa lớp bằng ma trận phiếu bầu $V(x, k) = \sum_{t=1}^T \mathbb{I}(h_t(x) == k)$ và chọn lớp có số phiếu cao nhất: $\hat{y} = \arg\max_k V(x, k)$.
   * **Staged Predict:** Cung cấp generator dự đoán tích lũy để theo dõi sự ổn định khi tăng dần số lượng cây $T$.

2. **AdaBoost ([`adaboost.py`](file:///D:/Khoa%2015/TON%20DUC%20THANG/ML/MidTerm/adaboost.py)):**
   * **Giải thuật SAMME đa lớp:** Khởi tạo $w_i = 1/n$.
   * **Hệ số tin cậy SAMME:** Với $K=10$ lớp, hệ số $\alpha_t = \ln\left(\frac{1 - \epsilon_t}{\epsilon_t}\right) + \ln(K - 1)$, cho phép mô hình yếu chỉ cần đạt độ chính xác $> 1/K = 10\%$ là đã có đóng góp tích cực.
   * **Cập nhật lũy thừa:** Tăng trọng số các mẫu bị phân lớp sai: $w_i \leftarrow w_i \exp(\alpha_t \cdot \mathbb{I}(y_i \neq h_t(x_i)))$.
   * **Dự đoán kết hợp:** $\hat{y} = \arg\max_k \sum_{t=1}^T \alpha_t \cdot \mathbb{I}(h_t(x) == k)$.

3. **Stacking ([`stacking.py`](file:///D:/Khoa%2015/TON%20DUC%20THANG/ML/MidTerm/stacking.py)):**
   * **Chống rò rỉ dữ liệu (Out-of-fold):** Sử dụng K-Fold CV ($k=5$) trên tập huấn luyện để tạo ma trận đặc trưng mới ($Z \in \mathbb{R}^{n \times (M \times K)}$) dựa trên vector xác suất `predict_proba` từ $M$ mô hình cơ sở cho $K$ lớp.
   * **Đa dạng hóa cấu trúc:** Tầng 1 kết hợp 3 họ mô hình khác biệt (Cây quyết định, K-Nearest Neighbors, Hồi quy Logistic).
   * **Meta Learner:** Huấn luyện mô hình tầng 2 (Multinomial Logistic Regression) tối ưu hóa trọng số kết hợp.

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

## 4. Tóm tắt kết quả thực nghiệm trên tập Digits

### Đối chiếu với Scikit-Learn trên tập kiểm thử (1 lần chia 70/30)

| Mô hình | Tự cài đặt | Scikit-Learn |
|---|:---:|:---:|
| **KNN ($k=5$)** | $0.9704$ | $0.9704$ |
| **Logistic Regression** | $0.9815$ | $0.9815$ |
| **Bagging ($T=50$)** | $0.9352$ | $0.9333$ |
| **AdaBoost ($T=100$ stump)** | $0.8167$ | $0.8167$ |
| **Stacking (Tree+KNN+LR $\to$ LR)** | $\mathbf{0.9852}$ | $\mathbf{0.9852}$ |

*Nhận xét:*
- **AdaBoost** nâng độ chính xác của Decision Stump (1 tầng) từ **19.63%** lên **81.67%**, khớp tuyệt đối 100% với giải thuật SAMME của `scikit-learn`.
- **Stacking** kết hợp sức mạnh của Cây, KNN và Logistic Regression đạt độ chính xác kiểm thử cao nhất: **98.52%**.

### Độ ổn định qua 10 lần phân chia ngẫu nhiên (Mean $\pm$ Std)

| Mô hình | Test Accuracy (TB $\pm$ Std) |
|---|:---:|
| **Stump** (1 tầng) | $0.1911 \pm 0.0093$ |
| **Cây sâu** (depth=12) | $0.8506 \pm 0.0196$ |
| **KNN** ($k=5$) | $0.9731 \pm 0.0073$ |
| **LogReg** | $0.9696 \pm 0.0055$ |
| **Bagging** ($T=25$) | $0.9520 \pm 0.0098$ |
| **AdaBoost** ($T=50$) | $0.7461 \pm 0.0401$ |
| **Stacking** | $\mathbf{0.9815 \pm 0.0051}$ |
