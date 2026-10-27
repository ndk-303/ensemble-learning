# Ensemble Learning: Bagging, AdaBoost, Stacking (Handwritten Digits 0-9)

Dự án tập trung trình bày và cài đặt chuyên sâu các giải thuật **Học kết hợp (Ensemble Learning)** từ nền tảng toán học, kết hợp tối ưu hóa các thành phần phụ trợ (mô hình cơ sở, tiền xử lý, đo đạc) bằng thư viện `scikit-learn`. Đánh giá thực nghiệm trên tập dữ liệu chuẩn nhận diện chữ số viết tay **Scikit-Learn Digits** (1797 mẫu, ảnh 8x8, 64 đặc trưng, 10 lớp `0..9`) và đối chiếu trực tiếp với `scikit-learn`.

---

## 1. Cấu trúc thư mục

```text
.
├── train.py                     # Entrypoint huan luyen & danh gia duy nhat (CLI Arguments)
├── src/                         # Ma nguon module hoa
│   ├── __init__.py              # Package init
│   ├── data.py                  # Tai va phan chia tap Digits, khoi tao base models
│   ├── utils.py                 # Logging chuan ML, in bang ASCII, xuat bieu do
│   └── methods/                 # Cac phuong phap Ensemble tu cai dat
│       ├── __init__.py          # Export Bagging, AdaBoost, Stacking
│       ├── bagging.py           # Thuat toan Bagging: Bootstrap sampling + Majority voting
│       ├── adaboost.py          # Thuat toan AdaBoost: SAMME da lop + trong so mau luy thua
│       └── stacking.py          # Thuat toan Stacking: K-Fold Out-of-Fold (OOF) chong ro ri
├── figs/                        # Bieu do ket qua truc quan hoa
│   ├── acc_vs_T.png             # Do chinh xac theo so luong mo hinh T (1..100)
│   ├── compare.png              # Bieu do so sanh Test Accuracy tren Digits
│   ├── compare_repeated.png     # Ket qua lap 10 lan ngau nhien (Mean ± Std)
│   └── confusion.png            # Ma tran nham lan cua mo hinh xuat sac nhat (Stacking)
├── docs/                        # Tai lieu dac ta kien truc & thiet ke
├── HUONG_DAN_CHAY_TAY_THUAT_TOAN_ENSEMBLE.md # So tay huong dan chay tay tung buoc (sinh vien)
├── README.md                    # Huong dan su dung & bao cao thuc nghiem
└── requirements.txt             # Thu vien phu thuoc (numpy, matplotlib, scikit-learn)
```

---

## 2. Điểm nhấn kỹ thuật Ensemble Learning cho bài toán Đa lớp (Multi-class)

Các kỹ thuật học kết hợp được thiết kế và mở rộng cho bài toán 10 lớp:

1. **Bagging ([`src/methods/bagging.py`](file:///D:/Khoa%2015/TON%20DUC%20THANG/ML/MidTerm/src/methods/bagging.py)):**
   * **Bootstrap Sampling:** Lấy ngẫu nhiên $n$ mẫu có hoàn lại từ tập huấn luyện ban đầu.
   * **Đa dạng hóa:** Huấn luyện $T$ cây quyết định sâu độc lập trên các tập con mẫu khác nhau nhằm triệt tiêu phương sai (variance).
   * **Majority Voting:** Tổng hợp kết quả đa lớp bằng ma trận phiếu bầu $V(x, k) = \sum_{t=1}^T \mathbb{I}(h_t(x) == k)$ và chọn lớp có số phiếu cao nhất: $\hat{y} = \arg\max_k V(x, k)$.
   * **Staged Predict:** Cung cấp generator dự đoán tích lũy để theo dõi sự ổn định khi tăng dần số lượng cây $T$.

2. **AdaBoost ([`src/methods/adaboost.py`](file:///D:/Khoa%2015/TON%20DUC%20THANG/ML/MidTerm/src/methods/adaboost.py)):**
   * **Giải thuật SAMME đa lớp:** Khởi tạo $w_i = 1/n$.
   * **Hệ số tin cậy SAMME:** Với $K=10$ lớp, hệ số $\alpha_t = \ln\left(\frac{1 - \epsilon_t}{\epsilon_t}\right) + \ln(K - 1)$, cho phép mô hình yếu chỉ cần đạt độ chính xác $> 1/K = 10\%$ là đã có đóng góp tích cực.
   * **Cập nhật lũy thừa:** Tăng trọng số các mẫu bị phân lớp sai: $w_i \leftarrow w_i \exp(\alpha_t \cdot \mathbb{I}(y_i \neq h_t(x_i)))$.
   * **Dự đoán kết hợp:** $\hat{y} = \arg\max_k \sum_{t=1}^T \alpha_t \cdot \mathbb{I}(h_t(x) == k)$.

3. **Stacking ([`src/methods/stacking.py`](file:///D:/Khoa%2015/TON%20DUC%20THANG/ML/MidTerm/src/methods/stacking.py)):**
   * **Chống rò rỉ dữ liệu (Out-of-fold):** Sử dụng K-Fold CV ($k=5$) trên tập huấn luyện để tạo ma trận đặc trưng mới ($Z \in \mathbb{R}^{n \times (M \times K)}$) dựa trên vector xác suất `predict_proba` từ $M$ mô hình cơ sở cho $K$ lớp.
   * **Đa dạng hóa cấu trúc:** Tầng 1 kết hợp 3 họ mô hình khác biệt (Cây quyết định, K-Nearest Neighbors, Hồi quy Logistic).
   * **Meta Learner:** Huấn luyện mô hình tầng 2 (Multinomial Logistic Regression) tối ưu hóa trọng số kết hợp.

---

## 3. Cài đặt & Chạy thực nghiệm

Cài đặt thư viện:
```bash
pip install -r requirements.txt
```

### Chạy mặc định toàn bộ thí nghiệm (kèm lặp 10 lần đánh giá độ ổn định):
```bash
python train.py
```

### Xem hướng dẫn tất cả các đối số CLI:
```bash
python train.py --help
```

### Các tùy chọn dòng lệnh (Command-Line Arguments):
| Tham số | Giá trị mặc định | Giải thích |
|---|:---:|---|
| `--seed` | `42` | Random seed cho phân chia dữ liệu và mô hình |
| `--test_size` | `0.3` | Tỷ lệ dữ liệu kiểm thử (30%) |
| `--bagging_t` | `50` | Số lượng cây quyết định trong Bagging |
| `--adaboost_t` | `100` | Số lượng weak learners trong AdaBoost SAMME |
| `--stacking_k` | `5` | Số fold cho K-Fold Out-of-Fold (OOF) trong Stacking |
| `--repeats` | `10` | Số lần lặp phân chia ngẫu nhiên (`0` để bỏ qua) |
| `--fig_dir` | `figs` | Thư mục lưu trữ biểu đồ |
| `--no_plots` | `False` | Bỏ qua bước vẽ biểu đồ (chỉ in log terminal) |

#### Ví dụ các câu lệnh tùy biến:
```bash
# Chạy nhanh với số lượng mô hình nhỏ và 3 lần lặp:
python train.py --bagging_t 25 --adaboost_t 50 --repeats 3

# Chỉ chạy thực nghiệm chính, bỏ qua bước lặp 10 lần để lấy kết quả tức thì:
python train.py --repeats 0

# Chạy kiểm thử terminal thuần, không ghi đè biểu đồ ra ổ đĩa:
python train.py --no_plots
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
