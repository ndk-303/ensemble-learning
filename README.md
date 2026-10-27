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

### Xem hướng dẫn tất cả các đối số CLI:
```bash
python train.py --help
```

### Quy ước 6 tham số dòng lệnh (Command-Line Arguments):
| Tham số | Bắt buộc | Mặc định | Ràng buộc giá trị & Giải thích |
|---|:---:|:---:|---|
| `--method` | **Có** | - | `bagging` \| `boosting` \| `stacking` \| `all` (so sánh toàn bộ) |
| `--base_model` | Không | Tùy method | `dt` \| `knn` \| `svm` \| `lr`.<br>• **Bagging:** Nhận 1 model (mặc định: `dt`).<br>• **Boosting:** Chỉ hỗ trợ `dt` (Decision Stump max_depth=1).<br>• **Stacking:** Nhận nhiều models (mặc định: `dt,knn,svm`). |
| `--seed` | Không | `42` | `int >= 0`. Random seed phân chia dữ liệu và khởi tạo mô hình |
| `--test_size` | Không | `0.3` | `0.0 < float < 1.0`. Tỷ lệ tập kiểm thử |
| `--n_estimators` | Không | 50 (bag) / 100 (boost) | `int >= 1`. Áp dụng cho **bagging** và **boosting**. *Báo lỗi nếu truyền cho stacking.* |
| `--learning_rate` | Không | 1.0 (boost) | `float > 0`. Tốc độ học AdaBoost. *Báo lỗi nếu truyền cho bagging/stacking.* |

> **Quy tắc kiểm tra ràng buộc:** Các argument không áp dụng cho phương pháp được chọn sẽ được bỏ qua khi không truyền, và **báo lỗi tường minh** nếu người dùng truyền vào (ví dụ truyền `--n_estimators` khi chạy Stacking, hoặc truyền `--learning_rate` khi chạy Bagging).

#### Ví dụ các câu lệnh chạy thực nghiệm:
```bash
# 1. Chạy so sánh toàn bộ các mô hình và xuất bộ 3 biểu đồ phân tích:
python train.py --method all

# 2. Chạy Bagging với Decision Tree (mặc định 50 estimators):
python train.py --method bagging

# 3. Chạy Bagging với KNN và 30 estimators:
python train.py --method bagging --base_model knn --n_estimators 30

# 4. Chạy Boosting (AdaBoost) với 100 stump và learning_rate 1.0:
python train.py --method boosting --n_estimators 100 --learning_rate 1.0

# 5. Chạy Stacking với 3 mô hình cơ sở (dt, knn, svm) và Meta-model Logistic Regression (cv=5):
python train.py --method stacking

# 6. Chạy Stacking tùy biến mô hình cơ sở:
python train.py --method stacking --base_model dt,knn,lr --test_size 0.25 --seed 123
```

### Định dạng Terminal Log chuẩn hóa (5 Phần)
Terminal log được định dạng gọn gàng, súc tích và chỉ hiển thị các tham số tương ứng với phương pháp đang chạy:
```text
==================================================
 ENSEMBLE LEARNING | TRAINING
==================================================
 Method       : Boosting (AdaBoost SAMME)
 Base Model   : Decision Tree (Stump, max_depth=1)
 Dataset      : Scikit-Learn Digits (0-9)
 Seed         : 42
 Test Size    : 0.3
 N Estimators : 100
 Learning Rate: 1.0
--------------------------------------------------
[INFO] Dataset loaded: 1797 samples, 64 features (10 classes: 0..9)
[INFO] Train: 1257 | Test: 540
[INFO] Initializing AdaBoost (SAMME, 100 weak learners, lr=1.0)...
[TRAIN] Training started...
[TRAIN] Iterating weak learners: 1/100 (1%)
[TRAIN] Iterating weak learners: 20/100 (20%)
[TRAIN] Iterating weak learners: 40/100 (40%)
[TRAIN] Iterating weak learners: 60/100 (60%)
[TRAIN] Iterating weak learners: 80/100 (80%)
[TRAIN] Iterating weak learners: 100/100 (100%)
[DONE] Training completed in 0.50s

[EVAL] Accuracy      : 0.8167 (81.67%)
[EVAL] Training Time : 0.50s
==================================================
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
