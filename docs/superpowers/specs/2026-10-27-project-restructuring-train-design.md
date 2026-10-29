# Thiết Kế Kiến Trúc: Tái Cấu Trúc Thư Mục Dự Án & File Huấn Luyện `train.py`

> Tài liệu lịch sử của lần tái cấu trúc 2026-10-27. Giao diện CLI và thiết kế thực nghiệm hiện hành được mô tả trong `README.md`.

- **Ngày tạo:** 2026-10-27
- **Tác giả:** Antigravity & User
- **Trạng thái:** Đã phê duyệt (Approved)

---

## 1. Bối cảnh & Mục tiêu

Dự án hiện tại là bài toán so sánh thực nghiệm 3 thuật toán Học kết hợp (Ensemble Learning): Bagging, AdaBoost (SAMME), Stacking trên tập dữ liệu chữ số viết tay Scikit-Learn Digits (0-9).
Codebase ban đầu phân bố phẳng tại thư mục gốc với các file `bagging.py`, `adaboost.py`, `stacking.py`, `experiments.py`, và `main.py`.

### Mục tiêu:
1. **Module hóa cấu trúc thư mục:** Chuyển các thuật toán tự cài đặt vào thư mục `src/methods/`, tách riêng module dữ liệu `src/data.py` và module tiện ích/hiển thị `src/utils.py`.
2. **Chuẩn hóa điểm khởi chạy:** Loại bỏ file `experiments.py` và `main.py`, thay thế bằng một file duy nhất `train.py` tiếp nhận các đối số dòng lệnh (`argparse`) theo chuẩn của các pipeline học máy chuyên nghiệp.
3. **Đảm bảo tính tương thích và kết quả:** Toàn bộ thuật toán, công thức toán học và kết quả đối chiếu khớp $100\%$ với thư viện Scikit-Learn phải được bảo toàn nguyên vẹn.

---

## 2. Cấu trúc thư mục mới

```text
.
├── train.py                     # Entrypoint huấn luyện & đánh giá duy nhất
├── src/
│   ├── __init__.py              # Package init
│   ├── data.py                  # Tải dữ liệu, chia Stratified Train/Test, base estimators
│   ├── utils.py                 # Logging chuẩn ML, in bảng ASCII, xuất biểu đồ
│   └── methods/                 # Thư mục chứa các phương pháp Ensemble
│       ├── __init__.py          # Export Bagging, AdaBoost, Stacking
│       ├── bagging.py           # Thuật toán Bagging (Bootstrap Aggregating)
│       ├── adaboost.py          # Thuật toán AdaBoost (SAMME Đa lớp)
│       └── stacking.py          # Thuật toán Stacking (K-Fold OOF)
├── figs/                        # Biểu đồ kết quả trực quan hóa
│   ├── acc_vs_T.png
│   ├── compare.png
│   ├── compare_repeated.png
│   └── confusion.png
├── docs/                        # Tài liệu đặc tả kỹ thuật
├── HUONG_DAN_CHAY_TAY_THUAT_TOAN_ENSEMBLE.md
├── README.md
└── requirements.txt
```

---

## 3. Thiết kế chi tiết các thành phần

### 3.1 `src/methods/`
- `src/methods/__init__.py`:
  ```python
  from .bagging import Bagging
  from .adaboost import AdaBoost
  from .stacking import Stacking
  ```
- Di chuyển `bagging.py`, `adaboost.py`, `stacking.py` vào đây và giữ nguyên logic toán học lõi.

### 3.2 `src/data.py`
- `load_data(test_size=0.3, seed=42)`: Tải `load_digits()`, chia Stratified train/test split theo tỷ lệ yêu cầu.
- `get_base_models(seed=0)`: Khởi tạo các mô hình cơ sở Scikit-Learn gồm:
  - Stump (1 tầng)
  - Cây sâu (depth=12)
  - KNN (k=5) với StandardScaler
  - Logistic Regression (max_iter=2000) với StandardScaler

### 3.3 `src/utils.py`
- `log(tag, msg)`: In log terminal có gắn thời gian thực `[HH:MM:SS]` và nhãn phân loại `[CONFIG]`, `[DATA]`, `[TRAIN]`, `[EVAL]`, `[BENCHMARK]`, `[STABILITY]`, `[VISUAL]`, `[DONE]`.
- `print_table(headers, rows, alignments=None)`: In bảng ASCII căn lề chuyên nghiệp.
- `save_visualizations(res, ytr, yte, Xtr, Xte, ada, bag, stk, adaboost_t, bagging_t, fig_dir)`: Xuất 3 biểu đồ đơn lẻ (`acc_vs_T.png`, `compare.png`, `confusion.png`).
- `save_stability_plot(names, means, stds, n, fig_dir)`: Xuất biểu đồ lặp ngẫu nhiên (`compare_repeated.png`).

### 3.4 `train.py`
- Phân tích đối số dòng lệnh qua `argparse`:
  - `--seed` (default: 42)
  - `--test_size` (default: 0.3)
  - `--bagging_t` (default: 50)
  - `--adaboost_t` (default: 100)
  - `--stacking_k` (default: 5)
  - `--repeats` (default: 10)
  - `--fig_dir` (default: "figs")
  - `--no_plots` (default: False)
- Điều phối toàn bộ quy trình:
  1. In banner & config
  2. Load data qua `src.data`
  3. Huấn luyện các mô hình cơ sở & Ensemble
  4. Đánh giá Accuracy & Fit time
  5. Đối chiếu kết quả với Scikit-Learn
  6. Thống kê ma trận nhầm lẫn
  7. Thực hiện lặp Monte-Carlo nếu `repeats > 0`
  8. Xuất biểu đồ nếu `not no_plots`
  9. In tổng thời gian chạy hoàn tất

---

## 4. Kế hoạch dọn dẹp & Di chuyển file
1. Tạo thư mục `src/methods/`
2. Di chuyển `bagging.py`, `adaboost.py`, `stacking.py` vào `src/methods/`
3. Tạo `src/data.py` và `src/utils.py`
4. Tạo `train.py`
5. Xóa file `experiments.py` và `main.py`
6. Cập nhật `README.md`
7. Chạy kiểm thử toàn bộ hệ thống với `python train.py --repeats 2` và `python train.py`
