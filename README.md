# Ensemble Learning trên bộ Digits

Project môn Nhập môn học máy: tìm hiểu và thực nghiệm **Bagging**, **AdaBoost SAMME** và **Stacking** cho bài toán phân loại 10 chữ số viết tay. Phần logic ensemble được cài đặt trong `src/methods/`; cây quyết định, KNN, SVM và Logistic Regression dùng `scikit-learn`.

## Câu hỏi thực nghiệm

1. Bagging cải thiện một cây quyết định sâu (`max_depth=12`) bao nhiêu?
2. AdaBoost cải thiện một cây một tầng (*decision stump*) bao nhiêu?
3. Stacking có vượt mô hình đơn mạnh nhất không?

Cả tám mô hình được huấn luyện và đánh giá trên **cùng một phép chia** ở mỗi lần lặp. Dữ liệu Digits có 1.797 mẫu, 64 đặc trưng và 10 lớp. Chia train/test có phân tầng, mặc định 70/30. Với `--repeats 10 --seed 42`, các lần chia dùng seed 42 đến 51. Chênh lệch (*paired gain*) được tính riêng trên từng lần chia rồi lấy trung bình.

| Mô hình | Cấu hình |
|---|---|
| Stump | Decision Tree `max_depth=1` |
| Tree | Decision Tree `criterion=entropy`, `max_depth=12` |
| KNN | StandardScaler + KNN (`k=5`) |
| LR | StandardScaler + Logistic Regression |
| SVM | StandardScaler + SVC RBF |
| Bagging | Bootstrap + 50 cây như Tree, bỏ phiếu đa số |
| AdaBoost | 100 stump, SAMME đa lớp, learning rate 1.0 |
| Stacking | DT `max_depth=6` + KNN + LR → LR, dự đoán out-of-fold qua 5 fold |

Stacking dùng dự đoán out-of-fold để huấn luyện tầng hai, sau đó huấn luyện lại các mô hình tầng một trên toàn bộ tập train. Tiền xử lý của KNN/LR/SVM nằm trong pipeline, nên scaler ở mỗi fold chỉ học trên dữ liệu train của fold đó.

## Cài đặt và chạy

```bash
pip install -r requirements.txt
python train.py --method all
```

Lệnh trên chạy 10 lần chia và in cấu hình, tiến độ từng lần, bảng kết quả lần đầu, bảng đối chiếu scikit-learn và bảng tổng kết. Các biểu đồ được lưu trong `figs/`. Muốn chạy nhanh hoặc không ghi biểu đồ:

```bash
python train.py --method all --repeats 2 --no_plots
python train.py --method all --base_model dt,knn,svm --repeats 10
python train.py --method bagging --n_estimators 50
python train.py --method boosting --n_estimators 100 --learning_rate 1.0
python train.py --method stacking --base_model dt,knn,lr
python train.py --help
```

Các tùy chọn chính: `--method` (bắt buộc: `bagging`, `boosting`, `stacking`, `all`), `--seed` (mặc định 42), `--test_size` (0–1, mặc định 0.3), `--n_estimators` (Bagging/AdaBoost; mặc định lần lượt 50/100), `--learning_rate` (AdaBoost; mặc định 1.0), `--base_model` (Bagging: một mô hình; Stacking và `all`: danh sách cách nhau bằng dấu phẩy), `--repeats` (chỉ dùng để so sánh `all`, mặc định 10), `--no_plots`, `--fig_dir` (mặc định `figs`). Với `all`, Bagging luôn dùng cây sâu, AdaBoost luôn dùng stump; `--base_model` chỉ thay danh sách mô hình tầng một của Stacking.

## Kết quả tham khảo

Kết quả từ `python train.py --method all --repeats 10 --no_plots` với seed 42–51, `test_size=0.3`, NumPy 2.4.6 và scikit-learn 1.9.1:

| Mô hình | Test accuracy, trung bình ± độ lệch chuẩn | Chênh lệch ghép cặp |
|---|---:|---:|
| Stump | 0.1989 ± 0.0024 | — |
| Tree | 0.8533 ± 0.0149 | — |
| KNN | 0.9722 ± 0.0041 | — |
| LR | 0.9691 ± 0.0076 | — |
| SVM | 0.9796 ± 0.0053 | — |
| Bagging | 0.9531 ± 0.0089 | +0.0998 so với Tree |
| AdaBoost | 0.8113 ± 0.0175 | +0.6124 so với Stump |
| Stacking | 0.9804 ± 0.0031 | +0.0007 so với SVM |

Độ lệch chuẩn ở đây là độ lệch chuẩn tổng thể của 10 điểm số, không phải khoảng tin cậy. Các phép chia có dữ liệu chồng lặp, nên không dùng các con số này để khẳng định ý nghĩa thống kê. Trong lần chia seed 42, Stacking đạt 0.9852; lợi thế trung bình trước SVM qua 10 lần chia chỉ 0.0007. Điều này cho thấy cần xem cả các lần lặp, thay vì kết luận từ một lần chia thuận lợi.

Baseline của Stacking là mô hình đơn có accuracy trung bình cao nhất trong bốn mô hình Tree, KNN, LR và SVM. Đây là so sánh mô tả trên tập test, không phải quy trình chọn mô hình để triển khai; nếu cần chọn mô hình, hãy dùng validation chỉ trong tập train rồi giữ một tập test cuối chưa sử dụng.

### Đối chiếu với scikit-learn

Lệnh `--method all` còn huấn luyện các lớp ensemble tương ứng của scikit-learn trên **lần chia đầu tiên** với cùng mô hình cơ sở và các siêu tham số chính. Kết quả seed 42:

| Ensemble | Tự cài đặt | scikit-learn |
|---|---:|---:|
| Bagging | 0.9407 | 0.9463 |
| AdaBoost | 0.8167 | 0.8167 |
| Stacking | 0.9852 | 0.9852 |

Điểm số bằng nhau trên một tập kiểm thử không chứng minh hai mô hình có dự đoán giống nhau trên mọi mẫu. Bagging có cách lấy mẫu ngẫu nhiên khác giữa hai bản cài đặt; Stacking cũng có thể khác ở cách tạo đặc trưng xác suất. Bảng này là kiểm tra tham khảo, còn bảng 10 lần chia ở trên dùng để so sánh hiệu năng.

## Cấu trúc

- `train.py`: CLI và log huấn luyện.
- `src/data.py`: tải dữ liệu, chia có phân tầng và tạo mô hình cơ sở.
- `src/experiments.py`: cấu hình các baseline, chạy cùng phép chia và tổng hợp chênh lệch ghép cặp.
- `src/methods/`: Bagging, AdaBoost SAMME và Stacking tự cài đặt.
- `src/utils.py`: bảng log và biểu đồ.
- `tests/`: kiểm tra cấu hình so sánh, phép tính chênh lệch và CLI.
