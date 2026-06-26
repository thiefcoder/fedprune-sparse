# FedPrune-Sparse

نسخه انگلیسی اصلی در `README.md` قرار دارد. این فایل نسخه فارسی همراه پروژه است.
زبان: فارسی  | [EN](README.md)

## هدف پروژه

FedPrune-Sparse یک شبیه‌سازی یادگیری فدرال برای بررسی ترکیب دو ایده است:

- **هرس مدل در سمت کلاینت** برای کاهش هزینه آموزش محلی.
- **پراکنده‌سازی دلتا قبل از ارسال به سرور** برای کاهش هزینه ارتباطی.

کد برای آزمایش‌های پایان‌نامه/رساله آماده شده است: هر اجرا تنظیمات، لاگ دقیق انگلیسی، متریک‌های هر round، وزن مدل نهایی و نمودارهای 300 DPI را در پوشه `results/` ذخیره می‌کند.

## روش کلی هر Round

```text
مدل سراسری از سرور
        │
        ▼
کپی مدل در کلاینت
        │
        ▼
هرس اختیاری مدل
        │
        ▼
آموزش محلی روی داده non-IID
        │
        ▼
محاسبه delta = وزن محلی - وزن سراسری
        │
        ▼
پراکنده‌سازی اختیاری delta با error feedback
        │
        ▼
ارسال delta فشرده‌شده به سرور
        │
        ▼
FedAvg و به‌روزرسانی مدل سراسری
```

## ساختار پروژه

| مسیر                               | کاربرد                                                          |
| ---------------------------------- | --------------------------------------------------------------- |
| `models/cnn.py`                    | مدل CNN کوچک برای MNIST با ساختار مناسب برای هرس ساختاری.       |
| `utils/model_pruning.py`           | هرس غیرساختاری، هرس ساختاری و نسبت هرس adaptive برای هر کلاینت. |
| `utils/gradient_sparsification.py` | روش‌های Top-K، Random و Cost-Weighted همراه با error feedback.  |
| `client/client.py`                 | منطق کلاینت: هرس، آموزش محلی، محاسبه delta و sparse کردن آن.    |
| `server/server.py`                 | منطق سرور: broadcast مدل، FedAvg و ارزیابی.                     |
| `run_simulation.py`                | اجرای کامل آزمایش و ذخیره خروجی‌ها.                             |
| `scripts/compare_ablations.py`     | اجرای خودکار چند آزمایش ablation.                               |

## نصب

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

اگر از GPU استفاده می‌کنید، بهتر است نسخه PyTorch سازگار با CUDA سیستم را مطابق راهنمای رسمی PyTorch نصب کنید.

## اجرای سریع

اجرای حالت ترکیبی پیش‌فرض:

```bash
python run_simulation.py
```

اجرای سبک برای تست سریع:

```bash
python run_simulation.py \
  --rounds 2 \
  --num_clients 4 \
  --clients_per_round 2 \
  --local_epochs 1 \
  --run_name smoke_test
```

اجرای baseline ساده FedAvg:

```bash
python run_simulation.py \
  --no_pruning \
  --no_sparsification \
  --no_adaptive_ratio \
  --run_name baseline_fedavg
```

اجرای ablation کامل:

```bash
python scripts/compare_ablations.py
```

## خروجی‌ها

هر اجرا یک زیرپوشه داخل `results/` می‌سازد:

```text
results/<run_name>/
├── config.json
├── metrics.csv
├── run.log
├── summary.json
├── final_model.pt
├── accuracy_curve.png
├── compression_pruning_curve.png
├── round_time_curve.png
└── experiment_dashboard.png
```

| فایل             | توضیح                                                   |
| ---------------- | ------------------------------------------------------- |
| `config.json`    | همه تنظیمات اجرای همان آزمایش.                          |
| `metrics.csv`    | متریک‌های round به round برای تحلیل آماری.              |
| `run.log`        | لاگ دقیق انگلیسی شامل انتخاب کلاینت‌ها و آمار هر round. |
| `summary.json`   | خلاصه نهایی، بهترین accuracy و مسیر artifactها.         |
| `final_model.pt` | وزن مدل سراسری بعد از آخرین round.                      |
| `*.png`          | نمودارهای ذخیره‌شده برای گزارش و رساله.                 |

## آرگومان‌های مهم

| آرگومان               |         پیش‌فرض | توضیح                                       |
| --------------------- | --------------: | ------------------------------------------- |
| `--rounds`            |            `20` | تعداد roundهای فدرال.                       |
| `--num_clients`       |            `20` | تعداد کلاینت‌های شبیه‌سازی‌شده.             |
| `--clients_per_round` |            `10` | تعداد کلاینت‌های انتخاب‌شده در هر round.    |
| `--local_epochs`      |             `2` | تعداد epoch آموزش محلی.                     |
| `--pruning_mode`      |    `structured` | نوع هرس: `structured` یا `unstructured`.    |
| `--pruning_ratio`     |           `0.4` | نسبت پایه هرس.                              |
| `--adaptive_ratio`    |            فعال | نسبت هرس جداگانه برای هر کلاینت.            |
| `--sparsify_method`   | `cost_weighted` | روش sparse کردن delta.                      |
| `--sparsity_ratio`    |          `0.95` | نسبت مقدارهایی که قبل از ارسال صفر می‌شوند. |
| `--results_dir`       |     `./results` | محل ذخیره خروجی‌ها.                         |
| `--run_name`          |       زمان اجرا | نام اختیاری پوشه خروجی.                     |

## نکته مهم

برای تولید نمودارها باید `matplotlib` نصب باشد. این وابستگی در `requirements.txt` اضافه شده است.
