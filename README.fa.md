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
aggregation انتخاب‌شده و به‌روزرسانی مدل سراسری
```

## ساختار پروژه

| مسیر                               | کاربرد                                                          |
| ---------------------------------- | --------------------------------------------------------------- |
| `models/cnn.py`                    | مدل CNN کوچک برای MNIST با ساختار مناسب برای هرس ساختاری.       |
| `utils/model_pruning.py`           | هرس غیرساختاری، هرس ساختاری و نسبت هرس adaptive برای هر کلاینت. |
| `utils/gradient_sparsification.py` | روش‌های Top-K، Random و Cost-Weighted همراه با error feedback.  |
| `client/client.py`                 | منطق کلاینت: هرس، آموزش محلی با پشتیبانی FedProx، محاسبه delta و sparse کردن آن. |
| `server/server.py`                 | منطق سرور: broadcast مدل، FedAvg/Krum/Trimmed Mean و ارزیابی.   |
| `run_simulation.py`                | اجرای کامل آزمایش و ذخیره خروجی‌ها.                             |
| `scripts/compare_ablations.py`     | اجرای خودکار baselineها، sweep FedProx و مقایسه hybrid در 50 round. |

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

اجرای FedProx خالص:

```bash
python run_simulation.py \
  --no_pruning \
  --no_sparsification \
  --no_adaptive_ratio \
  --baseline_fedprox_mu 0.01 \
  --run_name baseline_fedprox
```

اجرای Krum برای aggregation مقاوم:

```bash
python run_simulation.py \
  --no_pruning \
  --no_sparsification \
  --no_adaptive_ratio \
  --aggregation krum \
  --krum_f 1 \
  --clients_per_round 5 \
  --run_name baseline_krum
```

اجرای کامل baselineها و hybrid:

```bash
python scripts/compare_ablations.py
```

این اسکریپت FedAvg، FedProx، Krum، Trimmed Mean و روش hybrid را اجرا می‌کند.
برای Krum باید تعداد کلاینت‌های انتخابی حداقل `2 * f + 3` باشد؛ `f` همان
`--krum_f` است.

اسکریپت مقایسه اکنون به‌طور پیش‌فرض 50 round اجرا می‌شود، sweep
`μ=[0.001, 0.01, 0.1, 1.0]` برای FedProx انجام می‌دهد و بهترین μ را بر اساس
final accuracy برای نمودار اصلی انتخاب می‌کند. hybrid به‌طور پیش‌فرض 5 round
warm-up و سپس 5 round ramp دارد:

```bash
python scripts/compare_ablations.py \
  --num_rounds 50 \
  --dirichlet_alpha 0.3 \
  --warmup_rounds 5 \
  --compression_ramp_rounds 5
```

## Baselineهای استاندارد

- **FedProx:** ترم `mu / 2 * ||w_local - w_global||^2` را به loss محلی اضافه
  می‌کند و baseline استاندارد برای کاهش client drift در داده‌های non-IID است.
- **Krum:** deltaای را انتخاب می‌کند که به نزدیک‌ترین updateهای دیگر کمترین
  فاصله اقلیدسی را دارد؛ برای مقاومت در برابر کلاینت‌های outlier یا مخرب است.
- **Trimmed Mean:** برای هر مختصات پارامتر، درصدی از مقادیر کوچک و بزرگ را
  حذف و بقیه را میانگین می‌گیرد؛ بنابراین اثر updateهای ناهنجار کاهش می‌یابد.

هر سه aggregation با contribution maskهای هرس ساختاری سازگار هستند؛ هر
کلاینت فقط روی پارامترهایی اثر می‌گذارد که نگه داشته و آموزش داده است.

## پارتیشن non-IID و زمان‌بندی فشرده‌سازی

- **Dirichlet label skew:** روش پیش‌فرض `--partition_strategy dirichlet` است.
  مقدار `--dirichlet_alpha 0.3` ناهمگنی واقعی ایجاد می‌کند و مقدارهای مثبت
  کوچک‌تر non-IID شدیدتری دارند. برای روش shard قبلی از
  `--partition_strategy shard` استفاده کنید.
- **Warm-up و ramp:** در `--warmup_rounds` اولیه، pruning و sparsification
  غیرفعال‌اند. سپس `--compression_ramp_rounds` شدت آن‌ها را به‌صورت خطی تا
  مقدار هدف افزایش می‌دهد. مقدار پیش‌فرض هر دو `0` است و رفتار معمول قبلی را
  حفظ می‌کند.

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

متریک‌های کلیدی شامل `test_accuracy`، `avg_transmitted_ratio`,
`avg_pruning_ratio`، `avg_sparsity_ratio` (شدت مؤثر sparsification پس از
warm-up/ramp) و `avg_round_time_sec` هستند.

## آرگومان‌های مهم

| آرگومان               |         پیش‌فرض | توضیح                                       |
| --------------------- | --------------: | ------------------------------------------- |
| `--rounds`            |            `20` | تعداد roundهای فدرال.                       |
| `--num_clients`       |            `20` | تعداد کلاینت‌های شبیه‌سازی‌شده.             |
| `--clients_per_round` |            `10` | تعداد کلاینت‌های انتخاب‌شده در هر round.    |
| `--local_epochs`      |             `2` | تعداد epoch آموزش محلی.                     |
| `--partition_strategy` |     `dirichlet` | روش پارتیشن: `dirichlet` یا `shard` قبلی.   |
| `--dirichlet_alpha`  |           `0.3` | تمرکز Dirichlet؛ مقدار مثبت کوچک‌تر non-IID شدیدتر است. |
| `--min_samples_per_client` |       `10` | حداقل نمونه هر کلاینت در پارتیشن Dirichlet. |
| `--pruning_mode`      |    `structured` | نوع هرس: `structured` یا `unstructured`.    |
| `--pruning_ratio`     |           `0.4` | نسبت پایه هرس.                              |
| `--adaptive_ratio`    |            فعال | نسبت هرس جداگانه برای هر کلاینت.            |
| `--sparsify_method`   | `cost_weighted` | روش sparse کردن delta.                      |
| `--sparsity_ratio`    |          `0.95` | نسبت مقدارهایی که قبل از ارسال صفر می‌شوند. |
| `--warmup_rounds`     |             `0` | roundهای اولیه بدون pruning و sparsification. |
| `--compression_ramp_rounds` |        `0` | تعداد roundهای افزایش خطی فشرده‌سازی پس از warm-up. |
| `--baseline_fedprox_mu` |          `0.0` | ضریب proximal برای FedProx؛ صفر همان SGD عادی است. |
| `--aggregation`       |        `fedavg` | روش aggregation: `fedavg`، `krum` یا `trimmed_mean`. |
| `--krum_f`            |          `0` | تعداد فرضی کلاینت‌های مخرب/outlier برای Krum. |
| `--trim_ratio`        |          `0.0` | نسبت حذف‌شده از هر انتهای مقادیر در Trimmed Mean؛ کمتر از `0.5`. |
| `--results_dir`       |     `./results` | محل ذخیره خروجی‌ها.                         |
| `--run_name`          |       زمان اجرا | نام اختیاری پوشه خروجی.                     |

## نکته مهم

برای تولید نمودارها باید `matplotlib` نصب باشد. این وابستگی در `requirements.txt` اضافه شده است.

اسکریپت `scripts/compare_ablations.py` اجراهای جداگانه را در
`results/comparison/runs/` ذخیره می‌کند و دو نمودار 300 DPI مشترک را در
`results/comparison/` می‌سازد:

- `accuracy_comparison.png`
- `transmitted_ratio_comparison.png`
- `fedprox_mu_sweep.png`

همچنین `fedprox_mu_sweep.csv` و `analysis.md` در همین پوشه ذخیره می‌شوند.

ساختار خروجی مقایسه:

```text
results/comparison/
├── analysis.md
├── accuracy_comparison.png
├── fedprox_mu_sweep.csv
├── fedprox_mu_sweep.png
├── transmitted_ratio_comparison.png
└── runs/
    ├── baseline_fedavg/
    ├── baseline_krum/
    ├── baseline_trimmed_mean/
    ├── fedprox_mu_*/
    └── hybrid_structured_cwmp/
```
