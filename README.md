# PKR‑Regression

Ultra‑light, transparent and noise‑proof regression library

---
## 🚀 Why PKR‑Regression?
| Pain Point | Classic ML | **PKR** |
|------------|-----------|---------|
| Hyper‑parameters | ❌ Yes | ✅ No (3‑4 flags) |
| Black‑box outputs | ❌ Yes | ✅ Rule‑based explanation |
| Sensitive to noise | ❌ | ✅ Middle zone muted by median |
| Training time | Long / tuning | Seconds |

In short: *“Keep the crystal‑clear signal, ignore the rest.”*

---
## 🔍 4‑Step Method
1. **Pole Labeling**  
   Sort target *y*.  
   * Bottom 33 % → label `0`  
   * Top 33 %   → label `1`  
2. **Cell Scanning**  
   Build tiny cells for every 1‑, 2‑, 3‑feature combo (fixed width / exact category).  
3. **Valuable Kernel Selection**  
   * ≥ 90 % same label  
   * ≥ 10 rows  
   ⇒ kernel value = mean of that cell.  
4. **Prediction**  
   * Test row hits ≥1 valuable kernel → average(kernel_values)  
   * Hits none → global median.

> Optional flags: `--bins 5` (five quantiles), `--max-dim 1/2/3`, `--gpu` for fast parallel scan.

---
## ⚡ Quick Start
```bash
pip install pkr-regression

pkr \
  --train train.csv \
  --test  test.csv \
  --target Listening_Time_minutes \
  --output submission.csv
```
Outputs: `submission.csv` (Kaggle‑ready) + `kernels.txt` (selected cells)

---
## 📊 Kernel Example
| Kernel Rule | Prediction |
|-------------|-----------|
| Podcast = *Comedy Corner* & Length 0‑0.5 | **6.4 min** |
| Podcast = *Innovators* & Length 0.5‑1.0 | **91.7 min** |

A test row intersecting the first rule returns **6.4**; the second rule → **91.7**.

---
## 💡 FAQ
**Q:** Overfitting risk?  
**A:** No – noisy cells (< 90 % majority or < 10 rows) are discarded automatically.

**Q:** Big data?  
**A:** 1M+ rows train in minutes on one core; `--gpu` accelerates further.

---
Make your model **explainable** with PKR‑Regression — try it & share feedback!


