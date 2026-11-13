# SupeRR Selector Season 2 — Round 1: Identifying Reliable T20 Players Under Pressure

A ball-by-ball **Dual-Innings Neural Win-Probability Leverage** and **Empirical-Bayes** player evaluation system built for the **Rajasthan Royals SupeRR Selector Season 2 (Round 1)** analytics challenge.

---

## Problem Statement
Traditional T20 averages, strike rates, and economy rates treat all deliveries equally and conflate high-scoring leagues with slower international conditions. This project quantifies **situational match pressure** on every delivery and measures how batters and bowlers perform in high-pressure moments relative to their own normal-situation baselines and peer context expectations.

---

## Key Highlights & Model Performance
- **Clean Dataset:** Filtered 883 matches (200,999 deliveries) down to **778 clean IPL and T20I matches (185,926 deliveries)** from 2022–2026 by removing A-team/Emerging tournaments, DLS/rain-affected games, ties, no-results, and incomplete innings.
- **Dual Residual+Skip Neural Win-Probability Models (5-Fold Match-Grouped CV):**
  - **2nd Innings (`inn2_wp_net_5fold.pkl`):** LogLoss **0.3839** | Brier Score **0.1239** | ROC-AUC **0.9066**
  - **1st Innings (`inn1_wp_net_5fold.pkl`):** LogLoss **0.6321** | Brier Score **0.2210** | ROC-AUC **0.6953**
- **Context-Adjusted Baselines:** Empirical-Bayes expected runs (`xR_adj`, scaled by smoothed venue par run rate) and expected wickets (`xW`) across `(innings, over_bucket, tier3, wkts_bucket, pace_or_spin)` using an empirical T20 wicket value ($\lambda = 14.0$ runs).
- **Own-Baseline Shrinkage:** High-pressure ball values are shrunk ($m = 150$ balls) toward each player's own normal-situation prior rather than the league average, separating genuine pressure elevation from small-sample noise.

---

## Methodology Pipeline

1. **Data Hygiene & Context Tagging (`src/01_clean_and_eda.py`)**
   - Standardizes dismissal labels (`Obstracting the field`), converts `bowl_speed = 0.0` to `NaN`, isolates retired-hurt events, and separates bowler-credited wickets from run-outs/obstructions.
   - Excludes wides and no-balls from legal delivery counts and reconstructs pre-delivery runs, wickets in hand, balls remaining, required run rate (RRR), and consecutive dot-ball streaks.
   - Separates competition tiers into **IPL** (354 matches), **Full-Member T20Is** (374 matches), and **Associate T20Is** (50 matches).

2. **6-Outcome Counterfactual Leverage (`src/02_hybrid_pipeline.py`)**
   - Evaluates six simulated next-ball outcomes (`dot, 1, 2, 4, 6, wicket`) through the 5-fold Neural WP ensembles to compute probability-weighted **Expected Leverage**, scaled by a `1.15×` multiplier for Knockouts, Finals, Qualifiers, Eliminators, and Semi-Finals.
   - Flags deliveries in the **top 25% of within-innings leverage** where the contest remains alive ($0.03 \le \text{WP} \le 0.97$) as **High-Pressure Balls** (46,437 balls; 25.0%), compared against Normal-Pressure balls (102,318 balls; 55.0%).

3. **Three-Pillar Composite Rating**
   - **Eligibility:** $\ge 180$ high-pressure balls (batters) / $\ge 160$ (bowlers), $\ge 12$ high-pressure matches, and $\ge 450$ total balls (78 batters and 89 bowlers qualify).
   - **Score Formula:**
     $$\text{Score} = 0.55 \cdot z(\text{Shrunk HP Impact}) + 0.25 \cdot z(\text{Pressure Lift}) + 0.20 \cdot z(\text{Shrunk Consistency})$$

---

## Final Top 10 Rankings

### Top 10 Batters Under Pressure
| Rank | Batter | HP Matches | HP Balls | HP Runs | HP SR | Norm SR | HP Avg | Pos Match % | Score |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Vaibhav Sooryavanshi** | 25 | 204 | 421 | 206.4 | 219.9 | 70.2 | 86% | **2.34** |
| **2** | **Virat Kohli** | 76 | 582 | 885 | 152.1 | 143.1 | 63.2 | 85% | **1.70** |
| **3** | **Rajat Patidar** | 24 | 233 | 437 | 187.6 | 163.6 | 62.4 | 58% | **1.62** |
| **4** | **Abhishek Sharma** | 106 | 680 | 1219 | 179.3 | 189.3 | 42.0 | 71% | **1.55** |
| **5** | **Heinrich Klaasen** | 36 | 326 | 574 | 176.1 | 163.4 | 47.8 | 73% | **1.45** |
| **6** | **Shubman Gill** | 88 | 690 | 1058 | 153.3 | 155.4 | 50.4 | 67% | **1.22** |
| **7** | **Andrew Balbirnie** | 25 | 212 | 284 | 134.0 | 115.9 | 40.6 | 71% | **1.08** |
| **8** | **Reeza Hendricks** | 29 | 245 | 333 | 135.9 | 155.8 | 66.6 | 74% | **1.04** |
| **9** | **Daryl Mitchell** | 34 | 243 | 373 | 153.5 | 141.3 | 41.4 | 61% | **1.04** |
| **10** | **Hardik Pandya** | 60 | 474 | 669 | 141.1 | 144.5 | 47.8 | 60% | **1.03** |

### Top 10 Bowlers Under Pressure
| Rank | Bowler | HP Matches | HP Balls | HP Wkts | HP Econ | Norm Econ | HP SR | Pos Match % | Score |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Josh Hazlewood** | 50 | 407 | 34 | 7.86 | 8.60 | 12.0 | 58% | **2.10** |
| **2** | **Taskin Ahmed** | 37 | 308 | 24 | 6.55 | 9.12 | 12.8 | 50% | **1.84** |
| **3** | **Ravindra Jadeja** | 35 | 257 | 15 | 6.65 | 8.16 | 17.1 | 68% | **1.59** |
| **4** | **Shaheen Afridi** | 41 | 379 | 31 | 7.22 | 8.53 | 12.2 | 50% | **1.41** |
| **5** | **Axar Patel** | 81 | 591 | 34 | 6.88 | 8.49 | 17.4 | 52% | **1.32** |
| **6** | **Mohammad Nawaz** | 32 | 223 | 18 | 7.21 | 7.13 | 12.4 | 61% | **1.30** |
| **7** | **Kuldeep Yadav** | 51 | 400 | 31 | 7.78 | 8.09 | 12.9 | 54% | **1.29** |
| **8** | **Trent Boult** | 65 | 569 | 31 | 7.72 | 9.33 | 18.4 | 49% | **1.15** |
| **9** | **Varun Chakravarthy** | 59 | 510 | 35 | 7.24 | 8.23 | 14.6 | 52% | **1.09** |
| **10** | **Glenn Maxwell** | 33 | 163 | 10 | 7.44 | 8.94 | 16.3 | 61% | **1.07** |

---

## Repository Structure
```text
SupeRR-Selector-S2/
├── models/
│   ├── inn1_wp_net_5fold.pkl        # 5-fold 1st-innings ResidualWPNet weights + scalers
│   └── inn2_wp_net_5fold.pkl        # 5-fold 2nd-innings ResidualWPNet weights + scalers
├── outputs/
│   ├── figures/
│   │   ├── rr_calibration_phase.png # Dual Neural WP calibration & phase share
│   │   ├── rr_top10_batters.png     # Top 10 Batters HP vs. Normal SR
│   │   └── rr_top10_bowlers.png     # Top 10 Bowlers HP vs. Normal Economy
│   ├── hybrid_batter_rankings.csv   # Full ranked list of 78 eligible batters
│   └── hybrid_bowler_rankings.csv   # Full ranked list of 89 eligible bowlers
├── src/
│   ├── 01_clean_and_eda.py          # Data cleaning & exploratory analysis
│   ├── 02_hybrid_pipeline.py        # Neural WP training, leverage & rankings
│   └── 03_generate_rr_report.py     # Rajasthan Royals-themed PDF report generator
├── requirements.txt
└── README.md
```

---

## Quick Start

```powershell
pip install -r requirements.txt

# 1. Clean raw CSVs and run EDA
python src/01_clean_and_eda.py

# 2. Train 5-fold Neural WP models, compute leverage, and generate rankings
python src/02_hybrid_pipeline.py

# 3. Generate Rajasthan Royals-themed charts and PDF report
python src/03_generate_rr_report.py
```

---

**Author:** [Aadit Jain](https://www.google.com/search?q=https://github.com/AaditJain11)
