import os
import html
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import weasyprint
import pypdfium2 as pdfium

out_dir = "/working_dir/c_34ccf09f11a5f93d/cleaned_data"
inter_dir = "/working_dir/c_34ccf09f11a5f93d/artifacts/file_generation/ttl=63d/intermediate"
pdf_out = "/working_dir/c_34ccf09f11a5f93d/artifacts/file_generation/ttl=63d/output/SupeRR_Selector_RR_Submission_Report.pdf"

reg = pd.read_parquet(f"{out_dir}/hybrid_ball_by_ball_scored.parquet")
bat = pd.read_csv(f"{out_dir}/hybrid_batter_rankings.csv")
bowl = pd.read_csv(f"{out_dir}/hybrid_bowler_rankings.csv")

RR_PINK = "#EA1A85"
RR_PLUM = "#172B75"
RR_GOLD = "#254AA5"
RR_MUTED = "#6B7280"
RR_LIGHT = "#F9A8D4"

plt.rcParams['font.size'] = 10

# --- 1. CHARTS IN RAJASTHAN ROYALS PALETTE ---
# Chart 1: Calibration + High-Pressure Phase Share
fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.1), constrained_layout=True)
for inn, lbl, col in [(1, '1st Inn. Neural WP (Brier 0.221)', RR_GOLD), (2, '2nd Inn. Neural WP (Brier 0.124)', RR_PINK)]:
    sub = reg[reg['innings'] == inn].copy()
    sub['b'] = pd.cut(sub['wp'], bins=np.linspace(0, 1, 11))
    cal = sub.groupby('b', observed=False).agg(p=('wp', 'mean'), o=('bat_won', 'mean')).dropna()
    axes[0].plot(cal['p'], cal['o'], 'o-', color=col, linewidth=2.2, markersize=4.5, label=lbl)
axes[0].plot([0, 1], [0, 1], '--', color='#9CA3AF', linewidth=1.2)
axes[0].set_title('Dual Neural WP Calibration', fontweight='bold', fontsize=10.5, color=RR_PLUM)
axes[0].set_xlabel('Predicted Win Probability')
axes[0].set_ylabel('Observed Win Rate')
axes[0].legend(frameon=False, fontsize=8, loc='upper left')
axes[0].grid(alpha=0.25)

phase_order = ['Powerplay (1-6)', 'Middle (7-15)', 'Death (16-20)']
hp_sh = reg.groupby('phase')['is_hp'].mean().reindex(phase_order) * 100
axes[1].bar(['Powerplay\n(1-6)', 'Middle\n(7-15)', 'Death\n(16-20)'], hp_sh.values, color=[RR_MUTED, RR_PLUM, RR_PINK])
axes[1].set_title('High-Pressure Ball Share by Phase', fontweight='bold', fontsize=10.5, color=RR_PLUM)
axes[1].set_ylabel('High-Pressure Share (%)')
axes[1].set_ylim(0, 65)
axes[1].grid(axis='y', alpha=0.25)
for i, v in enumerate(hp_sh.values):
    axes[1].text(i, v + 1.5, f"{v:.1f}%", ha='center', fontsize=9, fontweight='bold', color=RR_PLUM)
fig.savefig(f"{inter_dir}/rr_calibration_phase.png", dpi=220, bbox_inches='tight')
plt.close(fig)

# Chart 2: Top 10 Batters
b10 = bat.head(10).iloc[::-1].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(6.8, 3.7), constrained_layout=True)
y = np.arange(len(b10))
ax.hlines(y=y, xmin=b10['norm_sr'], xmax=b10['hp_sr'], color='#E5E7EB', linewidth=3.2, zorder=1)
ax.scatter(b10['norm_sr'], y, color=RR_MUTED, s=65, label='Normal-Pressure SR', zorder=2)
ax.scatter(b10['hp_sr'], y, color=RR_PINK, s=85, label='High-Pressure SR', zorder=3)
ax.set_yticks(y)
ax.set_yticklabels([f"#{int(r['rank'])} {r['striker_name']}" for _, r in b10.iterrows()], fontsize=9.5, color=RR_PLUM)
ax.set_xlabel('Strike Rate (Runs per 100 Balls)')
ax.set_title('Top 10 Batters: High-Pressure vs. Normal Strike Rate', fontweight='bold', fontsize=10.5, color=RR_PLUM, pad=22)
ax.legend(frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=9)
ax.grid(axis='x', alpha=0.25)
for i, r in b10.iterrows():
    ax.text(max(r['hp_sr'], r['norm_sr']) + 2.5, i, f"Score: {r['pressure_score']:.2f}", va='center', fontsize=8.5, color=RR_PLUM, fontweight='bold')
ax.set_xlim(108, 245)
fig.savefig(f"{inter_dir}/rr_top10_batters.png", dpi=220, bbox_inches='tight')
plt.close(fig)

# Chart 3: Top 10 Bowlers
w10 = bowl.head(10).iloc[::-1].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(6.8, 3.7), constrained_layout=True)
y = np.arange(len(w10))
ax.hlines(y=y, xmin=w10['hp_econ'], xmax=w10['norm_econ'], color='#E5E7EB', linewidth=3.2, zorder=1)
ax.scatter(w10['norm_econ'], y, color=RR_MUTED, s=65, label='Normal-Pressure Economy', zorder=2)
ax.scatter(w10['hp_econ'], y, color=RR_PLUM, s=85, label='High-Pressure Economy', zorder=3)
ax.set_yticks(y)
ax.set_yticklabels([f"#{int(r['rank'])} {r['bowler_name']}" for _, r in w10.iterrows()], fontsize=9.5, color=RR_PLUM)
ax.set_xlabel('Economy Rate (Runs Conceded per Over — Lower is Better)')
ax.set_title('Top 10 Bowlers: High-Pressure vs. Normal Economy', fontweight='bold', fontsize=10.5, color=RR_PLUM, pad=22)
ax.legend(frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=9)
ax.grid(axis='x', alpha=0.25)
for i, r in w10.iterrows():
    ax.text(max(r['hp_econ'], r['norm_econ']) + 0.15, i, f"Wkts: {int(r['hp_wkts'])} | {r['pressure_score']:.2f}", va='center', fontsize=8.5, color=RR_PINK, fontweight='bold')
ax.set_xlim(6.0, 10.6)
fig.savefig(f"{inter_dir}/rr_top10_bowlers.png", dpi=220, bbox_inches='tight')
plt.close(fig)

# --- 2. BUILD HTML IN RAJASTHAN ROYALS THEME (NO REFERENCES, APPLICANT -> RR) ---
bat_rows = "".join([
    f"""<tr><td class="num"><strong>{int(r['rank'])}</strong></td><td><strong>{html.escape(r['striker_name'])}</strong></td><td class="num">{int(r['hp_matches'])}</td><td class="num">{int(r['hp_balls'])}</td><td class="num">{int(r['hp_runs'])}</td><td class="num">{r['hp_sr']:.1f}</td><td class="num">{r['norm_sr']:.1f}</td><td class="num">{r['hp_avg']:.1f}</td><td class="num">{r['pos_rate']*100:.0f}%</td><td class="num"><strong>{r['pressure_score']:.2f}</strong></td></tr>"""
    for _, r in bat.head(10).iterrows()
])
bowl_rows = "".join([
    f"""<tr><td class="num"><strong>{int(r['rank'])}</strong></td><td><strong>{html.escape(r['bowler_name'])}</strong></td><td class="num">{int(r['hp_matches'])}</td><td class="num">{int(r['hp_balls'])}</td><td class="num">{int(r['hp_wkts'])}</td><td class="num">{r['hp_econ']:.2f}</td><td class="num">{r['norm_econ']:.2f}</td><td class="num">{r['hp_sr']:.1f}</td><td class="num">{r['pos_rate']*100:.0f}%</td><td class="num"><strong>{r['pressure_score']:.2f}</strong></td></tr>"""
    for _, r in bowl.head(10).iterrows()
])

html_doc = rf"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <style>
    :root {{
      --rr-pink: #EA1A85;
      --rr-plum: #172B75;
      --rr-gold: #254AA5;
      --surface: #FDF2F8;
      --border:  #FBCFE8;
      --text:    #1F2937;
      --muted:   #4B5563;
      --bg:      #FFFFFF;
    }}
    @page {{
      size: A4;
      margin: 18mm 18mm;
      background: var(--bg);
      @bottom-left {{
        content: "Rajasthan Royals | SupeRR Selector Season 2 — Round 1";
        font-family: 'Inter', sans-serif;
        font-size: 8.5pt;
        color: var(--muted);
      }}
      @bottom-right {{
        content: "Page " counter(page) " of " counter(pages);
        font-family: 'Inter', sans-serif;
        font-size: 8.5pt;
        color: var(--muted);
      }}
    }}
    @page :first {{
      @bottom-left {{ content: none; }}
      @bottom-right {{ content: none; }}
    }}
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      font-size: 10.8pt;
      line-height: 1.55;
      color: var(--text);
      margin: 0;
      padding: 0;
    }}
    .header-card {{
      background: linear-gradient(135deg, #172B75 0%, #254AA5 45%, #EA1A85 100%);
      color: #FFFFFF;
      padding: 20px 24px;
      border-radius: 8px;
      margin-bottom: 14px;
    }}
    .header-kicker {{
      font-size: 9pt;
      font-weight: 700;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: #FBCFE8;
      margin-bottom: 6px;
    }}
    .header-card h1 {{
      font-size: 20pt;
      font-weight: 800;
      line-height: 1.2;
      margin: 0 0 6px 0;
      color: #FFFFFF;
    }}
    .header-sub {{
      font-size: 10.5pt;
      color: #F3E8FF;
      margin: 0 0 12px 0;
    }}
    .meta-strip {{
      display: flex;
      justify-content: space-between;
      font-size: 9.5pt;
      border-top: 1px solid rgba(255,255,255,0.22);
      padding-top: 9px;
      color: #FCE7F3;
    }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 10px;
      margin: 12px 0 16px 0;
      break-inside: avoid;
    }}
    .kpi-box {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-top: 3px solid var(--rr-pink);
      border-radius: 6px;
      padding: 9px 10px;
      text-align: center;
    }}
    .kpi-val {{
      font-size: 14pt;
      font-weight: 800;
      color: var(--rr-plum);
      line-height: 1.2;
    }}
    .kpi-lbl {{
      font-size: 8.5pt;
      color: var(--muted);
      margin-top: 2px;
    }}
    h2 {{
      font-size: 13.5pt;
      font-weight: 700;
      margin: 20px 0 8px 0;
      padding-bottom: 4px;
      border-bottom: 2px solid var(--rr-pink);
      color: var(--rr-plum);
      break-after: avoid;
    }}
    p {{ margin: 0 0 8px 0; }}
    ul {{ margin: 0 0 8px 0; padding-inline-start: 20px; }}
    li {{ margin-bottom: 4px; }}
    .callout {{
      background: var(--surface);
      border-inline-start: 4px solid var(--rr-pink);
      border-radius: 6px;
      padding: 11px 14px;
      margin: 12px 0;
      break-inside: avoid;
    }}
    .callout-title {{
      font-weight: 700;
      color: var(--rr-plum);
      margin-bottom: 3px;
      font-size: 10.5pt;
    }}
    table {{ width: 100%; border-collapse: collapse; margin: 10px 0 12px 0; }}
    th, td {{ text-align: start; padding: 6.5px 7px; font-size: 9.8pt; vertical-align: top; }}
    th.num, td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    th {{ background: var(--rr-plum); font-weight: 700; color: #FFFFFF; }}
    td {{ border-bottom: 1px solid var(--border); }}
    tbody tr:nth-child(even) {{ background: var(--surface); }}
    tr {{ break-inside: avoid; }}
    figure {{ margin: 10px 0; break-inside: avoid; }}
    figure img {{ width: 100%; height: auto; display: block; border-radius: 4px; }}
    figcaption {{ font-size: 9pt; color: var(--muted); margin-top: 5px; text-align: center; }}
  </style>
</head>
<body>
<main class="document-flow">

  <div class="header-card">
    <div class="header-kicker">Rajasthan Royals &bull; SupeRR Selector Season 2 &bull; Round 1</div>
    <h1>Identifying the Most Reliable T20 Players Under Pressure</h1>
    <p class="header-sub">Dual-Innings Neural Win-Probability Leverage &amp; Own-Baseline Empirical-Bayes Framework</p>
    <div class="meta-strip">
      <span><strong>To:</strong> Rajasthan Royals Analytics &amp; Selection Panel</span>
      <span><strong>Submitted by:</strong> Aadit Jain</span>
      <span><strong>Dataset:</strong> IPL &amp; T20I (2022&ndash;2026)</span>
    </div>
  </div>

  <div class="kpi-grid">
    <div class="kpi-box"><div class="kpi-val">778</div><div class="kpi-lbl">Clean Matches Analyzed</div></div>
    <div class="kpi-box"><div class="kpi-val">185,926</div><div class="kpi-lbl">Deliveries Modeled</div></div>
    <div class="kpi-box"><div class="kpi-val">0.9066</div><div class="kpi-lbl">2nd-Inn. Neural WP AUC</div></div>
    <div class="kpi-box"><div class="kpi-val">78 / 89</div><div class="kpi-lbl">Eligible Batters / Bowlers</div></div>
  </div>

  <div class="callout">
    <div class="callout-title">Executive Summary</div>
    To identify players who consistently deliver when the match is on the line, this system trains separate 5-fold match-grouped Residual+Skip Neural Networks for both innings to quantify ball-by-ball <strong>Win Probability Leverage</strong>. Every delivery is benchmarked against venue- and competition-adjusted expected runs and wickets, shrunk toward each player's own normal-situation baseline, and scored across three pillars: <strong>High-Pressure Impact (55%)</strong>, <strong>Pressure Lift (25%)</strong>, and <strong>Match Consistency (20%)</strong>.
  </div>

  <h2>1. Data Cleaning &amp; Context Filtering</h2>
  <ul>
    <li><strong>Match Hygiene:</strong> Starting from 883 matches in <code>matches.csv</code>, 867 contained ball-by-ball records. To ensure fair opposition quality and clean win/loss targets, 30 Emerging/A-team tournament matches, DLS/rain-affected games, ties, and no-results were excluded, and complete 1st innings (&ge;114 legal balls or 10 wickets) with an active 2nd innings were required—yielding <strong>778 matches</strong> (354 IPL, 374 Full-Member T20Is, 50 Associate T20Is).</li>
    <li><strong>Delivery Hygiene:</strong> Standardized dismissal labels (<code>Obstracting the field</code>), converted <code>bowl_speed=0.0</code> to <code>NaN</code>, isolated retired-hurt events, separated bowler-credited wickets from run-outs/obstructions, excluded wides/no-balls from legal ball counts, and reconstructed pre-delivery runs, wickets in hand, balls remaining, required run rate (RRR), and consecutive dot-ball streaks.</li>
  </ul>

  <h2>2. Defining Pressure: Dual Neural Win Probability &amp; Leverage</h2>
  <p>In T20 cricket, pressure is the <strong>sensitivity of the match outcome to the next delivery</strong>. Separate 5-fold match-grouped Residual+Skip Neural Networks were trained for the 1st innings (Brier <strong>0.2210</strong>, ROC-AUC <strong>0.6953</strong>) and 2nd innings (Brier <strong>0.1239</strong>, ROC-AUC <strong>0.9066</strong>) using balls remaining, wickets in hand, run rates, <strong>Empirical-Bayes venue par run rate</strong>, competition tier (IPL / Full T20I / Associate T20I), <strong>knockout stakes (1.15&times; multiplier)</strong>, and <strong>dot-ball streaks</strong>.</p>
  <p>For every delivery, six counterfactual next-ball outcomes (dot, 1, 2, 4, 6, wicket) are simulated to compute probability-weighted <strong>Expected Leverage</strong>. Deliveries in the top 25% of within-innings leverage where the match remains alive (<code>0.03 &le; WP &le; 0.97</code>) are classified as <strong>High-Pressure Balls</strong> (46,437 balls; 25.0%), compared against Normal-Pressure balls (102,318 balls; 55.0%).</p>

  <figure>
    <img src="rr_calibration_phase.png" alt="Dual Neural WP Calibration and High-Pressure Phase Share">
    <figcaption>Figure 1: Out-of-fold calibration for 1st- and 2nd-innings Neural WP models (left) and high-pressure ball share by phase (right).</figcaption>
  </figure>

  <h2>3. Performance Measurement, Shrinkage &amp; Scoring</h2>
  <p>To avoid penalizing players who operate in tougher phases or lower-scoring T20Is (8.54 RPO vs. 9.30 RPO in IPL), every ball is evaluated against context-expected runs (<code>xR_adj</code>, scaled by venue par) and expected wickets (<code>xW</code>) across <code>(innings, over_bucket, tier3, wkts_bucket, pace_or_spin)</code> using the empirical T20 wicket value (&lambda; = 14.0 runs).</p>
  <ul>
    <li><strong>Eligibility:</strong> Minimum 180 high-pressure balls (batters) / 160 (bowlers), 12 high-pressure matches, and 450 total balls (78 batters and 89 bowlers qualify).</li>
    <li><strong>Own-Baseline Shrinkage:</strong> Each player's high-pressure value is shrunk (<code>m = 150</code> balls) toward their own normal-situation baseline rather than the league mean, isolating true pressure elevation from sample noise.</li>
    <li><strong>Composite Score:</strong> <code>0.55 &times; z(Shrunk HP Impact) + 0.25 &times; z(Pressure Lift) + 0.20 &times; z(Shrunk Consistency)</code>.</li>
  </ul>

  <h2>4. Top 10 Batters Under Pressure</h2>
  <figure>
    <img src="rr_top10_batters.png" alt="Top 10 Batters">
    <figcaption>Figure 2: Top 10 Batters — High-Pressure vs. Normal-Pressure Strike Rate.</figcaption>
  </figure>
  <table>
    <thead><tr><th class="num">#</th><th>Batter</th><th class="num">HP M</th><th class="num">HP Balls</th><th class="num">HP Runs</th><th class="num">HP SR</th><th class="num">Norm SR</th><th class="num">HP Avg</th><th class="num">Pos%</th><th class="num">Score</th></tr></thead>
    <tbody>{bat_rows}</tbody>
  </table>

  <h2>5. Top 10 Bowlers Under Pressure</h2>
  <figure>
    <img src="rr_top10_bowlers.png" alt="Top 10 Bowlers">
    <figcaption>Figure 3: Top 10 Bowlers — High-Pressure vs. Normal-Pressure Economy Rate.</figcaption>
  </figure>
  <table>
    <thead><tr><th class="num">#</th><th>Bowler</th><th class="num">HP M</th><th class="num">HP Balls</th><th class="num">HP Wkts</th><th class="num">HP Econ</th><th class="num">Norm Econ</th><th class="num">HP SR</th><th class="num">Pos%</th><th class="num">Score</th></tr></thead>
    <tbody>{bowl_rows}</tbody>
  </table>

  <h2>6. Reliability, Consistency &amp; Limitations</h2>
  <ul>
    <li><strong>Out-of-Fold Validation:</strong> Strict match-grouped 5-fold CV prevents intra-match leakage, yielding <strong>0.1239 Brier / 0.9066 AUC</strong> (2nd innings) and <strong>0.2210 Brier / 0.6953 AUC</strong> (1st innings).</li>
    <li><strong>Sample-Size &amp; Bootstrap Stability:</strong> Across 200 match-level bootstrap resamples, high-volume anchors like <strong>Virat Kohli (582 HP balls, 85% positive matches)</strong>, <strong>Abhishek Sharma (680 HP balls)</strong>, <strong>Josh Hazlewood (407 HP balls)</strong>, and <strong>Axar Patel (591 HP balls)</strong> remain highly stable, whereas newer/lower-volume entrants like <strong>Vaibhav Sooryavanshi (204 HP balls)</strong> and <strong>Glenn Maxwell (163 HP balls)</strong> carry wider confidence intervals.</li>
    <li><strong>Model Boundaries:</strong> No ball-tracking pitch-map or fielder-quality data is included, and 1st-innings win probability inherently carries higher uncertainty than 2nd-innings chases because the target is not yet set.</li>
  </ul>

</main>
</body>
</html>"""

html_path = f"{inter_dir}/rr_document.html"
with open(html_path, "w", encoding="utf-8") as f:
    f.write(html_doc)

weasyprint.HTML(filename=html_path, base_url=inter_dir + "/").write_pdf(pdf_out)
pdf = pdfium.PdfDocument(pdf_out)
print("RR PDF pages:", len(pdf))
for i in range(len(pdf)):
    p = os.path.join(inter_dir, f"rr_page_{i+1}.png")
    pdf[i].render(scale=2).to_pil().save(p)
    print("Rendered:", p)
pdf.close()
