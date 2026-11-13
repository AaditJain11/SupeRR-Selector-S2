import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

base = "/tmp/data_pack/Data Pack"
out_dir = "/working_dir/c_34ccf09f11a5f93d/cleaned_data"
fig_dir = "/working_dir/c_34ccf09f11a5f93d/artifacts/file_generation/ttl=63d/intermediate"
os.makedirs(out_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)

comp = pd.read_csv(f"{base}/competitions.csv")
teams = pd.read_csv(f"{base}/teams.csv")
matches = pd.read_csv(f"{base}/matches.csv")
p11 = pd.read_csv(f"{base}/playing_eleven.csv")
mapping = pd.read_csv(f"{base}/competition_team_player_mapping.csv")
bbb = pd.read_csv(f"{base}/ball_by_ball.csv", low_memory=False)

# --- 1. CLEAN COMPETITIONS & MATCHES ---
comp['comp_tier'] = np.where(
    comp['comp_name'].str.contains('IPL', case=False, na=False), 'IPL',
    np.where(comp['match_format'] == 'T20I', 'T20I', 'Other_T20')
)
comp.to_csv(f"{out_dir}/competitions_clean.csv", index=False)

matches['match_date'] = pd.to_datetime(matches['match_date'])
matches['man_of_the_match'] = matches['man_of_the_match'].replace(0, np.nan)
matches['is_dls'] = matches['revised_target'].notna() | matches['first_inn_revised_overs'].notna() | matches['second_inn_revised_overs'].notna()
matches['is_no_result'] = matches['result'].str.contains('Abandoned|No Result', case=False, na=False)
matches['is_tie'] = matches['result'].str.contains('Tie|Super Over', case=False, na=False)
matches['is_high_stakes'] = (matches['knock_out'] == 1) | matches['stage'].isin(['Final', 'Qualifier', 'Eliminator', 'Semi Final', 'Super 4'])
matches = matches.merge(comp[['comp_id', 'comp_name', 'match_format', 'season', 'comp_tier']], on='comp_id', how='left')
matches.to_csv(f"{out_dir}/matches_clean.csv", index=False)

# --- 2. CLEAN BALL-BY-BALL ---
# Fix dismissal_type typo
bbb['dismissal_type'] = bbb['dismissal_type'].replace({'Obstracting the field': 'Obstructing the field'})
# Fix retired hurt / non-wicket dismissed_player_id
bbb['is_retired_hurt'] = (~bbb['is_wicket']) & bbb['dismissed_player_id'].notna()
bbb.loc[bbb['is_retired_hurt'], 'dismissed_player_id'] = np.nan

# Bowl speed 0.0 -> NaN
bbb['bowl_speed_clean'] = bbb['bowl_speed'].replace(0.0, np.nan)

# Legal delivery & runs breakdown
bbb['is_wide'] = bbb['extras_type'] == 'wide'
bbb['is_noball'] = bbb['extras_type'] == 'Overstep'
bbb['is_legal'] = ~(bbb['is_wide'] | bbb['is_noball'])
bbb['total_runs'] = bbb['runs'] + bbb['extras1_runs'].fillna(0) + bbb['extras_2_runs'].fillna(0)
# Bowler conceded runs: bat runs + wides + no-balls (exclude byes, legbyes, penalty)
bbb['bowler_runs'] = bbb['runs'] + np.where(bbb['extras_type'].isin(['wide', 'Overstep']), bbb['extras1_runs'].fillna(0) + bbb['extras_2_runs'].fillna(0), 0)
# Bowler wicket: exclude RunOut, Obstructing the field, Retire Out
non_bowler_dismissals = {'RunOut', 'Obstructing the field', 'Retire Out'}
bbb['is_bowler_wicket'] = bbb['is_wicket'] & (~bbb['dismissal_type'].isin(non_bowler_dismissals))
bbb['is_dot_legal'] = bbb['is_legal'] & (bbb['total_runs'] == 0)
bbb['is_super_over'] = bbb['innings'].isin([3, 4])

# Phase
bbb['phase'] = np.where(bbb['overid'] <= 6, 'Powerplay (1-6)',
               np.where(bbb['overid'] <= 15, 'Middle (7-15)', 'Death (16-20)'))

# Canonical player names (most frequent name per ID)
striker_map = bbb.groupby('striker_id')['striker_name'].agg(lambda s: s.mode().iloc[0]).to_dict()
bowler_map = bbb.groupby('bowler_id')['bowler_name'].agg(lambda s: s.mode().iloc[0]).to_dict()
print("Striker IDs with multiple spellings:", (bbb.groupby('striker_id')['striker_name'].nunique() > 1).sum())
print("Bowler IDs with multiple spellings:", (bbb.groupby('bowler_id')['bowler_name'].nunique() > 1).sum())

# Merge match context
bbb = bbb.merge(matches[['match_id', 'comp_id', 'comp_name', 'comp_tier', 'match_format', 'season', 'match_date', 'stage', 'knock_out', 'is_high_stakes', 'is_dls', 'first_inn_revised_overs', 'second_inn_revised_overs', 'revised_target', 'win_team_id', 'venue_id', 'venue_country']], on='match_id', how='left')

# Sort chronologically within match & innings
bbb = bbb.sort_values(['match_id', 'innings', 'overid', 'innings_ball_num', 'over_ball_num']).reset_index(drop=True)

# Compute cumulative innings state BEFORE each delivery
grp = bbb.groupby(['match_id', 'innings'])
bbb['cum_runs_before'] = grp['total_runs'].cumsum() - bbb['total_runs']
bbb['cum_wkts_before'] = grp['is_wicket'].cumsum() - bbb['is_wicket'].astype(int)
bbb['legal_balls_before'] = grp['is_legal'].cumsum() - bbb['is_legal'].astype(int)

# Max overs per innings (handle DLS)
max_overs = np.where(
    (bbb['innings'] == 1) & bbb['first_inn_revised_overs'].notna(), bbb['first_inn_revised_overs'],
    np.where((bbb['innings'] == 2) & bbb['second_inn_revised_overs'].notna(), bbb['second_inn_revised_overs'], 20.0)
)
bbb['max_legal_balls'] = (max_overs * 6).astype(int)
bbb['balls_remaining'] = np.maximum(0, bbb['max_legal_balls'] - bbb['legal_balls_before'])
bbb['wickets_in_hand'] = np.maximum(0, 10 - bbb['cum_wkts_before'])

# Target for 2nd innings
inn1_totals = bbb[bbb['innings'] == 1].groupby('match_id')['total_runs'].sum().rename('inn1_total')
bbb = bbb.merge(inn1_totals, on='match_id', how='left')
bbb['target'] = np.where(
    bbb['innings'] == 2,
    np.where(bbb['revised_target'].notna(), bbb['revised_target'], bbb['inn1_total'] + 1),
    np.nan
)
bbb['runs_needed'] = np.where(bbb['innings'] == 2, np.maximum(0, bbb['target'] - bbb['cum_runs_before']), np.nan)
bbb['req_run_rate'] = np.where(
    (bbb['innings'] == 2) & (bbb['balls_remaining'] > 0),
    bbb['runs_needed'] / (bbb['balls_remaining'] / 6.0),
    np.where((bbb['innings'] == 2) & (bbb['runs_needed'] > 0), 36.0, 0.0)
)
bbb['curr_run_rate'] = np.where(
    bbb['legal_balls_before'] > 0,
    bbb['cum_runs_before'] / (bbb['legal_balls_before'] / 6.0),
    0.0
)

# Dot-ball streak before current ball (within innings)
def dot_streak(s):
    res = np.zeros(len(s), dtype=int)
    c = 0
    for i, val in enumerate(s):
        res[i] = c
        if val:
            c += 1
        else:
            c = 0
    return pd.Series(res, index=s.index)

# Vectorized dot streak before current ball
is_zero_bat = (bbb['total_runs'] == 0).astype(int)
# Groupby cumsum trick for consecutive zeros
block_id = (is_zero_bat == 0).groupby([bbb['match_id'], bbb['innings']]).cumsum()
streak_incl = is_zero_bat.groupby([bbb['match_id'], bbb['innings'], block_id]).cumsum()
bbb['dot_streak_before'] = grp_shift = streak_incl.groupby([bbb['match_id'], bbb['innings']]).shift(1).fillna(0).astype(int)

bbb.to_parquet(f"{out_dir}/ball_by_ball_clean.parquet", index=False)
p11.to_csv(f"{out_dir}/playing_eleven_clean.csv", index=False)
mapping.to_csv(f"{out_dir}/competition_team_player_mapping_clean.csv", index=False)
teams.to_csv(f"{out_dir}/teams_clean.csv", index=False)

# --- 3. ANALYST-LEVEL EDA ---
reg = bbb[~bbb['is_super_over']].copy()
print("\n=== EDA: TIER BREAKDOWN ===")
print(reg.groupby('comp_tier').agg(
    matches=('match_id', 'nunique'),
    balls=('is_legal', 'sum'),
    run_rate=('total_runs', lambda x: x.sum() / (reg.loc[x.index, 'is_legal'].sum() / 6)),
    boundary_pct=('runs', lambda x: ((reg.loc[x.index, 'is_four'] | reg.loc[x.index, 'is_six']).sum() / reg.loc[x.index, 'is_legal'].sum()) * 100),
    dot_pct=('is_dot_legal', lambda x: (x.sum() / reg.loc[x.index, 'is_legal'].sum()) * 100),
    wkts_per_match=('is_wicket', lambda x: x.sum() / reg.loc[x.index, 'match_id'].nunique())
).round(2))

print("\n=== EDA: PHASE & INNINGS BREAKDOWN ===")
phase_eda = reg.groupby(['innings', 'phase']).agg(
    legal_balls=('is_legal', 'sum'),
    runs=('total_runs', 'sum'),
    wkts=('is_wicket', 'sum'),
    fours=('is_four', 'sum'),
    sixes=('is_six', 'sum'),
    dots=('is_dot_legal', 'sum')
).reset_index()
phase_eda['run_rate'] = (phase_eda['runs'] / (phase_eda['legal_balls'] / 6)).round(2)
phase_eda['wicket_pct'] = (phase_eda['wkts'] / phase_eda['legal_balls'] * 100).round(2)
phase_eda['boundary_pct'] = ((phase_eda['fours'] + phase_eda['sixes']) / phase_eda['legal_balls'] * 100).round(2)
phase_eda['dot_pct'] = (phase_eda['dots'] / phase_eda['legal_balls'] * 100).round(2)
print(phase_eda[['innings', 'phase', 'run_rate', 'wicket_pct', 'boundary_pct', 'dot_pct']].to_string(index=False))

print("\n=== EDA: DOT-BALL STREAK PRESSURE EFFECT ===")
reg['streak_bucket'] = pd.cut(reg['dot_streak_before'], bins=[-1, 0, 1, 2, 100], labels=['0 dots', '1 dot', '2 dots', '3+ dots'])
streak_eda = reg[reg['is_legal']].groupby('streak_bucket', observed=False).agg(
    balls=('is_legal', 'count'),
    wicket_rate=('is_wicket', lambda x: x.mean() * 100),
    boundary_rate=('runs', lambda x: (reg.loc[x.index, 'is_four'] | reg.loc[x.index, 'is_six']).mean() * 100)
).round(2)
print(streak_eda)

print("\n=== EDA: 2ND INNINGS CHASE WIN RATE BY RRR BUCKET ===")
inn2 = reg[(reg['innings'] == 2) & reg['win_team_id'].notna()].copy()
inn2['chasing_won'] = (inn2['batting_team_id'] == inn2['win_team_id']).astype(int)
inn2['rrr_bucket'] = pd.cut(inn2['req_run_rate'], bins=[-1, 6, 8, 10, 12, 15, 100], labels=['<6', '6-8', '8-10', '10-12', '12-15', '15+'])
print(inn2.groupby('rrr_bucket', observed=False).agg(
    balls=('is_legal', 'count'),
    chase_win_pct=('chasing_won', lambda x: x.mean() * 100),
    wicket_pct=('is_wicket', lambda x: x.mean() * 100)
).round(2))

# --- 4. EDA CHARTS ---
plt.rcParams['font.size'] = 10
fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.3), constrained_layout=True)

# Plot 1: Phase Run Rate & Wicket % by Tier
tier_phase = reg.groupby(['comp_tier', 'phase']).agg(
    rr=('total_runs', lambda x: x.sum() / (reg.loc[x.index, 'is_legal'].sum() / 6)),
    wp=('is_wicket', lambda x: x.sum() / reg.loc[x.index, 'is_legal'].sum() * 100)
).reset_index()
phase_order = ['Powerplay (1-6)', 'Middle (7-15)', 'Death (16-20)']
tier_phase['phase'] = pd.Categorical(tier_phase['phase'], categories=phase_order, ordered=True)
tier_phase = tier_phase.sort_values('phase')

for tier, color in [('IPL', '#1e3a5f'), ('T20I', '#d97706')]:
    sub = tier_phase[tier_phase['comp_tier'] == tier]
    axes[0].plot(sub['phase'], sub['rr'], marker='o', linewidth=2.2, label=tier, color=color)
axes[0].set_title('Run Rate by Phase (IPL vs T20I)', fontweight='bold', fontsize=10.5)
axes[0].set_ylabel('Runs per Over')
axes[0].set_ylim(6, 12)
axes[0].tick_params(axis='x', rotation=15)
axes[0].legend(frameon=False)
axes[0].grid(axis='y', alpha=0.3)

# Plot 2: Dot-ball streak pressure on wicket rate
axes[1].bar(streak_eda.index.astype(str), streak_eda['wicket_rate'], color=['#94a3b8', '#64748b', '#d97706', '#b91c1c'])
axes[1].set_title('Wicket Rate by Prior Dot-Ball Streak', fontweight='bold', fontsize=10.5)
axes[1].set_ylabel('Wicket Probability (%)')
axes[1].grid(axis='y', alpha=0.3)
for i, v in enumerate(streak_eda['wicket_rate']):
    axes[1].text(i, v + 0.15, f"{v:.2f}%", ha='center', fontsize=9, fontweight='bold')

fig.savefig(f"{fig_dir}/eda_phase_and_pressure.png", dpi=220, bbox_inches='tight')
plt.close(fig)
print("\nSaved cleaned files and EDA chart successfully.")
