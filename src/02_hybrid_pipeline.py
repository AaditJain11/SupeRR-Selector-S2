import os
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import spearmanr

np.random.seed(42)
out_dir = "/working_dir/c_34ccf09f11a5f93d/cleaned_data"
bbb = pd.read_parquet(f"{out_dir}/ball_by_ball_clean.parquet")
matches = pd.read_csv(f"{out_dir}/matches_clean.csv")

# --- 1. CLEAN FILTERING (784-match style + IPL / T20I_Full / T20I_Assoc) ---
full_members = {
    'India', 'Australia', 'England', 'Pakistan', 'New Zealand',
    'South Africa', 'West Indies', 'Sri Lanka', 'Bangladesh',
    'Afghanistan', 'Zimbabwe', 'Ireland'
}
m_clean = matches[
    (matches['comp_tier'] != 'Other_T20') &
    (~matches['is_dls']) &
    (~matches['is_no_result']) &
    (~matches['is_tie']) &
    (matches['win_team_id'].notna())
].copy()

# Tag T20I_Full vs T20I_Assoc
def get_tier3(row):
    if row['comp_tier'] == 'IPL':
        return 'IPL'
    if row['home_team'] in full_members and row['away_team'] in full_members:
        return 'T20I_Full'
    return 'T20I_Assoc'

m_clean['tier3'] = m_clean.apply(get_tier3, axis=1)

reg = bbb[(~bbb['is_super_over']) & (bbb['match_id'].isin(m_clean['match_id']))].copy()
# Require complete 1st innings (>= 114 balls or 10 wkts) and existing 2nd innings
inn1_chk = reg[reg['innings'] == 1].groupby('match_id').agg(b=('is_legal', 'sum'), w=('is_wicket', 'sum'))
inn2_chk = reg[reg['innings'] == 2].groupby('match_id').agg(b=('is_legal', 'sum'))
valid_ids = inn1_chk[(inn1_chk['b'] >= 114) | (inn1_chk['w'] >= 10)].index.intersection(inn2_chk[inn2_chk['b'] >= 6].index)

reg = reg[reg['match_id'].isin(valid_ids)].copy().reset_index(drop=True)
reg = reg.merge(m_clean[['match_id', 'tier3']], on='match_id', how='left')
print(f"Hybrid Clean Sample: {reg['match_id'].nunique()} matches | {len(reg)} deliveries")
print("Tier3 match counts:\n", reg.groupby('tier3')['match_id'].nunique())

# Venue par RR (Empirical Bayes smoothed)
inn1_m = reg[reg['innings'] == 1].groupby(['match_id', 'venue_id', 'tier3']).agg(
    r=('total_runs', 'sum'), b=('is_legal', 'sum')
).reset_index()
inn1_m['rr'] = inn1_m['r'] / np.maximum(1, inn1_m['b'] / 6.0)
tier_rr = inn1_m.groupby('tier3')['rr'].mean().to_dict()
v_stats = inn1_m.groupby('venue_id')['rr'].agg(['count', 'mean']).to_dict('index')
reg['venue_par_rr'] = [
    (v_stats[v]['count'] * v_stats[v]['mean'] + 10.0 * tier_rr[t]) / (v_stats[v]['count'] + 10.0)
    if v in v_stats else tier_rr[t]
    for v, t in zip(reg['venue_id'], reg['tier3'])
]
reg['venue_par_total'] = reg['venue_par_rr'] * 20.0
reg['bat_won'] = (reg['batting_team_id'] == reg['win_team_id']).astype(np.float64)

# --- 2. NEURAL NETWORK CLASS ---
class ResidualWPNet:
    def __init__(self, in_dim, h_dim=48, lr=0.002, wd=5e-3):
        self.lr = lr
        self.wd = wd
        self.params = {
            'W1': np.random.randn(in_dim, h_dim) * np.sqrt(2.0 / in_dim), 'b1': np.zeros(h_dim),
            'W2': np.random.randn(h_dim, h_dim) * np.sqrt(2.0 / h_dim), 'b2': np.zeros(h_dim),
            'W3': np.random.randn(h_dim, 1) * np.sqrt(2.0 / h_dim),      'b3': np.zeros(1),
            'W_skip': np.random.randn(in_dim, 1) * 0.01
        }
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.t = 0

    def forward(self, X):
        a1 = np.tanh(np.clip(X @ self.params['W1'] + self.params['b1'], -15, 15))
        z2 = a1 @ self.params['W2'] + self.params['b2']
        a2 = a1 + np.tanh(np.clip(z2, -15, 15))
        logits = (a2 @ self.params['W3'] + self.params['b3'] + X @ self.params['W_skip']).ravel()
        probs = expit(np.clip(logits, -20, 20))
        return probs, (X, a1, z2, a2, probs)

    def step(self, X, y):
        B = len(y)
        probs, (X, a1, z2, a2, _) = self.forward(X)
        dlogits = (probs - y) / B
        grads = {
            'W_skip': X.T @ dlogits[:, None] + self.wd * self.params['W_skip'],
            'W3': a2.T @ dlogits[:, None] + self.wd * self.params['W3'],
            'b3': dlogits.sum(keepdims=True)
        }
        da2 = dlogits[:, None] @ self.params['W3'].T
        dz2 = da2 * (1 - np.tanh(np.clip(z2, -15, 15)) ** 2)
        grads['W2'] = a1.T @ dz2 + self.wd * self.params['W2']
        grads['b2'] = dz2.sum(axis=0)
        da1 = da2 + dz2 @ self.params['W2'].T
        dz1 = da1 * (1 - a1 ** 2)
        grads['W1'] = X.T @ dz1 + self.wd * self.params['W1']
        grads['b1'] = dz1.sum(axis=0)
        self.t += 1
        for k in self.params:
            self.m[k] = 0.9 * self.m[k] + 0.1 * grads[k]
            self.v[k] = 0.999 * self.v[k] + 0.001 * (grads[k] ** 2)
            mh = self.m[k] / (1 - 0.9 ** self.t)
            vh = self.v[k] / (1 - 0.999 ** self.t)
            self.params[k] -= self.lr * mh / (np.sqrt(vh) + 1e-8)

    def predict(self, X):
        return self.forward(X)[0]

def train_oof_nn(df, feat_fn, y_col='bat_won', epochs=25):
    X = feat_fn(df)
    y = df[y_col].values
    mids = df['match_id'].values
    um = np.unique(mids)
    np.random.shuffle(um)
    fmap = {m: i % 5 for i, m in enumerate(um)}
    folds = np.array([fmap[m] for m in mids])
    oof = np.zeros(len(y))
    models = []
    for f in range(5):
        tr, va = folds != f, folds == f
        mu, sd = X[tr].mean(axis=0), X[tr].std(axis=0) + 1e-6
        X_tr, X_va = (X[tr] - mu) / sd, (X[va] - mu) / sd
        net = ResidualWPNet(X.shape[1], h_dim=48, lr=0.002, wd=5e-3)
        idx = np.arange(tr.sum())
        best_ll, best_p, best_par = 999.0, None, None
        for ep in range(epochs):
            net.lr = 0.002 * 0.5 * (1 + np.cos(np.pi * ep / epochs))
            np.random.shuffle(idx)
            for s in range(0, len(idx), 2048):
                b = idx[s:s+2048]
                net.step(X_tr[b], y[tr][b])
            p_va = net.predict(X_va)
            ll = -np.mean(y[va] * np.log(p_va + 1e-9) + (1 - y[va]) * np.log(1 - p_va + 1e-9))
            if ll < best_ll:
                best_ll, best_p, best_par = ll, p_va.copy(), {k: v.copy() for k, v in net.params.items()}
        net.params = best_par
        oof[va] = best_p
        models.append((net, mu, sd))
    brier = np.mean((y - oof) ** 2)
    ll = -np.mean(y * np.log(oof + 1e-9) + (1 - y) * np.log(1 - oof + 1e-9))
    order = np.argsort(oof)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(oof) + 1)
    n1 = y.sum()
    auc = (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))
    return oof, models, (ll, brier, auc)

def feat_inn1(df):
    br = df['balls_remaining'].values.astype(float)
    wh = df['wickets_in_hand'].values.astype(float)
    runs = df['cum_runs_before'].values.astype(float)
    crr = np.clip(df['curr_run_rate'].values, 0, 25)
    vpar = df['venue_par_total'].values.astype(float)
    return np.column_stack([
        br / 120.0, wh / 10.0, runs / 200.0, crr / 12.0,
        (runs + crr * (br / 6.0) - vpar) / 40.0,
        (br / 120.0) * (wh / 10.0),
        (df['tier3'] == 'IPL').astype(float),
        (df['tier3'] == 'T20I_Full').astype(float),
        df['is_high_stakes'].astype(float),
        np.minimum(df['dot_streak_before'].values, 6) / 6.0
    ])

def feat_inn2(df):
    br = df['balls_remaining'].values.astype(float)
    wh = df['wickets_in_hand'].values.astype(float)
    rn = df['runs_needed'].values.astype(float)
    rrr = np.clip(df['req_run_rate'].values, 0, 36)
    crr = np.clip(df['curr_run_rate'].values, 0, 25)
    tgt = df['target'].values.astype(float)
    vpar = df['venue_par_total'].values.astype(float)
    return np.column_stack([
        br / 120.0, wh / 10.0, rn / 200.0, rrr / 12.0, crr / 12.0,
        (rrr - crr) / 10.0, (br / 120.0) * (wh / 10.0),
        np.log1p(rn) / np.log1p(np.maximum(1.0, br)),
        (tgt - vpar) / 40.0,
        (df['tier3'] == 'IPL').astype(float),
        (df['tier3'] == 'T20I_Full').astype(float),
        df['is_high_stakes'].astype(float),
        np.minimum(df['dot_streak_before'].values, 6) / 6.0,
        (wh <= 3).astype(float)
    ])

m1 = reg['innings'] == 1
m2 = reg['innings'] == 2
df1 = reg[m1].copy().reset_index(drop=True)
df2 = reg[m2].copy().reset_index(drop=True)

oof1, mod1, met1 = train_oof_nn(df1, feat_inn1, epochs=20)
oof2, mod2, met2 = train_oof_nn(df2, feat_inn2, epochs=28)
print(f"Inn1 Neural WP: LogLoss={met1[0]:.4f} | Brier={met1[1]:.4f} | AUC={met1[2]:.4f}")
print(f"Inn2 Neural WP: LogLoss={met2[0]:.4f} | Brier={met2[1]:.4f} | AUC={met2[2]:.4f}")

# --- 3. 6-OUTCOME COUNTERFACTUAL LEVERAGE FOR BOTH INNINGS ---
def eval_ens(models, X):
    p = np.zeros(len(X))
    for net, mu, sd in models:
        p += net.predict((X - mu) / sd)
    return p / len(models)

# Empirical outcome probs: dot (~0.36), 1 (~0.36), 2 (~0.06), 4 (~0.115), 6 (~0.06), W (~0.045)
outcomes = [(0, 0, 0.36), (1, 0, 0.36), (2, 0, 0.06), (4, 0, 0.115), (6, 0, 0.06), (0, 1, 0.045)]

def compute_leverage(df, wp_base, models, feat_fn, is_inn2=False):
    lev = np.zeros(len(df))
    for r_add, w_add, prob in outcomes:
        d = df.copy()
        d['balls_remaining'] = np.maximum(0, d['balls_remaining'] - 1)
        d['legal_balls_before'] = d['legal_balls_before'] + 1
        d['cum_runs_before'] = d['cum_runs_before'] + r_add
        d['wickets_in_hand'] = np.maximum(0, d['wickets_in_hand'] - w_add)
        d['curr_run_rate'] = d['cum_runs_before'] / (np.maximum(1, d['legal_balls_before']) / 6.0)
        d['dot_streak_before'] = np.where(r_add == 0, d['dot_streak_before'] + 1, 0)
        if is_inn2:
            d['runs_needed'] = np.maximum(0, d['runs_needed'] - r_add)
            d['req_run_rate'] = np.where(d['balls_remaining'] > 0, d['runs_needed'] / (d['balls_remaining'] / 6.0), np.where(d['runs_needed'] > 0, 36.0, 0.0))
        wp_next = eval_ens(models, feat_fn(d))
        lev += prob * np.abs(wp_next - wp_base)
    return lev

lev1 = compute_leverage(df1, oof1, mod1, feat_inn1, is_inn2=False)
lev2 = compute_leverage(df2, oof2, mod2, feat_inn2, is_inn2=True)

reg['wp'] = np.nan
reg['leverage'] = np.nan
reg.loc[m1, 'wp'] = oof1
reg.loc[m2, 'wp'] = oof2
reg.loc[m1, 'leverage'] = lev1 * np.where(df1['is_high_stakes'], 1.15, 1.0)
reg.loc[m2, 'leverage'] = lev2 * np.where(df2['is_high_stakes'], 1.15, 1.0)

# Rank-normalize within each innings, exclude dead rubbers (WP < 0.03 or WP > 0.97)
reg['pressure_pct'] = reg.groupby('innings')['leverage'].rank(pct=True)
reg['is_hp'] = (reg['pressure_pct'] >= 0.75) & (reg['wp'].between(0.03, 0.97))
reg['is_norm'] = (~reg['is_hp']) & (reg['pressure_pct'] <= 0.55) | (~reg['wp'].between(0.03, 0.97))
print("HP balls:", reg['is_hp'].sum(), f"({reg['is_hp'].mean()*100:.1f}%) | Normal balls:", reg['is_norm'].sum(), f"({reg['is_norm'].mean()*100:.1f}%)")

# --- 4. CONTEXT BASELINES (xR_adj, xW) & BALL VALUE ---
reg['over_bucket'] = pd.cut(reg['overid'], bins=[0, 3, 6, 10, 15, 18, 20], labels=['1-3', '4-6', '7-10', '11-15', '16-18', '19-20'])
reg['wkts_bucket'] = pd.cut(reg['wickets_in_hand'], bins=[-1, 3, 6, 10], labels=['1-3', '4-6', '7-10'])

t3_xr = reg.groupby('tier3')['total_runs'].mean().to_dict()
t3_xw = reg.groupby('tier3')['is_wicket'].mean().to_dict()
grp_cols = ['innings', 'over_bucket', 'tier3', 'wkts_bucket', 'pace_or_spin']
cs = reg.groupby(grp_cols, observed=False).agg(n=('total_runs', 'count'), sr=('total_runs', 'sum'), sw=('is_wicket', 'sum')).reset_index()
cs['xR'] = (cs['sr'] + 50 * cs['tier3'].map(t3_xr)) / (cs['n'] + 50)
cs['xW'] = (cs['sw'] + 50 * cs['tier3'].map(t3_xw)) / (cs['n'] + 50)
reg = reg.merge(cs[grp_cols + ['xR', 'xW']], on=grp_cols, how='left')
vf = np.clip(reg['venue_par_rr'] / reg['tier3'].map(tier_rr), 0.85, 1.15)
reg['xR_adj'] = reg['xR'] * vf

WICKET_VAL = 14.0
reg['bv_bat'] = (reg['runs'] - reg['xR_adj']) - WICKET_VAL * (reg['is_wicket'].astype(float) - reg['xW'])
reg['bv_bowl'] = (reg['xR_adj'] - reg['bowler_runs']) + WICKET_VAL * (reg['is_bowler_wicket'].astype(float) - reg['xW'])

# --- 5. OWN-BASELINE EMPIRICAL BAYES SHRINKAGE & RANKING ---
def rank_hybrid(df, id_col, name_col, val_col, runs_col, wkt_col, min_hp_balls, min_tot_balls=450, min_hp_matches=12):
    sub = df[~df['is_wide']].copy() if id_col == 'striker_id' else df.copy()
    sub['ball_cnt'] = sub['is_legal'].astype(int)

    tot = sub.groupby([id_col, name_col]).agg(
        tot_balls=('ball_cnt', 'sum'), tot_matches=('match_id', 'nunique')
    ).reset_index()

    norm = sub[sub['is_norm']].groupby(id_col).agg(
        norm_balls=('ball_cnt', 'sum'), norm_bv=(val_col, 'mean'),
        norm_runs=(runs_col, 'sum'), norm_wkts=(wkt_col, 'sum')
    ).reset_index()

    hp_sub = sub[sub['is_hp']].copy()
    hp = hp_sub.groupby(id_col).agg(
        hp_balls=('ball_cnt', 'sum'), hp_matches=('match_id', 'nunique'),
        hp_runs=(runs_col, 'sum'), hp_wkts=(wkt_col, 'sum'),
        hp_bv=(val_col, 'mean')
    ).reset_index()

    m_hp = hp_sub.groupby([id_col, 'match_id']).agg(m_bv=(val_col, 'sum'), m_b=('ball_cnt', 'sum')).reset_index()
    m_hp = m_hp[m_hp['m_b'] >= 3]
    m_cons = m_hp.groupby(id_col).agg(
        pos_rate=('m_bv', lambda x: (x > 0).mean()),
        m_n=('m_bv', 'count')
    ).reset_index()
    # Shrink consistency toward 0.50 with 6 pseudo-innings
    m_cons['cons_shrunk'] = (m_cons['m_n'] * m_cons['pos_rate'] + 6 * 0.50) / (m_cons['m_n'] + 6)

    elig = tot.merge(norm, on=id_col).merge(hp, on=id_col).merge(m_cons, on=id_col)
    elig = elig[(elig['hp_balls'] >= min_hp_balls) & (elig['tot_balls'] >= min_tot_balls) & (elig['hp_matches'] >= min_hp_matches)].copy()

    # Fit linear prior hp_bv ~ slope * norm_bv + intercept across eligible pool
    slope, intercept = np.polyfit(elig['norm_bv'], elig['hp_bv'], 1)
    elig['prior_bv'] = intercept + slope * elig['norm_bv']
    m_shrink = 150.0
    elig['hp_bv_shrunk'] = (elig['hp_balls'] * elig['hp_bv'] + m_shrink * elig['prior_bv']) / (elig['hp_balls'] + m_shrink)
    elig['pressure_lift'] = elig['hp_bv_shrunk'] - elig['norm_bv']

    def z(s): return (s - s.mean()) / (s.std() + 1e-9)
    elig['pressure_score'] = 0.55 * z(elig['hp_bv_shrunk']) + 0.25 * z(elig['pressure_lift']) + 0.20 * z(elig['cons_shrunk'])

    if id_col == 'striker_id':
        elig['hp_sr'] = (elig['hp_runs'] / elig['hp_balls'] * 100).round(1)
        elig['norm_sr'] = (elig['norm_runs'] / elig['norm_balls'] * 100).round(1)
        elig['hp_avg'] = (elig['hp_runs'] / np.maximum(1, elig['hp_wkts'])).round(1)
    else:
        elig['hp_econ'] = (elig['hp_runs'] / (elig['hp_balls'] / 6.0)).round(2)
        elig['norm_econ'] = (elig['norm_runs'] / (elig['norm_balls'] / 6.0)).round(2)
        elig['hp_sr'] = (elig['hp_balls'] / np.maximum(1, elig['hp_wkts'])).round(1)

    elig = elig.sort_values('pressure_score', ascending=False).reset_index(drop=True)
    elig['rank'] = elig.index + 1
    return elig, slope

bat_h, bat_slope = rank_hybrid(reg, 'striker_id', 'striker_name', 'bv_bat', 'runs', 'is_wicket', min_hp_balls=180)
bowl_h, bowl_slope = rank_hybrid(reg, 'bowler_id', 'bowler_name', 'bv_bowl', 'bowler_runs', 'is_bowler_wicket', min_hp_balls=160)

print(f"\nOwn-baseline prior slopes: Batters={bat_slope:.2f}, Bowlers={bowl_slope:.2f}")
print(f"Eligible Batters: {len(bat_h)} | Eligible Bowlers: {len(bowl_h)}")
print("\n=== HYBRID TOP 10 BATTERS ===")
print(bat_h[['rank', 'striker_name', 'hp_matches', 'hp_balls', 'hp_runs', 'hp_sr', 'norm_sr', 'hp_avg', 'pos_rate', 'pressure_score']].head(10).round(3).to_string(index=False))
print("\n=== HYBRID TOP 10 BOWLERS ===")
print(bowl_h[['rank', 'bowler_name', 'hp_matches', 'hp_balls', 'hp_wkts', 'hp_econ', 'norm_econ', 'hp_sr', 'pos_rate', 'pressure_score']].head(10).round(3).to_string(index=False))

bat_h.to_csv(f"{out_dir}/hybrid_batter_rankings.csv", index=False)
bowl_h.to_csv(f"{out_dir}/hybrid_bowler_rankings.csv", index=False)
reg.to_parquet(f"{out_dir}/hybrid_ball_by_ball_scored.parquet", index=False)

# --- 6. SAVE TRAINED 5-FOLD MODEL WEIGHTS & SCALERS ---
import pickle
model_dir = "/tmp/repo_bundle/models"
os.makedirs(model_dir, exist_ok=True)
with open(f"{model_dir}/inn1_wp_net_5fold.pkl", "wb") as f:
    pickle.dump([{'params': net.params, 'mu': mu, 'sd': sd} for net, mu, sd in mod1], f)
with open(f"{model_dir}/inn2_wp_net_5fold.pkl", "wb") as f:
    pickle.dump([{'params': net.params, 'mu': mu, 'sd': sd} for net, mu, sd in mod2], f)
print("Saved trained 5-fold model checkpoints to models/")
