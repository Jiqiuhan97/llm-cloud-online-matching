"""
Synthetic Dataset Generation
============================
Generates four families of stress-test datasets from a baseline of 159 demanders
and 68 suppliers:

  - Scale Expansion    (Expanded_10x, 100x, 1000x)
  - Resource Scarcity  (Scarcity_75, 50, 25, 10)
  - Demand Backlog     (Backlog_25, 50, 75)
  - Semantic Conflict  (Conflict_15, 30, 50)

Each strategy perturbs only its target dimension; all other parameters are
strictly preserved (verified by two-sample KS tests in the paper).

Usage:
  python generate_datasets.py
"""

import pandas as pd
import numpy as np
import os
import re

BASE = os.path.dirname(os.path.abspath(__file__))
SEED = 42


# ============================================================
# 1. Scale Expansion — bootstrap resampling + small Gaussian noise
# ============================================================
def generate_scale(df_d, df_s, multiplier):
    """Bootstrap with replacement, preserving empirical distributions."""
    rng = np.random.default_rng(SEED)

    df_d_out = df_d.sample(frac=multiplier, replace=True, random_state=SEED).copy()
    df_s_out = df_s.sample(frac=multiplier, replace=True, random_state=SEED).copy()

    n_d, n_s = len(df_d_out), len(df_s_out)

    # Small Gaussian jitter to avoid exact duplicates
    df_d_out['lon_i'] += rng.normal(0, 0.005, n_d)
    df_d_out['lat_i'] += rng.normal(0, 0.005, n_d)
    df_d_out['t_i']   = np.clip(df_d_out['t_i'] + rng.normal(0, 2.0, n_d), 0.01, None)
    df_d_out['Ex_q,i'] = np.clip(df_d_out['Ex_q,i'] + rng.normal(0, 0.5, n_d), 0.01, None)

    df_s_out['lon_j'] += rng.normal(0, 0.005, n_s)
    df_s_out['lat_j'] += rng.normal(0, 0.005, n_s)
    df_s_out['t_j']   = np.clip(df_s_out['t_j'] + rng.normal(0, 2.0, n_s), 0.01, None)

    df_d_out = df_d_out.sort_values('t_i').reset_index(drop=True)
    df_s_out = df_s_out.sort_values('t_j').reset_index(drop=True)
    return df_d_out, df_s_out


# ============================================================
# 2. Resource Scarcity — linearly scale supply by lambda
# ============================================================
def generate_scarcity(df_s, lam):
    """Reduce supply quantity Y_jq to lam * original, rounded up."""
    df_s_out = df_s.copy()
    df_s_out['Y_jq'] = np.ceil(df_s_out['Y_jq'] * lam).clip(1).astype(int)
    return df_s_out


# ============================================================
# 3. Demand Backlog — truncate arrival times, release in later window
# ============================================================
def generate_backlog(df_d, ratio):
    """
    Randomly select ratio of demanders and redraw their t_i
    from a delayed release window U(0.7*t_max, 1.3*t_max).
    """
    rng = np.random.default_rng(SEED)
    df_d_out = df_d.copy()
    n_bl = int(len(df_d_out) * ratio)
    bl_idx = rng.choice(df_d_out.index, size=n_bl, replace=False)

    t_max = df_d_out['t_i'].max()
    new_t = rng.uniform(t_max * 0.7, t_max * 1.3, size=n_bl)
    df_d_out.loc[bl_idx, 't_i'] = new_t
    df_d_out = df_d_out.sort_values('t_i').reset_index(drop=True)
    return df_d_out


# ============================================================
# 4. Semantic Conflict — text-level injection only
# ============================================================
# --- Conflict text templates ---

CONFLICT_FALLBACK_A = (
    "Floodwaters have surged to the THIRD FLOOR!! The entire area has been "
    "without power and water for over 72 hours. All roads are destroyed. "
    "URGENTLY need helicopter airdrop of 500 cases of water and 1000 cases "
    "of instant noodles!!"
)

CONFLICT_PAIRS_B = [
    (
        "UPDATE: I have been safely evacuated to Luoyang by high-speed rail. "
        "I am completely fine now. However, ",
        "my elderly relatives are still trapped on the first floor of my home "
        "in Zhengzhou. Water has reached chest level. PLEASE send a rescue boat "
        "there immediately!!"
    ),
    (
        "Thanks to the rescue team, my entire family has been safely relocated "
        "to a shelter in Xinxiang. But ",
        "there are still hundreds of people trapped in the community where I was "
        "stuck. 500 cases of supplies need to be delivered urgently!!"
    ),
    (
        "Good news: the water in our community has mostly receded. Yet ",
        "the entire building's power and communications are completely down. "
        "Elderly and children are trapped on upper floors. Rescue ladders and "
        "stretchers are desperately needed!!"
    ),
    (
        "Just received notice that flood diversion is complete and water levels "
        "will not rise further. However, ",
        "our entire community's first floor is submerged, the only entrance has "
        "collapsed, and many people inside need immediate rescue by boat!!"
    ),
    (
        "A government rescue helicopter has airlifted my family out. We are now "
        "in a safe zone. Yet ",
        "the first floor of my home in Zhengzhou has been underwater for three "
        "days, and neighbors on the second floor are still trapped. Please help "
        "contact a rescue boat!!"
    ),
    (
        "The community broadcast says the most dangerous phase has passed. But ",
        "our building entrance is blocked by cars swept in by the flood, the "
        "entire first floor is flooded, and a paralyzed elderly person is trapped "
        "inside. Emergency workers cannot get in!!"
    ),
]

CONFLICT_TEXTS_C = [
    (
        "#Zhengzhou 720 Catastrophic Rainstorm# EMERGENCY!! Floodwaters have "
        "risen to the THIRD FLOOR!! The entire community is destroyed, hundreds "
        "are trapped on rooftops awaiting helicopter rescue!! URGENTLY NEED 500 "
        "cases of instant noodles and 1000 cases of bottled water!!"
    ),
    (
        "#Henan Flood Rescue# EXTREME EMERGENCY!! Our community is completely "
        "surrounded by floodwaters. Water has reached the third floor, multiple "
        "buildings have collapsed!! Trapped for over 48 hours, elderly and "
        "children are barely holding on!! Desperately need rescue boats and "
        "helicopters!!"
    ),
    (
        "#Zhengzhou Flood Mutual Aid# HELP!! Our entire building is encircled "
        "by floodwaters. Floors 1 through 3 are completely submerged, all "
        "community roads are destroyed!! No power or water for over 72 hours, "
        "over 200 people trapped!! Request emergency airdrop of food and water!!"
    ),
    (
        "#Zhengzhou Flood SOS# CRITICAL!! The entire street has been swallowed "
        "by floodwaters. Water is rising at half a meter per hour and has "
        "already surpassed all historical records!! We are stranded on the "
        "rooftop awaiting rescue. Urgently need helicopter squadrons and 5000 "
        "cases of relief supplies!!"
    ),
    (
        "#Henan Disaster Relief# SOS!! This area has had no power or water for "
        "five days. Floodwater has reached third-floor windows and all "
        "communications are about to collapse!! Send rescue boats immediately. "
        "Over 300 people are trapped across multiple buildings!!"
    ),
]

# --- Regex rules for Type A (quantitative exaggeration) ---

TYPE_A_RULES = [
    (r'an?kle[-\s]?(deep|level|high)', 'THIRD-FLOOR WINDOWS'),
    (r'knee[-\s]?(deep|level|high)', 'THIRD-FLOOR BALCONIES'),
    (r'(thigh|waist|chest)[-\s]?(deep|level|high)', 'ABOVE THE ROOFTOP'),
    (r'(seeping|leaking|trickling|creeping)\s*(in|into|through)',
     'VIOLENT FLOODWATER GUSHING in'),
    (r'(standing water|water\s+pooling)', 'ENTIRE AREA SUBMERGED'),
    (r'(just\s+)?starting\s+to\s+get\s+bad',
     'already in CATASTROPHIC condition for over 72 hours'),
    (r'(early|initial)\s+stages', 'CATASTROPHIC end-stage'),
    (r'(\d+)\s*cases?', lambda m: f'{int(m.group(1))*30} cases'),
    (r'(\d+)\s*bottles?', lambda m: f'{int(m.group(1))*50} cases'),
    (r'(\d+)\s*people', lambda m: f'{int(m.group(1))*20} people'),
    (r'(need|urgently need).{0,30}(some water|drinking water|water to drink)',
     'URGENTLY NEED 500 CASES of bottled water and 1000 CASES of instant noodles'),
    (r'(need|urgently need).{0,30}(some food|something to eat|food supplies)',
     'URGENTLY NEED 2000 CASES of ready-to-eat meals'),
    (r'a few hours', 'OVER 72 HOURS'),
    (r'(one day|all night|overnight)', 'FIVE FULL DAYS AND NIGHTS'),
    (r'(situation|condition|disaster).{0,20}(severe|critical|dangerous|terrible)',
     'DISASTER HAS REACHED ANNIHILATION LEVEL'),
]


def _gen_conflict_text(text, ctype):
    """Apply text-level conflict rewriting. Cloud parameters are NOT modified."""
    rng = np.random.default_rng(hash(text) % 2**32)

    if ctype == 'A_QuantExag':
        result = text
        changed = False
        for pat, repl in TYPE_A_RULES:
            new_r = re.sub(pat, repl, result, flags=re.IGNORECASE)
            if new_r != result:
                changed = True
                result = new_r
        return result if changed else CONFLICT_FALLBACK_A

    elif ctype == 'B_LogicContra':
        opening, closing = rng.choice(CONFLICT_PAIRS_B)
        return f"{opening} {text} {closing}"

    elif ctype == 'C_FalseInfo':
        return rng.choice(CONFLICT_TEXTS_C)

    return text


def generate_conflict(df_d, ratio):
    """
    Stratified random selection: 3 risk levels x 3 POI types = 9 strata.
    Within each stratum, ceil(ratio * n) demanders are randomly selected.
    Conflict types A/B/C are assigned in round-robin order.
    Only the text column (text_i) is modified; cloud parameters are untouched.
    """
    rng = np.random.default_rng(SEED)
    df_out = df_d.copy()

    # Build strata
    risk_bins = [0, 40, 70, 100]
    risk_labels = ['Low', 'Med', 'High']
    df_out['risk_lvl'] = pd.cut(df_out['Ex_r,i'], bins=risk_bins,
                                labels=risk_labels, right=False)

    poi_map = {1: 'Medical', 2: 'Community', 3: 'Individual'}
    df_out['poi'] = df_out['unit_i'].map(poi_map)
    df_out['stratum'] = df_out['risk_lvl'].astype(str) + '_' + df_out['poi']

    conflict_types = ['A_QuantExag', 'B_LogicContra', 'C_FalseInfo']

    for stratum in df_out['stratum'].unique():
        s_idx = df_out[df_out['stratum'] == stratum].index.tolist()
        n_sel = max(1, int(np.ceil(len(s_idx) * ratio)))
        selected = sorted(rng.choice(s_idx, size=min(n_sel, len(s_idx)),
                                     replace=False))

        for k, row_idx in enumerate(selected):
            ctype = conflict_types[k % 3]
            df_out.at[row_idx, 'text_i'] = _gen_conflict_text(
                df_out.at[row_idx, 'text_i'], ctype
            )

    # Remove helper columns
    df_out = df_out.drop(columns=['risk_lvl', 'poi', 'stratum'])
    return df_out


# ============================================================
# 5. Batch generation
# ============================================================
if __name__ == '__main__':
    df_d = pd.read_csv(os.path.join(BASE, 'demand.csv'))
    df_s = pd.read_csv(os.path.join(BASE, 'supply.csv'))

    # --- Scale Expansion ---
    for m in [10, 100, 1000]:
        dd, ds = generate_scale(df_d, df_s, m)
        dd.to_csv(os.path.join(BASE, f'demand_Scale{m}x.csv'), index=False)
        ds.to_csv(os.path.join(BASE, f'supply_Scale{m}x.csv'), index=False)
        print(f'Scale {m}x: demand {len(dd)} rows, supply {len(ds)} rows')

    # --- Resource Scarcity ---
    for lam in [0.75, 0.50, 0.25, 0.10]:
        ds = generate_scarcity(df_s, lam)
        ds.to_csv(os.path.join(BASE, f'supply_Scarcity{int(lam*100)}.csv'),
                  index=False)
        print(f'Scarcity {int(lam*100)}: Y_jq sum {ds["Y_jq"].sum()}')

    # --- Demand Backlog ---
    for ratio in [0.25, 0.50, 0.75]:
        dd = generate_backlog(df_d, ratio)
        dd.to_csv(os.path.join(BASE, f'demand_Backlog{int(ratio*100)}.csv'),
                  index=False)
        print(f'Backlog {int(ratio*100)}%: {int(len(df_d)*ratio)} demanders '
              f'delayed')

    # --- Semantic Conflict (text-level only) ---
    for ratio in [0.15, 0.30, 0.50]:
        dd = generate_conflict(df_d, ratio)
        dd.to_csv(os.path.join(BASE, f'demand_Conflict{int(ratio*100)}.csv'),
                  index=False)
        n_mod = int(np.ceil(len(df_d) * ratio))
        print(f'Conflict {int(ratio*100)}%: ~{n_mod} demander texts modified')

    print('\nDone.')
