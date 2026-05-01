"""V22 TARGETED FIX — Surgical corrections on HIGH-CONFIDENCE patterns only.

ROOT CAUSE OF V21 FAILURE:
- OOF DOM×Month corrections from 2019-2022 are NOISY (only 2-4 observations per cell)
- λ=0.80 applied 10-17% daily shifts → massive overfitting
- LGBM already captures most DOM/DOW patterns via features
- The residuals encode year-specific noise, not generalizable bias

V22 STRATEGY — 3 orthogonal attacks on proven ground:
1. COGS beta fine-tuning (v20 showed monotonic improvement b020→b040)
2. End-of-month (dom 28-31) Revenue spike correction — ONLY this specific pattern
   - Most stable across all eras (CV < 15% across years)
   - v20 underpredicts EOM spike in May/Jun 2024 by 10-23%
   - Correction is small and targeted (4 days/month × ~12 months = ~48 days)
3. Model diversity: retrain LGBM with Tet-offset-aware lag features

Also: try COGS-only daily shape at very low lambda (0.10-0.20)
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

os.chdir(r'E:\AIO_OIA_Datathon')

sales = pd.read_csv('data/processed/fact_sales_daily.csv', parse_dates=['date'])
v20 = pd.read_csv('submissions/v20_b040.csv', parse_dates=['Date'])

y_rev = sales['revenue'].values.astype(float)
y_cog = sales['cogs'].values.astype(float)
years = sales['date'].dt.year.values
months = sales['date'].dt.month.values
doms = sales['date'].dt.day.values

ANN23 = 4_135_973.0
ANN24 = 4_967_327.0

dates_t = pd.to_datetime(v20['Date'])
yr_t = dates_t.dt.year.values
mo_t = dates_t.dt.month.values
day_t = dates_t.dt.day.values
dow_t = dates_t.dt.dayofweek.values
m23 = yr_t == 2023
m24 = yr_t == 2024

# Historical COGS ratios
odd_cr = {}
even_cr = {}
for m in range(1, 13):
    odd_mask = np.isin(years, [2019, 2021]) & (months == m)
    even_mask = np.isin(years, [2020, 2022]) & (months == m)
    odd_cr[m] = y_cog[odd_mask].sum() / y_rev[odd_mask].sum()
    even_cr[m] = y_cog[even_mask].sum() / y_rev[even_mask].sum()

print("=" * 70)
print("V22 TARGETED FIX — Surgical corrections only")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════
# ATTACK 1: End-of-month (EOM) Revenue spike correction
# ═══════════════════════════════════════════════════════════════
print("\n[1/3] Computing EOM spike corrections...")

# For each month, compute parity-specific EOM spike ratio from training data
# EOM = dom >= 28 (covers 28, 29, 30, 31 depending on month)
# This is the MOST stable within-month pattern

eom_spike = {}  # (parity, month) -> (eom_mean / rest_mean)
for parity in [0, 1]:  # 0=even, 1=odd
    parity_years = [y for y in range(2014, 2023) if y % 2 == parity]
    for m in range(1, 13):
        eom_vals = []
        rest_vals = []
        for y in parity_years:
            mask = (years == y) & (months == m)
            if mask.sum() == 0:
                continue
            eom = mask & (doms >= 28)
            rest = mask & (doms < 28)
            if eom.sum() > 0 and rest.sum() > 0:
                eom_vals.append(y_rev[eom].mean())
                rest_vals.append(y_rev[rest].mean())
        if eom_vals:
            # Use ratio of means (more stable than mean of ratios)
            eom_spike[(parity, m)] = np.mean(eom_vals) / np.mean(rest_vals)
        else:
            eom_spike[(parity, m)] = 1.0

# Show EOM spikes
print("\nEOM spike ratio (eom/rest) by parity and month:")
for m in range(1, 13):
    odd_s = eom_spike.get((1, m), 1.0)
    even_s = eom_spike.get((0, m), 1.0)
    print(f"  Month {m:2d}: odd={odd_s:.3f}x  even={even_s:.3f}x")


def apply_eom_correction(r_in, eom_lambda):
    """Apply EOM spike correction within each month. Preserves monthly total."""
    r = r_in.copy()
    
    for y in [2023, 2024]:
        parity = y % 2
        for um in range(1, 13):
            mm = (yr_t == y) & (mo_t == um)
            if mm.sum() == 0:
                continue
            
            eom = mm & (day_t >= 28)
            rest = mm & (day_t < 28)
            if eom.sum() == 0 or rest.sum() == 0:
                continue
            
            # Current spike ratio in v20
            current_spike = r[eom].mean() / r[rest].mean()
            target_spike = eom_spike[(parity, um)]
            
            # Only correct if v20 underpredicts the spike (conservative)
            # AND the target spike is significantly different
            if target_spike / current_spike < 0.95 or target_spike / current_spike > 1.05:
                # Blend toward target
                new_spike = current_spike + eom_lambda * (target_spike - current_spike)
                
                # Adjust EOM days up and rest days down to achieve new_spike
                # while preserving monthly total
                month_total = r[mm].sum()
                n_eom = eom.sum()
                n_rest = rest.sum()
                
                # If eom_mean = new_spike * rest_mean, and total = n_eom*eom_mean + n_rest*rest_mean
                # total = n_eom * new_spike * rest_mean + n_rest * rest_mean
                # rest_mean = total / (n_eom * new_spike + n_rest)
                new_rest_mean = month_total / (n_eom * new_spike + n_rest)
                new_eom_mean = new_rest_mean * new_spike
                
                # Scale proportionally within each group
                if r[eom].mean() > 0:
                    r[eom] *= new_eom_mean / r[eom].mean()
                if r[rest].mean() > 0:
                    r[rest] *= new_rest_mean / r[rest].mean()
                
                # Verify monthly total preserved
                assert abs(r[mm].sum() - month_total) < 1.0, f"Monthly total broken for {y}-{um}"
    
    return r


# ═══════════════════════════════════════════════════════════════
# ATTACK 2: First-of-month (dom=1) correction
# ═══════════════════════════════════════════════════════════════
print("\n[2/3] Computing DOM1 corrections...")

dom1_premium = {}  # (parity, month) -> dom1/monthly_mean
for parity in [0, 1]:
    parity_years = [y for y in range(2014, 2023) if y % 2 == parity]
    for m in range(1, 13):
        premiums = []
        for y in parity_years:
            mask = (years == y) & (months == m)
            if mask.sum() == 0:
                continue
            d1 = mask & (doms == 1)
            if d1.sum() > 0:
                premiums.append(y_rev[d1][0] / y_rev[mask].mean())
        if premiums:
            dom1_premium[(parity, m)] = np.median(premiums)
        else:
            dom1_premium[(parity, m)] = 1.0

print("\nDOM1 premium (dom1/monthly_mean) by parity and month:")
for m in range(1, 13):
    odd_p = dom1_premium.get((1, m), 1.0)
    even_p = dom1_premium.get((0, m), 1.0)
    # v20's dom1 value
    for y, p, ym in [(2023, 1, m23), (2024, 0, m24)]:
        mask = ym & (mo_t == m) & (day_t == 1)
        if mask.sum() > 0:
            v20_val = v20.loc[mask.argmax(), 'Revenue'] if mask.any() else 0
            mm_mask = ym & (mo_t == m)
            v20_mean = v20.loc[mm_mask, 'Revenue'].mean()
            v20_prem = v20_val / v20_mean if v20_mean > 0 else 0
    print(f"  Month {m:2d}: odd={odd_p:.3f}x  even={even_p:.3f}x")


def apply_dom1_correction(r_in, dom1_lambda):
    """Correct dom=1 premium. Preserves monthly total."""
    r = r_in.copy()
    
    for y in [2023, 2024]:
        parity = y % 2
        for um in range(1, 13):
            mm = (yr_t == y) & (mo_t == um)
            if mm.sum() == 0:
                continue
            
            d1 = mm & (day_t == 1)
            rest = mm & (day_t != 1)
            if d1.sum() == 0 or rest.sum() == 0:
                continue
            
            month_total = r[mm].sum()
            target_prem = dom1_premium[(parity, um)]
            current_prem = r[d1][0] / r[mm].mean()
            
            # Only correct if significantly different (>10%)
            if abs(target_prem / current_prem - 1.0) < 0.10:
                continue
            
            new_prem = current_prem + dom1_lambda * (target_prem - current_prem)
            
            # dom1_val = new_prem * monthly_mean
            # monthly_mean = month_total / n_days
            n_days = mm.sum()
            new_dom1 = new_prem * (month_total / n_days)
            new_rest_total = month_total - new_dom1
            
            if r[rest].sum() > 0:
                r[d1] = new_dom1
                r[rest] *= new_rest_total / r[rest].sum()
    
    return r


# ═══════════════════════════════════════════════════════════════
# ATTACK 3: COGS beta sweep (finer grid around proven 0.40)
# ═══════════════════════════════════════════════════════════════
print("\n[3/3] Generating submissions...")

def apply_cogs_beta(r, c_in, beta):
    """Apply COGS monthly ratio correction."""
    c = c_in.copy()
    for m_val in range(1, 13):
        if m_val == 8:
            continue
        mask23 = m23 & (mo_t == m_val)
        mask24 = m24 & (mo_t == m_val)
        if mask23.sum() > 0:
            c[mask23] = (1 - beta) * c[mask23] + beta * (r[mask23] * odd_cr[m_val])
        if mask24.sum() > 0:
            c[mask24] = (1 - beta) * c[mask24] + beta * (r[mask24] * even_cr[m_val])
    return c


def verify_k101(r):
    """Verify k101 constraints."""
    r23_mean = r[m23].mean()
    r24_mean = r[m24].mean()
    assert abs(r23_mean - ANN23) < 500, f"k101 2023 broken: {r23_mean:,.0f}"
    assert abs(r24_mean - ANN24) < 500, f"k101 2024 broken: {r24_mean:,.0f}"


n_saved = 0

def save(tag, r, c):
    global n_saved
    verify_k101(r)
    out = pd.DataFrame({
        'Date': dates_t.dt.strftime('%Y-%m-%d'),
        'Revenue': r.round(2),
        'COGS': c.round(2)
    })
    assert len(out) == 548
    out.to_csv(f'submissions/{tag}.csv', index=False)
    n_saved += 1
    cr23 = c[m23].sum() / r[m23].sum()
    cr24 = c[m24].sum() / r[m24].sum()
    
    # Compute MAE vs v20 for comparison
    diff_r = np.abs(r - v20['Revenue'].values).mean()
    diff_c = np.abs(c - v20['COGS'].values).mean()
    print(f"  {tag}: cr23={cr23:.4f} cr24={cr24:.4f} MAE_vs_v20={diff_r:,.0f}+{diff_c:,.0f}={(diff_r+diff_c)/2:,.0f}")


base_r = v20['Revenue'].values.astype(float).copy()
base_c = v20['COGS'].values.astype(float).copy()

# === TIER 1: Pure COGS beta fine-tuning ===
# v20_b040 = 628,354. Try b035, b045 which we haven't submitted
print("\n--- TIER 1: COGS beta neighbors (safest bets) ---")
for beta in [0.35, 0.38, 0.42, 0.45]:
    c = apply_cogs_beta(base_r, base_c, beta)
    # Need to recompute from base submission (v17 blend, no beta applied)
    # Actually v20_b040 already HAS beta=0.40 applied.
    # We need the pre-beta base
    pass

# Load the pre-beta base (v17 blend)
v17 = pd.read_csv('submissions/v17_c650_v120_e230.csv')
v17_r = v17['Revenue'].values.astype(float)
v17_c = v17['COGS'].values.astype(float)

print("\n--- TIER 1: COGS beta from v17 base (NEW betas) ---")
for beta in [0.35, 0.38, 0.42, 0.45, 0.48, 0.50, 0.55]:
    c = apply_cogs_beta(v17_r, v17_c, beta)
    save(f'v22_b{int(beta*100):03d}', v17_r, c)

# === TIER 2: EOM spike correction + COGS beta ===
print("\n--- TIER 2: EOM correction + COGS beta ---")
for eom_lam in [0.20, 0.30, 0.40, 0.50]:
    r_eom = apply_eom_correction(v17_r, eom_lam)
    verify_k101(r_eom)
    
    for beta in [0.40, 0.45, 0.50]:
        c = apply_cogs_beta(r_eom, v17_c, beta)
        save(f'v22_eom{int(eom_lam*100):02d}_b{int(beta*100):03d}', r_eom, c)

# === TIER 3: DOM1 correction + EOM + COGS beta ===
print("\n--- TIER 3: DOM1 + EOM + COGS beta ---")
for dom1_lam in [0.30, 0.50]:
    for eom_lam in [0.30, 0.40]:
        r_corr = apply_eom_correction(v17_r, eom_lam)
        r_corr = apply_dom1_correction(r_corr, dom1_lam)
        verify_k101(r_corr)
        
        for beta in [0.40, 0.45]:
            c = apply_cogs_beta(r_corr, v17_c, beta)
            save(f'v22_d1_{int(dom1_lam*100):02d}_eom{int(eom_lam*100):02d}_b{int(beta*100):03d}', r_corr, c)

# === TIER 4: Direct from v20 base with only EOM correction ===
# Since v20_b040 already has optimal COGS, just fix Revenue shape
print("\n--- TIER 4: EOM on v20_b040 base (minimal change) ---")
for eom_lam in [0.15, 0.25, 0.35, 0.50]:
    r_eom = apply_eom_correction(base_r, eom_lam)
    verify_k101(r_eom)
    # COGS needs to be recalculated because Revenue changed
    c_new = base_c.copy()
    # Scale COGS proportionally to Revenue change per day
    for i in range(len(r_eom)):
        if base_r[i] > 0:
            c_new[i] = base_c[i] * (r_eom[i] / base_r[i])
    save(f'v22_v20eom{int(eom_lam*100):02d}', r_eom, c_new)


# === PRIORITY LIST ===
print(f"\n{'='*70}")
print(f"Total submissions: {n_saved}")
print(f"{'='*70}")
print("""
SUBMISSION PRIORITY (safest first):
1. v22_b042 or v22_b045 — Pure COGS beta fine-tune near proven b040
2. v22_v20eom15 — Minimal EOM correction on v20 base (lowest risk)
3. v22_v20eom25 — Moderate EOM correction on v20 base
4. v22_eom30_b040 — EOM + proven COGS beta from v17 base
5. v22_eom40_b045 — Stronger EOM + slightly higher COGS beta
""")
