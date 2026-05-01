"""Answer all 10 official MCQ questions from the competition."""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

RAW = "src/data/raw"

# Load all data
orders      = pd.read_csv(f"{RAW}/orders.csv",      parse_dates=["order_date"])
products    = pd.read_csv(f"{RAW}/products.csv")
customers   = pd.read_csv(f"{RAW}/customers.csv",    parse_dates=["signup_date"])
order_items = pd.read_csv(f"{RAW}/order_items.csv")
returns     = pd.read_csv(f"{RAW}/returns.csv",       parse_dates=["return_date"])
web         = pd.read_csv(f"{RAW}/web_traffic.csv",   parse_dates=["date"])
geography   = pd.read_csv(f"{RAW}/geography.csv")
payments    = pd.read_csv(f"{RAW}/payments.csv")
sales       = pd.read_csv(f"{RAW}/sales.csv",         parse_dates=["Date"])

print("All data loaded.\n")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════
# Q1: Median inter-order gap for customers with >1 order
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Q1: Median inter-order gap (days) for customers with >1 order")
print("=" * 70)

customer_orders = orders.sort_values(["customer_id", "order_date"])
customer_orders["prev_date"] = customer_orders.groupby("customer_id")["order_date"].shift(1)
customer_orders["gap_days"] = (customer_orders["order_date"] - customer_orders["prev_date"]).dt.days
gaps = customer_orders["gap_days"].dropna()

median_gap = gaps.median()
mean_gap = gaps.mean()
print(f"  Median inter-order gap: {median_gap:.1f} days")
print(f"  Mean inter-order gap:   {mean_gap:.1f} days")
print(f"  Count of gaps:          {len(gaps):,}")
print(f"  Percentiles: 25%={gaps.quantile(0.25):.0f}, 50%={gaps.quantile(0.50):.0f}, 75%={gaps.quantile(0.75):.0f}")

if median_gap < 60:
    print("  → ANSWER: A) 30 ngày")
elif median_gap < 135:
    print("  → ANSWER: B) 90 ngày")
elif median_gap < 270:
    print("  → ANSWER: C) 180 ngày")
else:
    print("  → ANSWER: D) 365 ngày")

# ══════════════════════════════════════════════════════════════════════
# Q2: Segment with highest avg gross margin (price-cogs)/price
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Q2: Segment with highest avg gross margin (price-cogs)/price")
print("=" * 70)

products["gross_margin"] = (products["price"] - products["cogs"]) / products["price"]
seg_margin = products.groupby("segment")["gross_margin"].mean().sort_values(ascending=False)
print("  Gross margin by segment:")
for seg, margin in seg_margin.items():
    print(f"    {seg:20s}: {margin:.4f} ({margin*100:.2f}%)")
print(f"  → ANSWER: {seg_margin.index[0]}")

# ══════════════════════════════════════════════════════════════════════
# Q3: Top return reason for Streetwear products
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Q3: Top return reason for Streetwear (join returns × products)")
print("=" * 70)

ret_prod = returns.merge(products[["product_id", "category"]], on="product_id", how="left")
streetwear_returns = ret_prod[ret_prod["category"] == "Streetwear"]
reason_counts = streetwear_returns["return_reason"].value_counts()
print(f"  Streetwear returns: {len(streetwear_returns):,}")
print("  Return reasons:")
for reason, cnt in reason_counts.items():
    pct = cnt / len(streetwear_returns) * 100
    print(f"    {reason:25s}: {cnt:>5,} ({pct:.1f}%)")
print(f"  → ANSWER: {reason_counts.index[0]}")

# ══════════════════════════════════════════════════════════════════════
# Q4: Traffic source with lowest avg bounce_rate
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Q4: Traffic source with lowest avg bounce_rate")
print("=" * 70)

bounce_by_source = web.groupby("traffic_source")["bounce_rate"].mean().sort_values()
print("  Avg bounce_rate by source:")
for src, br in bounce_by_source.items():
    print(f"    {src:20s}: {br:.4f} ({br*100:.2f}%)")
print(f"  → ANSWER: {bounce_by_source.index[0]} (lowest)")

# ══════════════════════════════════════════════════════════════════════
# Q5: % of order_items rows with promo_id not null
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Q5: % of order_items with promo_id not null")
print("=" * 70)

has_promo = order_items["promo_id"].notna().sum()
total_items = len(order_items)
promo_pct = has_promo / total_items * 100
print(f"  Total order_items rows: {total_items:,}")
print(f"  Rows with promo_id:     {has_promo:,}")
print(f"  Percentage:             {promo_pct:.2f}%")

if promo_pct < 18:
    print("  → ANSWER: A) 12%")
elif promo_pct < 32:
    print("  → ANSWER: B) 25%")
elif promo_pct < 46:
    print("  → ANSWER: C) 39%")
else:
    print("  → ANSWER: D) 54%")

# ══════════════════════════════════════════════════════════════════════
# Q6: Age group with highest avg orders per customer (non-null age_group)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Q6: Age group with highest avg orders per customer")
print("=" * 70)

cust_with_age = customers[customers["age_group"].notna()].copy()
orders_per_cust = orders.groupby("customer_id").size().reset_index(name="n_orders")
cust_orders = cust_with_age.merge(orders_per_cust, on="customer_id", how="left")
cust_orders["n_orders"] = cust_orders["n_orders"].fillna(0)

avg_orders_by_age = cust_orders.groupby("age_group")["n_orders"].mean().sort_values(ascending=False)
total_by_age = cust_orders.groupby("age_group")["n_orders"].agg(["sum", "count", "mean"])
print("  Avg orders per customer by age_group:")
for ag in avg_orders_by_age.index:
    row = total_by_age.loc[ag]
    print(f"    {ag:10s}: {row['mean']:.2f} avg ({row['sum']:.0f} orders / {row['count']:.0f} customers)")
print(f"  → ANSWER: {avg_orders_by_age.index[0]}")

# ══════════════════════════════════════════════════════════════════════
# Q7: Region with highest total Revenue
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Q7: Region with highest total Revenue")
print("=" * 70)

# Join orders → order_items → sum revenue per order → join geography
# Revenue per order from order_items: quantity * unit_price
order_items["line_total"] = order_items["quantity"] * order_items["unit_price"]
order_revenue = order_items.groupby("order_id")["line_total"].sum().reset_index(name="order_revenue")

# Join with orders to get zip
orders_with_rev = orders.merge(order_revenue, on="order_id", how="left")
# Join with geography to get region
orders_with_region = orders_with_rev.merge(geography[["zip", "region"]], on="zip", how="left")

rev_by_region = orders_with_region.groupby("region")["order_revenue"].sum().sort_values(ascending=False)
print("  Total Revenue by region:")
for reg, rev in rev_by_region.items():
    pct = rev / rev_by_region.sum() * 100
    print(f"    {reg:10s}: {rev:>18,.0f} ({pct:.1f}%)")

# Check if roughly equal
max_rev = rev_by_region.max()
min_rev = rev_by_region.min()
ratio = max_rev / min_rev if min_rev > 0 else 999
print(f"  Max/Min ratio: {ratio:.2f}")
if ratio < 1.15:
    print("  → ANSWER: D) Cả ba vùng có doanh thu xấp xỉ bằng nhau")
else:
    print(f"  → ANSWER: {rev_by_region.index[0]}")

# ══════════════════════════════════════════════════════════════════════
# Q8: Most used payment method for cancelled orders
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Q8: Most used payment method for cancelled orders")
print("=" * 70)

cancelled = orders[orders["order_status"] == "cancelled"]
print(f"  Total cancelled orders: {len(cancelled):,}")
pay_cancelled = cancelled["payment_method"].value_counts()
print("  Payment methods for cancelled:")
for method, cnt in pay_cancelled.items():
    pct = cnt / len(cancelled) * 100
    print(f"    {method:20s}: {cnt:>6,} ({pct:.1f}%)")
print(f"  → ANSWER: {pay_cancelled.index[0]}")

# ══════════════════════════════════════════════════════════════════════
# Q9: Size with highest return rate (returns count / order_items count)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Q9: Size with highest return rate (returns / order_items by product)")
print("=" * 70)

# Join order_items with products to get size
oi_prod = order_items.merge(products[["product_id", "size"]], on="product_id", how="left")
items_by_size = oi_prod.groupby("size").size().reset_index(name="n_items")

# Join returns with products to get size
ret_prod2 = returns.merge(products[["product_id", "size"]], on="product_id", how="left")
returns_by_size = ret_prod2.groupby("size").size().reset_index(name="n_returns")

size_rates = items_by_size.merge(returns_by_size, on="size", how="left")
size_rates["n_returns"] = size_rates["n_returns"].fillna(0)
size_rates["return_rate"] = size_rates["n_returns"] / size_rates["n_items"] * 100
size_rates = size_rates.sort_values("return_rate", ascending=False)

print("  Return rate by size:")
for _, row in size_rates.iterrows():
    print(f"    {row['size']:5s}: {row['n_returns']:.0f} returns / {row['n_items']:.0f} items = {row['return_rate']:.2f}%")
print(f"  → ANSWER: {size_rates.iloc[0]['size']}")

# ══════════════════════════════════════════════════════════════════════
# Q10: Installment plan with highest avg payment_value per order
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Q10: Installment plan with highest avg payment_value")
print("=" * 70)

avg_pay_by_inst = payments.groupby("installments")["payment_value"].mean().sort_values(ascending=False)
print("  Avg payment_value by installments:")
for inst, avg_val in avg_pay_by_inst.items():
    cnt = (payments["installments"] == inst).sum()
    print(f"    {inst:>2d} kỳ: {avg_val:>12,.2f} (count: {cnt:,})")

# Map to answer choices
top_inst = avg_pay_by_inst.index[0]
if top_inst == 1:
    print("  → ANSWER: A) 1 kỳ (trả một lần)")
elif top_inst == 3:
    print("  → ANSWER: B) 3 kỳ")
elif top_inst == 6:
    print("  → ANSWER: C) 6 kỳ")
elif top_inst == 12:
    print("  → ANSWER: D) 12 kỳ")
else:
    print(f"  → ANSWER: {top_inst} kỳ (not in choices, pick closest)")

# ══════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FINAL ANSWERS SUMMARY")
print("=" * 70)
