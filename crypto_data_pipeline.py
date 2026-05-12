import os
import json
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("results/.mplconfig").resolve()))

import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime

print("Initiating GraphQL connection to Uniswap V2 Subgraph...")

POOL_ADDRESS = "0xb4e16d0168e52d35cacd2c6185b44281ec28c9dc"
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_API_KEY = os.environ.get("GRAPH_API_KEY", "").strip()
FETCH_LIMIT = int(os.environ.get("GRAPH_FETCH_LIMIT", "1000"))
MAX_PAGES = int(os.environ.get("GRAPH_MAX_PAGES", "5"))
START_DATE = os.environ.get("GRAPH_START_DATE", "2022-01-01")
END_DATE = os.environ.get("GRAPH_END_DATE", "2022-12-31")

# 1. Query Real Data from the Uniswap V2 API (USDC-ETH Pool)
# The Graph requires an API key for subgraph queries.
SUBGRAPH_ID = "A3Np3RQbaBA6oKJgiwDJeo5T3zrYfGHPWFYayMwtNDum"  # Uniswap V2

url = f"https://gateway.thegraph.com/api/{GRAPH_API_KEY}/subgraphs/id/{SUBGRAPH_ID}" if GRAPH_API_KEY else None


def to_unix_timestamp(date_str: str, end_of_day: bool = False) -> int:
    parsed = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    if end_of_day:
        parsed = parsed + datetime.timedelta(days=1) - datetime.timedelta(seconds=1)
    return int(parsed.replace(tzinfo=datetime.timezone.utc).timestamp())


def build_query(first: int, start_ts: int, end_ts: int) -> str:
    return f"""
{{
  swaps(
    first: {first},
    orderBy: timestamp,
    orderDirection: asc,
    where: {{
      pair: "{POOL_ADDRESS}",
      timestamp_gt: {start_ts},
      timestamp_lte: {end_ts}
    }}
  ) {{
    transaction {{ id }}
    timestamp
    sender
    amount0In
    amount1In
    amount0Out
    amount1Out
    amountUSD
  }}
}}
"""


def fetch_real_swaps(graph_url: str, start_ts: int, end_ts: int) -> list[dict]:
    rows: list[dict] = []
    cursor_ts = start_ts - 1

    for _ in range(MAX_PAGES):
        query = build_query(FETCH_LIMIT, cursor_ts, end_ts)
        response = requests.post(graph_url, json={"query": query}, timeout=60)
        response.raise_for_status()
        json_body = response.json()

        if json_body.get("errors"):
            raise RuntimeError(f"The Graph returned errors: {json_body['errors']}")

        batch = json_body.get("data", {}).get("swaps", [])
        if not batch:
            break

        rows.extend(batch)
        cursor_ts = int(batch[-1]["timestamp"])

        if len(batch) < FETCH_LIMIT:
            break

    return rows

data = None
data_source = "synthetic_demo"
query_window = {
    "start_date": START_DATE,
    "end_date": END_DATE,
    "start_timestamp": to_unix_timestamp(START_DATE),
    "end_timestamp": to_unix_timestamp(END_DATE, end_of_day=True),
}

if url:
    try:
        data = fetch_real_swaps(
            url,
            start_ts=query_window["start_timestamp"],
            end_ts=query_window["end_timestamp"],
        )
        if data:
            data_source = "the_graph_uniswap_v2"
            print(f"Fetched {len(data)} swaps from The Graph for {START_DATE} to {END_DATE}.")
        else:
            print("The Graph query succeeded but returned no swaps for the requested window.")
    except Exception as exc:
        print(f"The Graph query failed: {exc}")

if data is None:
    print("Generating synthetic USDC-ETH swap data for pipeline demonstration...")
    np.random.seed(42)
    n = 1000
    base_ts = int(datetime.datetime(2024, 1, 1).timestamp())
    timestamps = base_ts + np.sort(np.random.randint(0, 180 * 86400, size=n))
    wallets = [f"0x{''.join(np.random.choice(list('0123456789abcdef'), 40))}" for _ in range(200)]
    data = []
    for i in range(n):
        amt_usd = float(np.random.lognormal(mean=7, sigma=1.5))
        eth_price = 3000 + np.random.normal(0, 300)
        is_buy = np.random.random() < 0.5
        data.append({
            'transaction': {'id': f'0xsynth{i:06d}'},
            'timestamp': str(timestamps[i]),
            'sender': np.random.choice(wallets),
            'amount0In': str(amt_usd if is_buy else 0),
            'amount1In': str(amt_usd / eth_price if not is_buy else 0),
            'amount0Out': str(0 if is_buy else amt_usd),
            'amount1Out': str(0 if not is_buy else amt_usd / eth_price),
            'amountUSD': str(amt_usd),
        })

# 2. Convert JSON to Pandas DataFrame
df = pd.DataFrame(data)
df["transaction_id"] = df["transaction"].apply(lambda x: x.get("id") if isinstance(x, dict) else None)
df["pool_address"] = POOL_ADDRESS
df["data_source"] = data_source

# 3. Clean and format the data
df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
df['amountUSD'] = df['amountUSD'].astype(float)
df['amount0In'] = df['amount0In'].astype(float) # USDC
df['amount1In'] = df['amount1In'].astype(float) # ETH
df['amount0Out'] = df['amount0Out'].astype(float)
df['amount1Out'] = df['amount1Out'].astype(float)
df['trade_direction'] = np.where(df['amount0In'] > 0, 'buy_eth_with_usdc', 'sell_eth_for_usdc')
df['estimated_eth_price'] = np.where(
    df['amount1Out'] > 0,
    df['amountUSD'] / np.maximum(df['amount1Out'], 1e-12),
    np.where(df['amount1In'] > 0, df['amountUSD'] / np.maximum(df['amount1In'], 1e-12), np.nan),
)
df = df.sort_values('timestamp').reset_index(drop=True)

processed_columns = [
    'timestamp',
    'transaction_id',
    'sender',
    'amountUSD',
    'amount0In',
    'amount1In',
    'amount0Out',
    'amount1Out',
    'trade_direction',
    'estimated_eth_price',
    'pool_address',
    'data_source',
]
processed_df = df[processed_columns].copy()

train_end = int(len(processed_df) * 0.8)
val_end = int(len(processed_df) * 0.9)
split_dfs = {
    'train': processed_df.iloc[:train_end].copy(),
    'val': processed_df.iloc[train_end:val_end].copy(),
    'test': processed_df.iloc[val_end:].copy(),
}

processed_path = PROCESSED_DIR / "uniswap_v2_usdc_eth_processed.csv"
processed_df.to_csv(processed_path, index=False)

for split_name, split_df in split_dfs.items():
    split_df.to_csv(PROCESSED_DIR / f"uniswap_v2_usdc_eth_{split_name}.csv", index=False)

metadata = {
    "dataset_name": "uniswap_v2_usdc_eth_processed",
    "pool_address": POOL_ADDRESS,
    "data_source": data_source,
    "query_window": query_window,
    "num_rows": int(len(processed_df)),
    "num_unique_wallets": int(processed_df["sender"].nunique()),
    "start_timestamp": processed_df["timestamp"].min().isoformat() if not processed_df.empty else None,
    "end_timestamp": processed_df["timestamp"].max().isoformat() if not processed_df.empty else None,
    "split_strategy": "chronological_80_10_10",
    "columns": processed_columns,
}

with open(PROCESSED_DIR / "dataset_card.json", "w", encoding="utf-8") as fp:
    json.dump(metadata, fp, indent=2)

print(f"\nDataset Shape: {df.shape}")
print("\nFirst 3 rows of processed DEX data:")
print(df[['timestamp', 'sender', 'amountUSD']].head(3))
print(f"\nSaved processed dataset: {processed_path}")
print(f"Saved dataset metadata: {PROCESSED_DIR / 'dataset_card.json'}")

# ==========================================
# GENERATE REPORT FIGURES
# ==========================================

# Figure 2: Distribution of Trade Sizes (Sparsity Analysis)
plt.figure(figsize=(8, 5))
sns.histplot(df['amountUSD'][df['amountUSD'] < 50000], bins=50, kde=True, color='blue')
plt.title('Figure 2: Distribution of Trade Sizes in USDC-ETH Pool')
plt.xlabel('Trade Size (USD)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('fig2_trade_distribution.png', dpi=300)
print("\nSaved: fig2_trade_distribution.png")

# Figure 3: User Activity (Identifying active vs one-off traders)
user_counts = df['sender'].value_counts()
plt.figure(figsize=(8, 5))
plt.hist(user_counts.values, bins=range(1, 15), align='left', color='orange', edgecolor='black')
plt.title('Figure 3: User Transaction Frequency')
plt.xlabel('Number of Transactions per User Address')
plt.ylabel('Number of Unique Users')
plt.xticks(range(1, 15))
plt.tight_layout()
plt.savefig('fig3_user_frequency.png', dpi=300)
print("Saved: fig3_user_frequency.png")

# Figure 4: Feature Correlation Matrix
plt.figure(figsize=(7, 6))
# Correlate amounts to see trading patterns
corr_df = df[['amount0In', 'amount1In', 'amount0Out', 'amount1Out', 'amountUSD']].corr()
sns.heatmap(corr_df, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('Figure 4: Market Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('fig4_correlation.png', dpi=300)
print("Saved: fig4_correlation.png")

print("\nData pipeline execution complete. You are ready to add these to your report!")
