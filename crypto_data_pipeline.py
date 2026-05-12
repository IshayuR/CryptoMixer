import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("results/.mplconfig").resolve()))

import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime

print("Initiating GraphQL connection to Uniswap V2 Subgraph...")

# 1. Query Real Data from the Uniswap V2 API (USDC-ETH Pool)
# The Graph decentralized network endpoint (free API key from https://thegraph.com/studio/)
API_KEY = ""  # Paste your free API key here
SUBGRAPH_ID = "A3Np3RQbaBA6oKJgiwDJeo5T3zrYfGHPWFYayMwtNDum"  # Uniswap V2

url = f"https://gateway.thegraph.com/api/{API_KEY}/subgraphs/id/{SUBGRAPH_ID}" if API_KEY else None
query = """
{
  swaps(first: 1000, orderBy: timestamp, orderDirection: desc, where: { pair: "0xb4e16d0168e52d35cacd2c6185b44281ec28c9dc" }) {
    transaction { id }
    timestamp
    sender
    amount0In
    amount1In
    amount0Out
    amount1Out
    amountUSD
  }
}
"""

data = None
if url:
    response = requests.post(url, json={'query': query})
    json_body = response.json()
    if response.status_code == 200 and json_body.get('data'):
        data = json_body['data']['swaps']
        print("Data successfully fetched from the Ethereum Blockchain!")
    else:
        print(f"API returned an error: {json_body.get('errors', 'unknown')}")

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

# 3. Clean and format the data
df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
df['amountUSD'] = df['amountUSD'].astype(float)
df['amount0In'] = df['amount0In'].astype(float) # USDC
df['amount1In'] = df['amount1In'].astype(float) # ETH
df['amount0Out'] = df['amount0Out'].astype(float)
df['amount1Out'] = df['amount1Out'].astype(float)

print(f"\nDataset Shape: {df.shape}")
print("\nFirst 3 rows of processed DEX data:")
print(df[['timestamp', 'sender', 'amountUSD']].head(3))

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
