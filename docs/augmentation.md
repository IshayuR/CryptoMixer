# Augmentation Strategy

This project uses augmentation in two complementary ways.

## Dataset-Level Augmentation

The training workflow in [train_cryptomixer.py](/Users/ishayuray/CSE 5819/CryptoMixer/train_cryptomixer.py:89) constructs sparse DEX-style user sequences with:

- heterogeneous user activity levels
- market regime variation
- correlated price, volatility, gas, and pool-state behavior
- user-specific behavioral clusters
- noisy buy/sell outcomes

The resulting sequence representation includes timestep position, trade direction indicators, and market-state proxies. This serves as the data-construction layer used by the current reproducible training pipeline.

## Model-Level Augmentation

The architecture in [models.py](/Users/ishayuray/CSE 5819/CryptoMixer/models.py:9) includes a `MarketInfoAugmenter` module that applies learned self-attention across timesteps:

- query, key, and value projections are learned from hidden representations
- attention weights are computed over the full sequence
- the attention-weighted representation is added back to the original hidden state

This augmentation step is intended to enrich each timestep with neighboring market context before downstream fusion and classification.

## Practical Interpretation

In presentation terms, the augmentation story is:

> The project augments information at two levels. The training pipeline builds sparse DEX-like user sequences with structured market variability, and the model then applies attention-based market augmentation across timesteps so each observation is contextualized before prediction.

## Related Code Paths

- dataset construction: [train_cryptomixer.py](/Users/ishayuray/CSE 5819/CryptoMixer/train_cryptomixer.py:89)
- data splitting: [train_cryptomixer.py](/Users/ishayuray/CSE 5819/CryptoMixer/train_cryptomixer.py:225)
- market augmentation block: [models.py](/Users/ishayuray/CSE 5819/CryptoMixer/models.py:9)
- conditional market mixing: [models.py](/Users/ishayuray/CSE 5819/CryptoMixer/models.py:31)
- processed dataset export: [crypto_data_pipeline.py](/Users/ishayuray/CSE 5819/CryptoMixer/crypto_data_pipeline.py:1)
