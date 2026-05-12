# Dataset Metadata

This directory contains metadata describing the canonical dataset snapshot used by the repository.

## Files

- `dataset_card.json`: summary metadata for the four canonical CSV files in `data/`

## Canonical CSVs

The dataset files referenced by the metadata card are:

- `data/uniswap_v2_usdc_eth_processed_uniswap.csv`
- `data/uniswap_v2_usdc_eth_train_uniswap.csv`
- `data/uniswap_v2_usdc_eth_val_uniswap.csv`
- `data/uniswap_v2_usdc_eth_test_uniswap.csv`

## Notes

The row-level `data_source` column in those CSVs should be treated as the authoritative provenance indicator for the current repository snapshot.
