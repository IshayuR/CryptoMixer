# CryptoMixer Reproduction for DEX Trading Prediction

This repository contains the code, generated figures, and experiment outputs used for a course project studying the CryptoMixer architecture for fine-grained decentralized exchange (DEX) trading prediction.

## Repository Contents

- `models.py`: PyTorch implementations of the CryptoMixer-style model and baseline LSTM/GRU models.
- `train_cryptomixer.py`: End-to-end training, evaluation, ablation, and figure generation script.
- `crypto_data_pipeline.py`: Data pipeline and exploratory figure generation for the Uniswap V2 USDC-ETH pool.
- `results/`: Saved metrics tables and plots used in the report.
- `fig*.png`: Additional generated figures referenced during report drafting.

## Environment Setup

Use Python 3.10+ and install dependencies with:

```bash
python3 -m pip install -r requirements.txt
```

## Quick Reproduction

Run the training and evaluation workflow:

```bash
python3 train_cryptomixer.py
```

Run the data pipeline / exploratory analysis workflow:

```bash
python3 crypto_data_pipeline.py
```

Outputs are written to `results/` and to the repository root for the exploratory figures.

## Data Access

If you are sharing processed data outside GitHub because of file size limits, add the public OneDrive link here before submitting or sharing the repository.

`Processed data link: <ADD YOUR ONEDRIVE LINK HERE>`

## Submission Checklist

- Public GitHub repository link is accessible without requesting permission.
- `README.md` explains what the project is and how to run it.
- `requirements.txt` installs all Python dependencies.
- Processed data link is included if the dataset is too large for GitHub.
- Final report PDF is submitted alongside the repository link.

## Notes

The repository includes a lightweight end-to-end reproduction workflow and saved outputs so that a reviewer can inspect the architecture, rerun the code, and verify the generated figures/results structure quickly.
