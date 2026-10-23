# HealthTwin: AI-Powered Human Digital Twin With Blockchain-Secured Health Prediction

HealthTwin is a research-grade preventive healthcare prototype that combines an explainable AI digital twin, multi-disease risk prediction, lifestyle counterfactual simulation, and blockchain-anchored evidence integrity.

The project is designed for IEEE/Springer papers, final-year demonstrations, hackathons, and investor-facing technical demos.

## Research Hypothesis

A continuously updated AI digital twin trained on population-scale epidemiological data can estimate multi-disease risk trajectories, explain them with SHAP, simulate lifestyle interventions, and anchor medical evidence plus AI predictions with blockchain hashes.

## Core Contributions

- **Health Age:** biological health age estimated from multi-risk profile.
- **MDRC Score:** clinically weighted composite of CVD, diabetes, hypertension, metabolic syndrome, and stress risks.
- **Twin Divergence Score:** drift from the user's first recorded baseline.
- **Counterfactual Simulation:** before/after risk changes from modifiable interventions.
- **SHAP Explainability:** top risk-raising and risk-reducing factors as natural-language sentences.
- **Blockchain Evidence Chain:** SHA-256 hashes of reports and predictions stored through a Polygon-ready smart contract.
- **Presentation-Ready UX:** demo health profiles, dashboard evidence chain, printable summary report, and intervention cards.

## Architecture

```text
frontend/      Vanilla HTML/CSS/JS, Chart.js, MetaMask integration
backend/       Flask API, JWT auth, SQLite, upload hashing, AI routes
ml/            NHANES download, preprocessing, labels, XGBoost training
blockchain/    Solidity contract and Polygon deployment guide
database/      SQLite schema
docs/          Architecture, API reference, paper draft
```

## Run

```bash
pip install -r requirements.txt
python backend/app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Demo Flow

1. Sign up or log in.
2. Open `Health Form` and use one of the demo profile buttons.
3. Submit the snapshot to compute Health Age, MDRC, divergence, and risk explanations.
4. Open `Simulation` and run the counterfactual lifestyle scenario.
5. Open `Reports`, upload a PDF/image, anchor the hash, and verify integrity.
6. Return to `Dashboard` to see the latest blockchain evidence-chain card.
7. Use `Export Report` to print or save the dashboard as a PDF.

## Train NHANES-Derived Models

```bash
python ml/preprocessing/nhanes_loader.py
python ml/preprocessing/feature_engineer.py
python ml/preprocessing/label_creator.py
python ml/training/train_all_models.py --leakage-aware
```

The backend runs with deterministic clinical heuristics until the five model artifacts are generated in `ml/saved_models/`.
The repository intentionally ignores generated `.pkl`, metric `.json`, raw NHANES `.xpt`, and processed `.csv` files because they are reproducible artifacts and can be large. A fresh clone should run the commands above before claiming trained-model results.

Current processed dataset after the NHANES 2017-March 2020 pipeline:

```text
Merged NHANES table: 15,560 subjects, 259 raw columns
Engineered feature table: 10,767 subjects, 28 health features
Final labelled table: 10,767 subjects, 5 binary targets
```

Expected output artifacts:

```text
ml/saved_models/diabetes_model.pkl
ml/saved_models/cvd_model.pkl
ml/saved_models/hypertension_model.pkl
ml/saved_models/metabolic_model.pkl
ml/saved_models/stress_model.pkl
ml/saved_models/*_leakage_aware_model.pkl
ml/saved_models/training_metrics.json
ml/saved_models/training_metrics_leakage_aware.json
ml/saved_models/model_card.json
```

Several prototype labels are threshold-derived from biomarkers also present in the feature matrix. For IEEE-style reporting, use the leakage-aware ablation metrics from `training_metrics_leakage_aware.json` as the defensible internal result, and clearly label the original metrics as leakage-prone internal CV.

## API Highlights

- `POST /api/auth/register`
- `POST /api/auth/login`
- `POST /api/health/submit`
- `GET /api/health/history`
- `POST /api/predict`
- `GET /api/explain/<prediction_id>`
- `POST /api/simulate`
- `POST /api/upload`
- `POST /api/chain/storehash`
- `GET /api/chain/verify`
- `GET /api/dashboard`

## Blockchain Principle

HealthTwin never stores raw medical data on-chain. It stores `SHA-256(file_content)` and `SHA-256(prediction_json)` so the evidence can be verified without exposing private health information.

## Blockchain Modes

HealthTwin supports three evidence modes:

1. `local_demo`: no contract configured; the backend returns a synthetic local proof for UI demos.
2. MetaMask signing: the browser signs `storeRecord(...)` and the backend records the confirmed transaction hash.
3. Backend Web3.py signing: set `WEB3_PROVIDER`, `CONTRACT_ADDRESS`, `CHAIN_ID`, and `BLOCKCHAIN_PRIVATE_KEY` in `.env`; `BlockchainClient.store_hash()` submits a real smart-contract transaction from the backend signer.

For a local Hardhat proof:

```bash
cd blockchain
npm install
npm run node
npm run deploy:localhost
```

Then copy the deployed contract address and one funded Hardhat private key into `.env`.

## Disclaimer

This is a research and education prototype. It is not a certified medical device and must not be used for diagnosis or treatment decisions.
