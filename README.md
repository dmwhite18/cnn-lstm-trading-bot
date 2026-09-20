# CNN + LSTM Algorithmic Trading Bot

A self-directed machine-learning trading system (Python / TensorFlow-Keras / Alpaca
API) that combines **two neural networks which must agree before it trades**: a
convolutional network that reads financial-news text and predicts whether a stock
will rise, and a recurrent network that forecasts the stock's next-day price from its
own history. Prototyped in 2022 and iterated through 2025.

> **Portfolio / research project.** This ran against the Alpaca brokerage API (paper,
> and at times a live account). It is an experimental, self-taught project — not a
> production fund — and is presented here as an engineering and ML case study. No
> performance/alpha claim is made (see [Honest limitations](#honest-limitations)).

---

## Repository layout

| Path | What it is |
|---|---|
| [`reference/`](reference/) | **Clean, documented reference implementations** of the two models — the best place to read the architecture. |
| &nbsp;&nbsp;`reference/news_cnn.py` | Multi-kernel 1D CNN news classifier (`build_news_cnn`). |
| &nbsp;&nbsp;`reference/price_lstm.py` | Stacked Bidirectional-LSTM price forecaster (`build_price_lstm`). |
| [`original/`](original/) | The **authentic 2022–2025 prototype**, preserved as-is (credentials removed). Rough around the edges — kept for provenance. |
| &nbsp;&nbsp;`original/multiconv_sentiment_classifier.py` | The original CNN, adapted from Krohn's *Deep Learning Illustrated*. |
| &nbsp;&nbsp;`original/bot_integrated.py` | The full live bot: both models, universe discovery, execution, retraining, concurrency. |

---

## The two models

### 1. News sentiment — multi-kernel 1D CNN (`news_cnn.py`)

A convolutional **text classifier** over the content of Alpaca news articles.

- **Input:** article text → Keras `Tokenizer` → integer sequences → padded.
- **Architecture:** `Embedding` → `SpatialDropout1D` → **three parallel `Conv1D`
  branches** with different kernel sizes (2-, 3-, 4-gram patterns), each with
  `GlobalMaxPooling1D` → `concatenate` → `Dense` → `Dropout` → `Dense` →
  **`Dense(1, sigmoid)`**.
- **Objective:** `binary_crossentropy`. The label is whether price **actually rose**
  over a forward window after the news — so the network learns *news text →
  probability the stock goes up*, an outcome-grounded signal rather than generic tone.

### 2. Price forecasting — stacked Bidirectional LSTM (`price_lstm.py`)

A recurrent **regressor** over price history.

- **Input:** OHLCV history, `MinMaxScaler`-normalized, in rolling ~27-day windows.
- **Architecture:** `Bidirectional(LSTM) → Bidirectional(LSTM) → Dense(1)`.
- **Objective:** `mean_absolute_error`; predicts the **next day's opening price**.

---

## How they combine — agreement gating

The distinctive part is the decision logic: **both models must clear independent
quality bars, per ticker, in real time**, before any capital is committed.

1. **Discover a universe** — collect symbols co-tagged in recent Alpaca news; sample.
2. **CNN reliability gate** — rebuild the news→direction history for that ticker and
   require the CNN's **ROC-AUC ≥ 0.6** on it. *The news model must have been
   discriminative for this stock, not just on average.*
3. **CNN signal gate** — score the latest news; require **average predicted sentiment
   ≥ a learned threshold** (derived from the model's own score distribution).
4. **LSTM fit gate** — require the price model's **R² ≥ 0.9** on the ticker's recent
   actuals. *The price model must currently fit this stock well.*
5. **LSTM direction gate** — require the **predicted next-day open > current price**.
6. Only if **all** gates pass → the stock is queued to buy.

---

## Engineering around the models

- **Fault-tolerant data/API layer** — every Alpaca / Yahoo Finance call is wrapped in
  bounded retry-with-backoff loops.
- **Concurrent runtime** — four coordinated `multiprocessing` processes (`analyze`,
  market-`clock` watcher, `buying`, `selling`) sharing state via a `Manager`.
- **Execution** — bracket orders (take-profit at the LSTM's predicted price + a stop)
  when the account is large enough; market orders otherwise; a daily breakeven
  limit-sell exit; OTC vs listed handled with different time-in-force.
- **Adaptive position sizing** from account buying-power usage, persisted across
  restarts.
- **Automated weekly retraining** of both models; hyperparameters (kernel sizes,
  filters, embedding dim, LSTM units, learning rate, optimizer, windows, dropout)
  were tuned by **Bayesian optimization**.

---

## Tech stack

`Python` · `TensorFlow` / `Keras` (`Conv1D`, `Bidirectional LSTM`, functional &
sequential APIs) · `scikit-learn` (`MinMaxScaler`, `train_test_split`, ROC-AUC / R²) ·
`alpaca_trade_api` · `yfinance` · `numpy` · `multiprocessing` · Bayesian
hyperparameter optimization.

## Skills demonstrated

- Deep learning across **two architecture families** (CNN for text, RNN/LSTM for
  sequences) in one system.
- **NLP text classification** and **time-series forecasting**.
- **Model ensembling / decision gating** — combining heterogeneous models with
  explicit reliability checks.
- **Hyperparameter optimization** (Bayesian).
- **Real-time / concurrent systems** and **fault-tolerant API integration**.
- Quantitative-finance domain modeling (labeling, execution, risk via brackets/stops).

## Honest limitations

- A self-directed research project; results were **not** validated with the rigor of a
  held-out / walk-forward / cost-and-slippage gauntlet, so **no performance claim is
  made**.
- The per-ticker gates (AUC ≥ 0.6, R² ≥ 0.9) are computed on the same recent data used
  for the decision — a look-ahead risk a production version would need to remove.
- A high R² on a scaled price series can reflect a near-persistence ("tomorrow ≈
  today") fit rather than genuine predictive skill; treat the price model accordingly.

---

## Quickstart

```bash
pip install -r requirements.txt

# Read either model's architecture:
python reference/news_cnn.py      # prints the CNN summary
python reference/price_lstm.py    # prints the LSTM summary
```

To run the original bot you would supply Alpaca credentials via environment variables
(copy `.env.example` to `.env` and fill it in). **Use the paper endpoint.** The
original code is archived and unmaintained.

## Security

No API credentials are stored in this repository; the original hardcoded keys were
removed and replaced with environment-variable reads. Never commit real keys — `.env`
is git-ignored.

## License

MIT — see [`LICENSE`](LICENSE).
