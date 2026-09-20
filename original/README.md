# Original prototype (2022–2025)

These are the **authentic original files**, preserved for provenance. They are rough —
long functions, deep nesting, minimal structure — because they are exactly what the
project looked like as a self-taught prototype. For a clean, readable rendering of the
two model architectures, see [`../reference/`](../reference/).

The only modification made here: **hardcoded Alpaca API credentials were removed** and
replaced with environment-variable reads (see [`../.env.example`](../.env.example)).

| File | What it is |
|---|---|
| `multiconv_sentiment_classifier.py` | The original multi-kernel CNN sentiment classifier (Mar 2022), adapted from the multi-ConvNet pattern in Jon Krohn's *Deep Learning Illustrated*. |
| `bot_integrated.py` | The full live bot (latest, 2025): both models, news-driven universe discovery, the agreement-gating decision logic, bracket/breakeven execution, adaptive sizing, weekly Bayesian-tuned retraining, and a 4-process concurrent runtime. |

Other components existed in the original workspace (separate `News`, `Prices`, and
`Super Classifier` trainers, and earlier dated `bot_new_*` snapshots); they are omitted
here to keep the repository focused.
