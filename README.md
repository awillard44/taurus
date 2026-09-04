# Taurus

Experimental adaptive market intelligence and trading research system
built in Python.

Taurus is an early-stage project exploring how structured market data,
quantitative features, simulated trading, reinforcement learning, and
eventually LLM-assisted market intelligence can be combined into an
adaptive trading research system.

The project is designed around a layered approach: specialized
quantitative systems produce structured market information, learning
models consume controlled observations, and portfolio risk and execution
constraints remain outside the learning model wherever possible.

> **Status:** Early development. Taurus does not currently execute real
> trades or provide investment advice.

## Current Capabilities

### Market Data

- Retrieves historical market data through Yahoo Finance using
  `yfinance`
- Represents OHLCV observations as immutable `PriceBar` objects
- Supports explicit bar intervals so market data can be analyzed across
  different timeframes
- Currently supports intervals up to one month
- Persists market data locally using SQLite
- Separates stored observations by symbol, timestamp, source, and
  interval
- Supports multi-year historical datasets for evaluation across
  different market conditions

### Feature Engine

Taurus includes reusable quantitative calculations for:

- Returns
- Simple moving averages (SMA)
- Exponential moving averages (EMA)
- Volatility
- Relative Strength Index (RSI)
- Volume ratio
- Relative return versus a benchmark
- Average True Range (ATR)
- Bollinger Bands
- MACD
- Stochastic oscillator
- Rolling VWAP
- On-Balance Volume (OBV)
- ADX and Directional Movement indicators

Feature calculations operate on generic periods rather than being
permanently tied to daily data.

Taurus also includes a configurable feature framework built around
`IndicatorConfig` and `FeatureSetConfig`. This allows experiments to
select and parameterize groups of quantitative indicators without
hard-coding a single observation structure.

The current default feature preset includes:

- 1-period return
- 5-period return
- 20-period SMA
- 14-period RSI
- 14-period ATR
- 20-period volume ratio
- 20-period relative return versus SPY
- 14-period ADX
- 14-period +DI
- 14-period -DI

### Market State

Taurus can construct market states from stored historical data while
validating:

- Symbol consistency
- Bar intervals
- Required history
- Benchmark alignment
- Timestamp alignment

Feature-state sequences are built chronologically without using future
observations.

Historical context can also be separated from the scored evaluation
period. This allows indicators to warm up on prior observations without
forcing the first portion of each evaluation window to be discarded.

### Trading Simulation

Taurus includes a simulated trading layer with:

- Portfolio cash, shares, asset price, and total value
- Buy, sell, and hold actions
- Simulated trade execution
- Portfolio revaluation
- Trade records
- Closed-trade tracking
- Partial position tracking
- Commission modeling
- Slippage modeling

Execution costs are configurable so strategies can be evaluated under
more realistic assumptions than frictionless trading.

### Gymnasium Trading Environment

Taurus includes a Gymnasium-compatible trading environment.

The environment:

- Receives chronological feature states
- Exposes configurable market features to agents
- Includes normalized portfolio information in observations
- Supports dynamic observation sizes based on the selected feature set
- Executes portfolio actions through the simulation layer
- Advances through historical market states sequentially
- Produces rewards based on portfolio performance

Market features and portfolio values are normalized before being
presented as model observations where appropriate.

### Baseline Agents

Taurus includes simple agents used to validate the environment and
provide benchmarks for future learning models:

- Always Hold
- Always Buy
- Momentum
- Random

The Random baseline supports explicit seeds for reproducible evaluation
runs.

These agents are intentionally simple. Their purpose is to verify the
simulation and establish performance baselines rather than serve as
production trading strategies.

### Evaluation

Taurus includes an evaluation framework for running individual episodes
and repeated baseline experiments.

Current metrics include:

- Final portfolio value
- Total return
- Maximum drawdown
- Portfolio volatility
- Sharpe ratio
- Sortino ratio
- Total reward
- Trade count
- Win rate
- Average gain
- Average loss
- Profit factor

Repeated runs can also report:

- Mean final portfolio value
- Mean total reward
- Reward standard deviation

### Historical Validation

The current environment has been validated using:

- NVDA
- AAPL
- MSFT
- AMZN

with SPY used as a benchmark.

Historical validation currently spans annual evaluation windows from
2022 through 2026 year-to-date.

Each full-year evaluation uses preceding historical observations as
feature context while restricting trading and scoring to the actual
evaluation period.

This allows the same feature configuration, execution-cost assumptions,
portfolio logic, and evaluation metrics to be tested across multiple
assets and materially different historical conditions.

```text
Yahoo Finance
      |
      v
  PriceBars
      |
      v
    SQLite
      |
      v
Feature Engine
      |
      v
Feature States
      |
      v
Normalization
      |
      v
Gymnasium Environment
      |
      v
Trading Agent
      |
      v
Simulation / Execution
      |
      v
Evaluation Metrics
```

## Project Structure

```text
taurus/
├── scripts/
│   ├── ingest_history.py
│   ├── run_baseline_evaluation.py
│   └── show_market_state.py
├── src/
│   └── taurus/
│       ├── data/
│       │   ├── providers/
│       │   ├── ingestion.py
│       │   ├── provider.py
│       │   ├── schemas.py
│       │   └── sqlite_repository.py
│       ├── environment/
│       │   ├── feature_state.py
│       │   ├── feature_state_builder.py
│       │   ├── normalization.py
│       │   ├── observation.py
│       │   └── trading_environment.py
│       ├── evaluation/
│       │   ├── baseline_runner.py
│       │   └── metrics.py
│       ├── features/
│       │   ├── config.py
│       │   ├── executor.py
│       │   ├── feature_set_executor.py
│       │   ├── market_state.py
│       │   ├── presets.py
│       │   ├── registry.py
│       │   ├── state_builder.py
│       │   └── indicator modules
│       ├── models/
│       │   └── baselines/
│       └── simulation/
│           ├── actions.py
│           ├── costs.py
│           ├── execution.py
│           ├── portfolio.py
│           ├── step.py
│           └── trade tracking
├── tests/
│   └── unit/
├── pyproject.toml
└── README.md
```

## Development

Taurus uses Python 3.11 or newer.

Install the project in editable mode with development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Run the test suite:

```bash
pytest -q
```

At the `v0.2.0` milestone, the project contains 212 passing unit tests
covering market-data schemas, Yahoo data conversion, SQLite persistence,
ingestion, quantitative features, configurable feature execution,
market-state construction, simulation, execution costs, trade tracking,
observation normalization, the Gymnasium environment, baseline agents,
evaluation metrics, and input validation.

## Example Data Pipeline

Historical daily bars for the current validation symbols and SPY can be
ingested with:

```bash
python scripts/ingest_history.py
```

A market state can be generated from stored observations with:

```bash
python scripts/show_market_state.py
```

Historical baseline evaluation can be run with:

```bash
python scripts/run_baseline_evaluation.py
```

## Roadmap

### v0.3.0 — Initial Learning Experiments

The next development phase will introduce Taurus's first trainable
learning agent.

Planned work includes:

1. Introduce an initial reinforcement-learning agent
2. Establish chronological training, validation, and test splits
3. Train only on explicitly approved feature configurations
4. Compare learned behavior against deterministic and random baselines
5. Evaluate performance across multiple assets and historical regimes
6. Improve experimental reproducibility and model versioning
7. Test alternative feature sets using controlled experiments and
   ablation studies

Risk controls and execution constraints should remain independent of the
learning model rather than becoming behaviors the model is expected to
discover for itself.

### Longer-Term Research

Future work may include:

- Additional market-data providers
- Broader asset coverage
- Multiple market timeframes
- Regime and market-environment modeling
- Portfolio-level learning
- Position sizing
- Liquidity and execution modeling
- More realistic transaction-cost models
- Supervised predictive models
- Additional reinforcement-learning architectures
- Walk-forward and out-of-sample evaluation
- Model promotion and deployment governance
- Structured earnings, filings, news, and macroeconomic context
- LLM-assisted transformation of unstructured market information into
  structured, testable features
- Paper trading and shadow evaluation
- Eventual broker integration only after substantial validation and
  explicit risk controls

The long-term goal is not to make a language model or reinforcement
learning agent responsible for every trading decision. Taurus is intended
to combine specialized quantitative systems, controlled learning models,
market context, independent risk governance, and realistic execution as
separate components that can be evaluated individually.

## Disclaimer

Taurus is an experimental software and machine-learning research
project. It is not financial advice and should not be relied upon for
real-world investment decisions or automated trading without substantial
additional development, validation, and risk controls.