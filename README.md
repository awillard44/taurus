# Taurus

Experimental adaptive market intelligence and trading research system
built in Python.

Taurus is an early-stage project exploring how structured market data,
technical features, simulated trading, reinforcement learning, and
eventually LLM-assisted market intelligence can be combined into an
adaptive trading research system.

> **Status:** Early development. Taurus does not currently execute real
> trades or provide investment advice.

## Current Capabilities

### Market Data

-   Retrieves historical market data through Yahoo Finance using
    `yfinance`
-   Represents OHLCV observations as immutable `PriceBar` objects
-   Supports explicit bar intervals so market data can be analyzed
    across different timeframes
-   Currently supports intervals up to one month
-   Persists market data locally using SQLite
-   Separates stored observations by symbol, timestamp, source, and
    interval

### Feature Engine

Taurus includes reusable, period-based calculations for:

-   Returns
-   Simple moving averages (SMA)
-   Volatility
-   Relative Strength Index (RSI)
-   Volume ratio
-   Relative return versus a benchmark

Feature calculations operate on generic periods rather than being
permanently tied to daily data. The same calculations can therefore be
applied to different bar intervals.

### Market State

Taurus can combine stored market data into an immutable `MarketState`
containing a snapshot of an asset's current technical state, including:

-   1-period return
-   5-period return
-   20-period return
-   20-period volatility
-   20-period SMA
-   50-period SMA
-   14-period RSI
-   Volume ratio
-   Relative return versus a benchmark

The current pipeline has been tested with real NVDA market data using
SPY as the benchmark.

``` text
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
 MarketState
```

The state builder validates symbol consistency, bar intervals, required
history, benchmark intervals, and latest timestamp alignment before
constructing a market state.

## Project Structure

``` text
taurus/
├── scripts/
│   ├── ingest_history.py
│   └── show_market_state.py
├── src/
│   └── taurus/
│       ├── data/
│       │   ├── providers/
│       │   │   └── yahoo.py
│       │   ├── ingestion.py
│       │   ├── provider.py
│       │   ├── schemas.py
│       │   └── sqlite_repository.py
│       └── features/
│           ├── market_state.py
│           ├── moving_average.py
│           ├── relative_return.py
│           ├── returns.py
│           ├── rsi.py
│           ├── state_builder.py
│           ├── volatility.py
│           └── volume.py
├── tests/
│   └── unit/
├── pyproject.toml
└── README.md
```

## Development

Taurus uses Python 3.11 or newer.

Install the project in editable mode with development dependencies:

``` bash
python -m pip install -e ".[dev]"
```

Run the test suite:

``` bash
pytest -q
```

The project currently maintains unit tests covering market-data schemas,
Yahoo data conversion, SQLite persistence, ingestion, feature
calculations, interval handling, market-state construction, and input
validation.

## Example Data Pipeline

Historical NVDA and SPY daily bars can be ingested with:

``` bash
python scripts/ingest_history.py
```

A market state can then be generated from the stored observations with:

``` bash
python scripts/show_market_state.py
```

## Roadmap

The next development phase will introduce the simulated trading
environment:

1.  Portfolio state
2.  Buy, sell, and hold actions
3.  Simulated trade execution
4.  Portfolio valuation
5.  Trading environment
6.  Reward-function design
7.  Reinforcement-learning experimentation

Future work may also include additional data providers, broader feature
engineering, multiple market timeframes, benchmark selection, risk
controls, model evaluation, and LLM-assisted market context.

## Disclaimer

Taurus is an experimental software and machine-learning research
project. It is not financial advice and should not be relied upon for
real-world investment decisions or automated trading without substantial
additional development, validation, and risk controls.
