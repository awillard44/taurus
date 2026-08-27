from dataclasses import dataclass, field


@dataclass(frozen=True)
class IndicatorConfig:
    # Configuration for one calculated market indicator
    key: str
    indicator: str
    inputs: tuple[str, ...]
    parameters: dict[str, int | float | str] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class FeatureSetConfig:
    # Named collection of indicators used by an experiment
    name: str
    indicators: tuple[IndicatorConfig, ...]