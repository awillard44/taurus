from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class DateRange:
    start: date
    end: date

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError("Date range start must not be after end.")


@dataclass(frozen=True)
class TrainingExperimentConfig:
    name: str
    symbol: str
    benchmark_symbol: str
    training: DateRange
    validation: DateRange
    test: DateRange
    seed: int

    def __post_init__(self) -> None:
        if self.training.end >= self.validation.start:
            raise ValueError("Training period must end before validation begins.")

        if self.validation.end >= self.test.start:
            raise ValueError("Validation period must end before test begins.")