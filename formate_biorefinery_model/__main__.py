from __future__ import annotations

from . import format_summary, format_validation_report, run_baseline_cases, run_validation_suite


def main() -> None:
    print(format_summary(run_baseline_cases()))
    print()
    print(format_validation_report(run_validation_suite()))


if __name__ == "__main__":
    main()
