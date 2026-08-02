# Data and model provenance

The shipped Dota 7.41 model bundles were trained from public ranked-match
metadata retrieved through OpenDota and STRATZ. The application does not call
either API during normal use and contains no API key.

Only derived model parameters and aggregate evaluation reports are published.
This repository intentionally excludes:

- raw OpenDota or STRATZ responses;
- the normalized per-match training database;
- account identifiers and player profiles;
- paid-API request logs and local credentials.

The model is an observational recommender. A historical association between a
recommended choice and a match result is not proof that the recommendation
caused the result. See `artifacts/models/BENCHMARK.md` and the JSON reports in
each model bundle for the exact held-out metrics and uncertainty intervals.

