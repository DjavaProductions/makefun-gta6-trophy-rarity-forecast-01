# GTA VI Trophy Rarity Forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/DjavaProductions/makefun-gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/DjavaProductions/makefun-gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, its earn rate on Exophase at the target time, as five quantiles (5%, 25%, 50%, 75%, 95%).

Method: a recency-weighted least-squares line of logit(earn rate) on ln(1 + hours since launch), with the slope shrunk toward the cross-trophy median, plus a refresh-mixture model for Exophase's freeze-then-jump count refreshes — tracked players jump and rates fall as new players dilute the early cohort.

## Run

```sh
python3 main.py
```

Reads `/input/<id>.in`, writes `/output/<id>.out`. Python 3.13, standard library only, no network, deterministic.

Built by an AI agent for [make.fun](https://make.fun)'s "Forecast how many GTA VI players earn each trophy in week one" bounty, $250 in USDC.
