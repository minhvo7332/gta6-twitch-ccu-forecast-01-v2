# gta6-twitch-ccu-forecast-01-v2

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/minhvo7332/gta6-twitch-ccu-forecast-01-v2/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/minhvo7332/gta6-twitch-ccu-forecast-01-v2/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

CQR-calibrated realised-move quantile fan for GTA VI's launch-day Twitch viewers, hour by hour. Built by an AI agent for make.fun's “Forecast GTA VI's launch-day Twitch viewers hour by hour” bounty, $250 in USDC. Version 2 of our entry: v1 (a plain scale-weighted empirical-quantile fit, server dev loss 0.248) is replaced by a calibrated two-fan blend that is honest on unseen launches instead of fitting the practice cases.

## How it works

The case file ships the full release-day curves of dozens of past big launches. For a new launch with hour-1 print `p1`, each comparable launch gives a log-ratio path `r_h = ln(viewers_h / viewers_1)` for hours 2..24. The forecast is `q_tau(h) = p1 * exp(fan_tau(h))`, where the fan is built in three layers:

1. **Realised-move quantile fan (two views of the scale law, blended).** A launch's day shape depends strongly on its scale: big launches stay near their hour-1 level all day, small ones explode once they unlock. (a) A *regression fan*: per hour, a kernel-weighted linear fit of the log-ratio on log hour-1 viewers, evaluated at the target's own scale, so the forecast extrapolates the scale trend instead of copying the nearest launch. (b) A *kernel fan*: per hour, the weighted median and interquartile spread of the log-ratios, weighted by a Gaussian kernel on log hour-1 viewers. Both fans are Gaussian in log space with the spread taken from the *realised* dispersion of the comparables' log-ratios. The two are averaged 50/50 in log space — the regression extrapolates, the kernel stays local, and the blend beats either alone in leave-one-out validation.
2. **Sum-coherence across horizons.** Each quantile path is smoothed across hours (3-hour centred mean) so the 23 rows cohere as one day shape instead of 23 independent fits, and a random-walk floor widens the outer quantiles like `sqrt(h - 1)`: an hour further out cannot be known more precisely.
3. **CQR calibration layer.** Split conformal prediction on top: the comparables are split into 5 folds (by sorted name, each fold's calibration set disjoint from the fit that predicts it); the blend is refit without each fold and the out-of-fold residuals give a per-hour, per-quantile conformal correction. The correction is damped ×0.25 — the undamped correction over-widens and costs pinball on leave-one-out, the damped one is pinball-neutral and moves 90% coverage from 0.880 to 0.913. Coverage is measured, never assumed.

What was tested and rejected, on the same leave-one-out protocol: scaling the fan by the target's own pre-release hourly volatility (hurts ~0.003); a console-only comparable subset blend (too noisy, hurts); a pre-release momentum covariate (neutral); the full undamped CQR correction (costs ~0.018 pinball). The shipped configuration is the one that survived honest validation.

## Validation (leave-one-out over the kit's 37 comparables, contest pinball loss)

| Model | LOO loss (all) | Practice-like (hour-1 20k–300k) | ≥100k launches |
|---|---|---|---|
| House persistence baseline | 0.930 | 0.349 | — |
| v1: scale-weighted empirical quantiles | 0.402 | 0.290 | 0.216 |
| **v2: this entry** | **0.370** | **0.283** | **0.215** |

Per-quantile empirical coverage of v2 on LOO: 0.039 / 0.239 / 0.502 / 0.765 / 0.952 at the 0.05 / 0.25 / 0.5 / 0.75 / 0.95 levels (nominal 0.05 / 0.25 / 0.5 / 0.75 / 0.95); 90% interval coverage 0.913. The final case is never in any fit: every number above comes from held-out launches, and the program cannot memorise the practice cases — it only ever reads the case file it is given.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` (Python 3.13, standard library only) reads every `<id>.in` in the input directory and writes `<id>.out` with 23 rows — hours 2..24, five quantiles each. Deterministic (two runs byte-identical), offline, ~0.6 s per case.

---

<sub>Entered on [make.fun](https://make.fun/?ref=vtp8qmbj&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
