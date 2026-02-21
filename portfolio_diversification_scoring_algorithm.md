# Portfolio Diversification Scoring Algorithm

## Overview

A client's portfolio receives a single **Diversification Score** on a 1–10 scale, composed of four sub-scores with the following weights:

| Sub-Score | Weight |
|---|---|
| Asset Class Score | 40% |
| Geography Score | 30% |
| Sector Score | 20% |
| Concentration Score | 10% |

$$\text{Diversification Score} = 0.40 \times S_{\text{asset}} + 0.30 \times S_{\text{geo}} + 0.20 \times S_{\text{sector}} + 0.10 \times S_{\text{conc}}$$

Each sub-score is on a 1–10 scale (or 0–10 where noted), and the final composite score therefore also falls on a 1–10 scale.

---

## Algorithm 1: Asset Class Score (Weight: 40%)

### Purpose

Measures whether the portfolio is spread across multiple asset classes (Equity, Bonds, Real Assets, Cash) and whether any single asset class dominates excessively.

### Inputs

- A list of asset class weights $w_1, w_2, \ldots, w_k$ that sum to 1 (or 100%).
- The asset classes are: **Equity, Bonds, Real Assets, Cash**.

### Part A: Breadth Score

**STEP 1** — Filter out insignificant allocations:

Remove any asset class with weight $w_i < 0.05$ (less than 5%).

**STEP 2** — Count remaining asset classes:

Let $m$ = number of asset classes remaining after filtering.

**STEP 3** — Assign base breadth score:

$$B_{\text{base}} = \begin{cases} 1 & \text{if } m = 1 \\ 4 & \text{if } m = 2 \\ 6 & \text{if } m = 3 \\ 8 & \text{if } m \geq 4 \end{cases}$$

**STEP 4** — Find the smallest weight among the remaining (meaningful) asset classes:

$$w_{\min} = \min_{i \in \text{remaining}} w_i$$

**STEP 5** — Compute the uplift:

$$U = \begin{cases} 0 & \text{if } w_{\min} < 0.10 \\ 1 + \dfrac{w_{\min} - 0.10}{0.05} & \text{if } 0.10 \leq w_{\min} < 0.15 \\ 2 & \text{if } w_{\min} \geq 0.15 \end{cases}$$

Note: The middle case is a linear interpolation from 1 to 2 over the interval $[0.10, 0.15]$.

**STEP 6** — Breadth Score:

$$S_{\text{breadth}} = B_{\text{base}} + U$$

This yields a value in $[1, 10]$.

### Part B: Dominance Score

**STEP 7** — Find the largest asset class weight:

$$w_{\max} = \max(w_1, w_2, \ldots, w_k)$$

**STEP 8** — Compute the dominance score:

$$S_{\text{dominance}} = \begin{cases} 10 & \text{if } w_{\max} \leq 0.70 \\ 10 - 6 \times \dfrac{w_{\max} - 0.70}{0.20} & \text{if } 0.70 < w_{\max} \leq 0.90 \\ 4 - 3 \times \dfrac{w_{\max} - 0.90}{0.10} & \text{if } w_{\max} > 0.90 \end{cases}$$

- Linearly declines from 10 to 4 over the interval $(0.70, 0.90]$.
- Linearly declines from 4 to 1 over the interval $(0.90, 1.00]$.

### Part C: Final Asset Class Score

$$S_{\text{asset}} = 0.50 \times S_{\text{breadth}} + 0.50 \times S_{\text{dominance}}$$

Output: a value in $[1, 10]$.

---

## Algorithm 2: Geography Score (Weight: 30%)

### Purpose

Measures how well the portfolio's geographic allocation matches or covers a benchmark. Overweighting a region is not penalised (capped at full marks for that region), but underweighting is penalised proportionally.

### Inputs

- Client regional weights: $c_1, c_2, \ldots, c_R$ (summing to 1).
- Benchmark regional weights: $b_1, b_2, \ldots, b_R$ (summing to 1).
- Total points available: $T = 10$.

The default benchmark regions and weights are:

| Region | Benchmark Weight |
|---|---|
| United States | 0.65 |
| Developed ex-US | 0.25 |
| Emerging Markets | 0.10 |

### Algorithm

**STEP 1** — For each region $r$, compute the maximum available points:

$$P_r^{\max} = b_r \times T$$

**STEP 2** — For each region $r$, compute the region's contribution:

$$P_r = P_r^{\max} \times \min\!\left(1,\; \frac{c_r}{b_r}\right)$$

If the client meets or exceeds the benchmark weight, they get full marks for that region. If they underweight, the score is proportionally reduced.

**STEP 3** — Sum the contributions:

$$S_{\text{geo}} = \sum_{r=1}^{R} P_r$$

Output: a value in $[0, 10]$.

### Worked Example

Benchmark: $[0.65, 0.25, 0.10]$. Client: $[0.60, 0.30, 0.10]$.

- US: $6.5 \times \min(1, 0.60/0.65) = 6.5 \times 0.923 = 6.0$
- Dev ex-US: $2.5 \times \min(1, 0.30/0.25) = 2.5 \times 1.0 = 2.5$
- EM: $1.0 \times \min(1, 0.10/0.10) = 1.0 \times 1.0 = 1.0$
- **Total: 9.5 / 10**

---

## Algorithm 3: Sector Score (Weight: 20%)

### Purpose

Measures how well the portfolio's sector exposure covers its relevant benchmark. Uses a benchmark-gap approach identical in structure to the Geography Score, but applied to industry sectors.

### Inputs

- Client sector weights: $s_1, s_2, \ldots, s_K$ (normalised to sum to 100).
- Benchmark sector weights: $\hat{s}_1, \hat{s}_2, \ldots, \hat{s}_K$ (from a reference index).
- The benchmark depends on the portfolio type:
  - **Equity portfolios** → GICS sectors, benchmarked against ACWI.
  - **Fixed Income portfolios** → BICS sectors, benchmarked against BNDW (Vanguard Total World Bond ETF).
  - **REIT portfolios** → REIT sub-sector classification, benchmarked against the SRT SP EQ index.

### Reference Benchmarks

**Equity Benchmark (ACWI — GICS sectors):**

| Sector | Benchmark Weight (%) |
|---|---|
| Information Technology | 26.27 |
| Financials | 17.41 |
| Industrials | 11.33 |
| Consumer Discretionary | 9.80 |
| Health Care | 8.90 |
| Communication Services | 8.71 |
| Consumer Staples | 5.37 |
| Materials | 4.03 |
| Energy | 3.77 |
| Utilities | 2.58 |
| Real Estate | 1.78 |
| Not Classified | 0.04 |

**Fixed Income Benchmark (BNDW — BICS sectors):**

| Sector | Benchmark Weight (%) |
|---|---|
| Government | 64.77 |
| Financials | 10.94 |
| Not Classified | 9.55 |
| Utilities | 2.53 |
| Health Care | 1.94 |
| Industrials | 1.84 |
| Communications | 1.58 |
| Consumer Discretionary | 1.46 |
| Energy | 1.31 |
| Consumer Staples | 1.27 |
| Technology | 1.20 |
| Real Estate | 0.99 |
| Materials | 0.62 |

**REIT Benchmark (SRT SP EQ):**

| Sub-Sector | Benchmark Weight (%) |
|---|---|
| Industrial REITs | 38.16 |
| Diversified REITs | 22.85 |
| Specialized REITs | 13.60 |
| Retail REITs | 10.23 |
| Office REITs | 5.79 |
| Hotel & Resort REITs | 3.93 |
| Health Care REITs | 2.46 |
| Data Center REITs | 1.86 |
| Real Estate Operating Companies | 1.12 |

### Algorithm

For each sector $k$:

$$\text{contribution}_k = \min(s_k,\; \hat{s}_k)$$

That is, the client receives credit up to the benchmark weight for each sector. Overweighting beyond the benchmark earns no additional credit.

$$S_{\text{sector}}^{\text{raw}} = \sum_{k=1}^{K} \text{contribution}_k$$

This raw score is already on a 0–100 scale (since both the client weights and benchmark weights are percentages summing to 100). To convert to a 0–10 scale:

$$S_{\text{sector}} = \frac{S_{\text{sector}}^{\text{raw}}}{10}$$

### Handling Multi-Portfolio Clients (Growth + Income Split)

When a client holds a mix of equity (growth) and fixed income (income) portfolios, the sector score is computed separately for each type and then combined via a weighted average:

1. Compute $S_{\text{sector}}^{\text{growth}}$ using the equity benchmark for the equity portion.
2. Compute $S_{\text{sector}}^{\text{income}}$ using the fixed income benchmark for the income portion.
3. Let $w_g$ = total client weight in growth portfolios, $w_i$ = total client weight in income portfolios.

$$S_{\text{sector}} = \frac{S_{\text{sector}}^{\text{growth}} \times w_g + S_{\text{sector}}^{\text{income}} \times w_i}{w_g + w_i}$$

The result is then scaled to a 0–10 output.

---

## Algorithm 4: Concentration Score (Weight: 10%)

### Purpose

Measures how concentrated the portfolio is at the individual holding level, using the Herfindahl–Hirschman Index (HHI) and the concept of Effective Number of Stocks.

### Inputs

- A list of individual holding weights: $w_1, w_2, \ldots, w_n$ (summing to 1).
- A target number of holdings: $N_{\text{target}}$ (default: **1000**).

### Algorithm

**STEP 1** — Compute the Herfindahl–Hirschman Index (HHI):

$$\text{HHI} = \sum_{i=1}^{n} w_i^2$$

This is the sum of squared weights across all holdings.

**STEP 2** — Compute the Effective Number of Stocks:

$$N_{\text{eff}} = \frac{1}{\text{HHI}}$$

This represents the number of equally-weighted holdings that would produce the same concentration level. A perfectly equal-weight portfolio of $n$ holdings gives $N_{\text{eff}} = n$.

**STEP 3** — Compute the raw score using a logarithmic scale relative to the target:

$$S_{\text{raw}} = \min\!\left(1,\; \frac{\ln(N_{\text{eff}})}{\ln(N_{\text{target}})}\right)$$

The logarithmic scaling ensures:
- Diminishing marginal benefit as $N_{\text{eff}}$ increases.
- Smooth, monotonically increasing score.
- The score is capped at 1 when $N_{\text{eff}} \geq N_{\text{target}}$.

**STEP 4** — Scale to a 1–10 range:

$$S_{\text{conc}} = \max\!\left(1,\; \text{round}(S_{\text{raw}} \times 10,\; 2)\right)$$

The floor of 1 ensures no portfolio scores below 1.

### Key Properties

- An equal-weight portfolio of $N_{\text{target}}$ holdings scores exactly 10.
- A single-holding portfolio ($n = 1$) scores 1 (the floor).
- The logarithmic scaling means going from 10 to 100 holdings improves the score more than going from 1000 to 1090 holdings.

### Worked Examples (from the spreadsheet)

| Portfolio | $n$ | Max Holding | HHI | $N_{\text{eff}}$ | Score |
|---|---|---|---|---|---|
| 10 equal-ish holdings | 10 | 16.9% | 0.1168 | 8.56 | 3.11 |
| 50 holdings | 50 | 3.8% | 0.0271 | 36.93 | 5.22 |
| 100 holdings | 100 | 1.8% | 0.0130 | 76.76 | 6.28 |
| 500 holdings | 500 | 0.4% | 0.0027 | 371.04 | 8.56 |
| 1000 holdings | 1000 | 0.2% | 0.0013 | 763.00 | 9.61 |
| 2000 holdings | 2000 | 0.1% | 0.0007 | 1522.12 | 10.00 |
| 1000 holdings (1 at 50%) | 1000 | 50.0% | 0.2503 | 4.00 | 2.01 |

### Note on $N_{\text{target}}$

The spreadsheet notes suggest using $N_{\text{target}} = 1000$ as the default. The rationale is that the logarithmic penalty term $\frac{\ln(n)}{\ln(N_{\text{target}})}$ provides smoother growth with a larger target. This parameter is tuneable.

### Alternative Entropy-Based Formulation (from the "Concentration" sheet)

The spreadsheet also contains an alternative entropy-based concentration scoring method. This was exploratory and the HHI method (above) appears to be the chosen approach for the "Concentration w examples" sheet. For completeness:

1. Compute Shannon entropy: $H = -\sum_{i=1}^{n} w_i \ln(w_i)$ (treating $0 \times \ln(0) = 0$).
2. Compute perfect entropy for $n$ holdings: $H^* = n \times \frac{1}{n} \ln\!\left(\frac{1}{n}\right) \times (-1) = \ln(n)$.
3. Compute penalty: $\text{penalty} = \min\!\left(1,\; \frac{\ln(n)}{\ln(N_{\text{target}})}\right)$.
4. Raw score: $S = \frac{H}{H^*} \times \text{penalty}$.
5. Final score: $\max(1,\; \text{round}(S \times 10,\; 2))$.

The HHI method is simpler and is the one used in the final client scoring examples.

---

## Multi-Portfolio Client Handling

A client typically holds multiple Syfe portfolios (e.g. "Equity Core 50%, Income Enhance 30%, REIT100 10%, Cash+ 10%"). To compute the diversification score:

1. **Look through** each portfolio to its underlying holdings.
2. **Weight each underlying holding** by: $\text{final\_weight}_i = \text{weight\_in\_portfolio}_i \times \text{client\_allocation\_to\_portfolio}$.
3. **Aggregate** across all portfolios to get a single list of holdings with final weights.
4. **Handle duplicates**: If the same ticker appears in multiple portfolios, sum its final weights.
5. Apply each of the four algorithms to the aggregated portfolio.

### Asset Class Assignment

Each portfolio (or each holding) maps to one of the four asset classes:

- **Equity**: Equity Core, Equity ESG, thematic equity portfolios (China Growth, Disruptive Tech, etc.)
- **Bonds**: Income Enhance, fixed income funds.
- **Real Assets**: REIT100, REIT with Risk Management, Gold.
- **Cash**: Cash+ (money market funds, enhanced liquidity funds).

### Geographic Assignment

Each holding maps to a region (US, Developed ex-US, Emerging Markets). The geographic weights are the look-through weights at the portfolio level.

### Sector Assignment

Each holding maps to an industry sector via GICS (equity) or BICS (fixed income). The portfolio-level sector weights are the normalised, look-through weights.

---

## Summary of Scoring Pipeline

```
Client Portfolio Allocations
    │
    ├─► Look-through to individual holdings (with final weights)
    │
    ├─► Asset Class Score (40%)
    │     ├─ Breadth Score (filter < 5%, count classes, uplift)
    │     ├─ Dominance Score (penalise largest class > 70%)
    │     └─ Average of breadth + dominance
    │
    ├─► Geography Score (30%)
    │     └─ Benchmark-gap: min(1, client_wt / benchmark_wt) per region
    │
    ├─► Sector Score (20%)
    │     └─ Benchmark-gap: min(client_wt, benchmark_wt) per sector
    │        (separate benchmarks for equity / fixed income / REIT)
    │
    └─► Concentration Score (10%)
          └─ HHI → Neff → ln(Neff) / ln(N_target) → scale to 1–10
```

$$\boxed{S_{\text{diversification}} = 0.40 \times S_{\text{asset}} + 0.30 \times S_{\text{geo}} + 0.20 \times S_{\text{sector}} + 0.10 \times S_{\text{conc}}}$$
