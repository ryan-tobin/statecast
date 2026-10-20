# Mathematics

The statistical model behind statecast and the methods used to evaluate it.
Each section names the module that implements it. Column and table names refer
to [data_dictionary.md](data_dictionary.md); results are in
[injury-study.md](injury-study.md).

## 1. Notation

| Symbol | Meaning |
| --- | --- |
| $i$ | Player |
| $k$ | The player's $k$-th appearance (game), at time $t_k$ in days |
| $\Delta_k = t_k - t_{k-1}$ | Days since the previous appearance ($\Delta_1 = 0$) |
| $y_{kj}$ | Metric on pitch $j$ of appearance $k$ |
| $n_k$, $\bar y_k$, $S_k$ | Pitch count, mean and within-game sum of squares |
| $\theta(t)$ | Talent: the player's true mean for the metric at time $t$ |
| $\mu$ | The player's long-run level |
| $g(t)$ | League seasonal shift |
| $\kappa, \sigma, r, m, s, \tau$ | Population parameters (Section 3) |
| $v = \sigma^2 / (2\kappa)$ | Stationary variance of talent around the level |

## 2. Sufficient statistics

`features/appearance.py`, `models/kalman.py`

Within appearance $k$, pitches are independent draws around that game's mean
$\eta_k$:

$$
y_{kj} \mid \eta_k \sim N(\eta_k, r^2), \qquad j = 1, \dots, n_k
$$

Since $\sum_j (y_{kj} - \eta_k)^2 = S_k + n_k (\bar y_k - \eta_k)^2$, the joint
density factors into the density of the mean and a term free of $\eta_k$:

$$
\prod_{j} \phi(y_{kj}; \eta_k, r^2)
= \phi\!\left(\bar y_k; \eta_k, \tfrac{r^2}{n_k}\right)
\cdot \underbrace{(2\pi r^2)^{-(n_k - 1)/2}\, n_k^{-1/2}\, e^{-S_k / (2 r^2)}}_{\text{residual term}}
$$

So $(n_k, \bar y_k, S_k)$ carry all the information in a game's pitches, and
the model never touches individual pitches. The residual term identifies $r$;
its log is

$$
\ell^{\text{res}}_k = -\tfrac{n_k - 1}{2} \log(2\pi r^2) - \frac{S_k}{2 r^2} - \tfrac12 \log n_k .
$$

## 3. The model

`models/dynamics.py`, `models/kalman.py`, `models/seasonal.py`

**Level.** Each player has a constant long-run level drawn from the population:

$$
\mu_i \sim N(m, s^2)
$$

**Talent.** Talent follows an Ornstein-Uhlenbeck process that reverts to the
player's own level:

$$
d\theta_i(t) = -\kappa\,(\theta_i(t) - \mu_i)\,dt + \sigma\,dW_i(t)
$$

**Game-day shock.** Each game adds a shock $\delta_k \sim N(0, \tau^2)$ shared by
all its pitches (rest, weather, adrenaline), which does not carry over to the
next game.

**Seasonal curve.** A league-wide shift $g(t)$, piecewise linear in the day of
the season with knots on the first of April through October. It is anchored
at $g(\text{July 1}) = 0$ and held flat outside the knots. Its six free knot
values are parameters.

Putting these together, $\eta_k = \theta(t_k) + g(t_k) + \delta_k$, and
integrating out $\delta_k$:

$$
\bar y_k \mid \theta(t_k) \sim N\!\left(\theta(t_k) + g(t_k),\; \tau^2 + \frac{r^2}{n_k}\right)
$$

The filter works on season-adjusted means $\tilde y_k = \bar y_k - g(t_k)$, so
talent is quoted as its July 1 equivalent.

## 4. State-space form

**State.** $x = [\theta, \mu]^\top$, observed through $H = [1, 0]$ with
measurement variance $R_k = \tau^2 + r^2 / n_k$.

**Exact transition.** The OU process has a closed-form transition over any gap,
so irregular schedules and off-seasons need no approximation. With
$d = e^{-\kappa \Delta}$:

$$
\Phi(\Delta) = \begin{bmatrix} d & 1 - d \\ 0 & 1 \end{bmatrix}, \qquad
Q(\Delta) = \begin{bmatrix} \dfrac{\sigma^2 (1 - d^2)}{2\kappa} & 0 \\ 0 & 0 \end{bmatrix}
$$

$1 - d$ and $1 - d^2$ are computed with `expm1`, which stays accurate when
$\kappa \Delta$ is small.

**Prior.** At a player's first appearance the level is $N(m, s^2)$ and talent
is the level plus a stationary OU deviation:

$$
x_0 \sim N\!\left(\begin{bmatrix} m \\ m \end{bmatrix},
\begin{bmatrix} s^2 + v & s^2 \\ s^2 & s^2 \end{bmatrix}\right)
$$

Because the level is part of the state, the filter integrates it out exactly.
Each player is shrunk toward the population mean by exactly the right amount
for how much data he has.

## 5. Kalman filter and likelihood

`models/kalman.py: kalman_filter, population_log_likelihood`

For each appearance:

$$
\begin{aligned}
\hat x_k^- &= \Phi_k \hat x_{k-1}, & P_k^- &= \Phi_k P_{k-1} \Phi_k^\top + Q_k \\
e_k &= \tilde y_k - H \hat x_k^-, & F_k &= H P_k^- H^\top + R_k \\
K_k &= P_k^- H^\top / F_k, & \hat x_k &= \hat x_k^- + K_k e_k \\
P_k &= (I - K_k H) P_k^- (I - K_k H)^\top + K_k R_k K_k^\top
\end{aligned}
$$

The covariance update is the Joseph form, which stays symmetric and positive
semi-definite in floating point.

By the prediction-error decomposition, each player's exact log-likelihood is

$$
\ell_i = \sum_k \left[ -\tfrac12 \log(2\pi F_k) - \frac{e_k^2}{2 F_k} + \ell^{\text{res}}_k \right],
$$

and the population log-likelihood is $\ell = \sum_i \ell_i$. Players are
padded to a common length and run in parallel with `vmap`; padded steps carry
the state forward and add nothing.

## 6. Estimation

`models/estimation.py`

**Parameters.** The optimizer works on an unconstrained vector of length 12:

$$
\psi = (\log\kappa,\ \log\sigma,\ \log r,\ m,\ \log s,\ \log\tau,\ g_{\text{Apr}}, g_{\text{May}}, g_{\text{Jun}}, g_{\text{Aug}}, g_{\text{Sep}}, g_{\text{Oct}})
$$

**Objective.** The negative log-likelihood per pitch, $-\ell(\psi) / N$ with
$N = \sum n_k$. Dividing by $N$ makes the gradient tolerance ($10^{-8}$)
independent of how much data there is.

**Starting values.** Method of moments:

- $m_0$ is the pooled mean, and $r_0^2 = \sum S_k / \sum (n_k - 1)$.
- $s_0$ is the standard deviation of the player means.
- The excess variance of game means around player means is
  $e = \max\!\big(\text{mean}_k[(\bar y_k - \bar y_i)^2 - r_0^2/n_k],\ 0.01\, r_0^2\big)$.
- $\kappa_0 = \ln 2 / 60$ (a 60-day half-life).
- $e$ is split evenly: $\sigma_0 = \sqrt{e\, \kappa_0}$, so $v_0 = e/2$, and
  $\tau_0 = \sqrt{e/2}$.
- The seasonal knots start at 0.

**Unit-free coordinates.** L-BFGS runs on $u = (\psi - c) / w$ rather than on
$\psi$. Here $c$ is zero except $c_m = \bar y$ (the pooled mean), and $w$ is one
except $w_m = w_{g} = \hat\sigma_{\text{pool}}$, where

$$
\hat\sigma_{\text{pool}}^2 = \frac{\sum_k S_k + \sum_k n_k (\bar y_k - \bar y)^2}{N}
$$

is the pooled per-pitch variance. If the data are re-expressed as $a + b\,y$,
the log-scale parameters shift by $\log b$ (their curvature is unchanged), the
mean maps to $a + b\,m$, and the knots scale by $b$. In $u$ the objective
changes only by the constant $\log b$. L-BFGS therefore takes the same path
whether the metric is in mph, rpm or degrees.

**Uncertainty.** The covariance of $\hat\psi$ is the inverse observed
information, $\hat\Sigma = [\nabla^2(-\ell)(\hat\psi)]^{-1}$, computed by JAX
automatic differentiation in $\psi$ coordinates. 95% Wald intervals
$\hat\psi_j \pm 1.96\, \hat\Sigma_{jj}^{1/2}$ are exponentiated for the
log-scale parameters.

**Derived quantities.** Both are linear in $\psi$ on the log scale, so the
delta method is exact there:

$$
\log t_{1/2} = \log \ln 2 - \psi_1, \qquad
\log \sqrt{v} = \psi_2 - \tfrac12(\log 2 + \psi_1)
$$

Each standard error is $\sqrt{a^\top \hat\Sigma a}$, with $a = (-1, 0, \dots)$ and
$a = (-\tfrac12, 1, 0, \dots)$ respectively.

**Interpreting the fit.** $t_{1/2} = \ln 2 / \kappa$ is the time for half of a
talent deviation to decay. $\sqrt v$ is how far talent typically sits from the
player's level. The share of talent variance that is drift is $v / (v + s^2)$.

## 7. Smoother

`models/kalman.py: rts_smoother`

The Rauch-Tung-Striebel pass runs backward from the last appearance and
conditions every state on all of the player's games:

$$
\begin{aligned}
G_k &= P_k \Phi_{k+1}^\top (P_{k+1}^-)^{-1} \\
\hat x_k^s &= \hat x_k + G_k (\hat x_{k+1}^s - \hat x_{k+1}^-) \\
P_k^s &= P_k + G_k (P_{k+1}^s - P_{k+1}^-) G_k^\top
\end{aligned}
$$

The filtered estimates are real-time: what was knowable after game $k$. The
smoothed estimates are hindsight. Alerts use only filtered estimates.

## 8. Alert score

`models/trajectory.py: gap_scores`

The gap between talent and level is $g_k = \theta(t_k) - \mu = a^\top x_k$ with
$a = (1, -1)^\top$.

**The gap is stationary.** From the prior,
$\operatorname{Var}(g) = (s^2 + v) - 2s^2 + s^2 = v$ and
$\operatorname{Cov}(g, \mu) = s^2 - s^2 = 0$. The gap obeys
$dg = -\kappa g\,dt + \sigma\,dW$, so it is $N(0, v)$ at every appearance,
independent of the level.

**The posterior variance is fixed.** In a linear-Gaussian model the posterior
covariance $P_k$ depends on the schedule and sample sizes but not on the
observed values. So $P_k^{g} = a^\top P_k a$ is a known number.

**Calibration.** By the law of total variance,

$$
v = \operatorname{Var}\!\big(E[g_k \mid y_{1:k}]\big) + E\big[\operatorname{Var}(g_k \mid y_{1:k})\big]
= \operatorname{Var}\!\big(E[g_k \mid y_{1:k}]\big) + P_k^{g}
$$

The filtered gap mean is Gaussian with mean 0, so

$$
z_k = \frac{E[g_k \mid y_{1:k}]}{\sqrt{v - P_k^{g}}} \sim N(0, 1)
$$

exactly, under the model, for every player and appearance. The denominator is
floored at $10^{-12}$. The tempting alternative, dividing by
$\sqrt{P_k^{g}}$, is not calibrated: early in a career $P_k^{g} \approx v$, the
filtered mean barely moves, and real departures look small. The $z_k$ are
correlated across a player's consecutive games, so they are calibrated one at
a time, not as independent tests.

**Direction.** Evaluation uses $\text{score} = -z$ for metrics where a drop is
worrying (velocity, spin rate, extension), $z$ for rises, and $\lvert z \rvert$
for either direction (arm angle). Higher is always more alarming.

## 9. Workload z-scores

`features/usage.py`

For a workload value $w_k$ (pitches that day, or $\log(1 + \min(\text{rest}, 30))$),
the pitcher's earlier games give a running mean and sample variance:

$$
\bar w_{<k} = \frac{1}{c}\sum_{j<k} w_j, \qquad
s^2_{<k} = \frac{\sum_{j<k} w_j^2 - c\, \bar w_{<k}^2}{c - 1}, \qquad c = k - 1
$$

$$
z^w_k = \frac{w_k - \bar w_{<k}}{s_{<k}}
$$

Only earlier games enter, so nothing leaks from the future. The score is
undefined when $c < 5$, or when $s^2_{<k} \le 10^{-6}$. That floor matters for
the running-sums formula: identical values leave a variance near $10^{-16}$
rather than exactly 0, which would otherwise produce z-scores in the millions.

## 10. Injury evaluation

`evaluation/injury_alerts.py`, `evaluation/metric_scan.py`

### Labels and phases

Each game is joined to the pitcher's next IL placement (Section 4 of the
injury study). With horizon $h = 30$ days:

- **Positives:** games with $0 < \text{days to IL} \le h$ before a placement in
  the injury group of interest.
  - The **final game** (detection) is the last appearance before the
    placement.
  - **Warning games** are the earlier ones.
- **Negatives (healthy):** games with no placement within $h$ days.

Games before placements in other groups, or in the other phase, are excluded
from that comparison.

### AUC

The AUC is the weighted Mann-Whitney statistic: the probability that a
positive outscores a negative, counting ties as one half.

$$
\text{AUC} = \frac{\sum_{i \in +} \sum_{j \in -} w_i w_j \left[\mathbb 1(s_i > s_j) + \tfrac12 \mathbb 1(s_i = s_j)\right]}{\left(\sum_{i \in +} w_i\right)\left(\sum_{j \in -} w_j\right)}
$$

It is computed in $O(n \log n)$ by ranking unique scores and accumulating
weights per rank.

### Pitcher-cluster bootstrap

One pitcher's games are correlated, so pitchers are resampled rather than
games. Each of $B = 500$ replicates draws counts
$(c_1, \dots, c_G) \sim \text{Multinomial}(G; 1/G, \dots, 1/G)$ over the $G$
pitchers and gives every game of pitcher $p$ weight $c_p$. Every AUC is then
recomputed with those weights.

- **Interval:** the 2.5th and 97.5th percentiles of the replicates.
- **p-value** for a null value $\theta_0$ (0.5 for an AUC, 0 for a difference),
  with one pseudo-draw on each side so it is never 0:

$$
p = \min\!\left(1,\ 2 \min\!\left(\frac{\#\{\theta^* \le \theta_0\} + 1}{B + 1},\ \frac{\#\{\theta^* \ge \theta_0\} + 1}{B + 1}\right)\right)
$$

### Placebo contrast

The test statistic is
$D = \text{AUC}_{\text{arm}} - \text{AUC}_{\text{lower body}}$. Both AUCs are
computed on the same replicate weights, so $D^*$ is a paired difference: shared
variation in the healthy negatives cancels.

### Holm adjustment

For a scan of $m$ tests with ordered p-values
$p_{(1)} \le \dots \le p_{(m)}$:

$$
\tilde p_{(i)} = \max_{j \le i} \min\!\big(1,\ (m - j + 1)\, p_{(j)}\big)
$$

This controls the family-wise error rate at the nominal level under any
dependence between the tests.

### Event study

Each placement contributes its pitcher's last 8 appearances within the 60 days
before it, numbered $-1$ (the final game) back to $-8$. Healthy games are those
with no placement in the following 60 days. For group $G$ and position $k$:

$$
\Delta_{G,k} = \bar s_{G,k} - \bar s_{\text{healthy}}, \qquad
\text{SE} = \sqrt{\text{SE}_{G,k}^2 + \text{SE}_{\text{healthy}}^2}
$$

The 95% band is $\Delta \pm 1.96\,\text{SE}$. Differences between two injury
groups use the same formula, since the groups come from different placements.

### Alert thresholds

On warning games before arm placements together with healthy games, an alert
at threshold $c$ fires when $\text{score} \ge c$:

- **Alert rate:** share of games flagged.
- **Precision:** share of flagged games that are positives.
- **Lift:** precision divided by the base rate of positives.
- **Recall:** share of placements with at least one flagged warning game.
- **Lead time:** for each caught placement, the days from its earliest alert to
  the placement; the table reports the median.

## 11. Workload model

`evaluation/usage_model.py`

On warning games before arm placements together with healthy games:

$$
\Pr(Y = 1) = \operatorname{logit}^{-1}\!\left(\beta_0 + \beta_1 z^{\text{pitches}} + \beta_2 z^{\text{rest}} + \beta_3\, \text{score}\right)
$$

**Fit.** Newton-Raphson (IRLS), stopping when every step is below $10^{-10}$
(at most 50 iterations):

$$
\beta \leftarrow \beta + (X^\top W X)^{-1} X^\top (y - p), \qquad W = \operatorname{diag}\!\big(p(1 - p)\big)
$$

**Cluster-robust covariance.** Clustered by pitcher, with the small-sample
factor $G / (G - 1)$:

$$
\hat V = \frac{G}{G - 1}\, B^{-1} \Big(\sum_{g} u_g u_g^\top\Big) B^{-1}, \qquad
B = X^\top W X, \qquad u_g = \sum_{i \in g} x_i (y_i - p_i)
$$

Odds ratios are $e^{\beta}$, per one standard deviation of each term, with
intervals $e^{\beta \pm 1.96\,\text{SE}}$.

**Cross-validation.** Pitchers are randomly assigned to 5 folds. Each fold's
linear predictor comes from a model fitted on the other four, and one AUC is
taken over all out-of-fold predictions. This is done for workload alone and
for workload plus the alert score.

## 12. Verification by simulation

`evaluation/simulate.py`, `tests/`

`simulate_population` draws players exactly from the generative model of
Sections 3 and 4, at the pitch-summary level. The tests check that:

- maximum likelihood recovers the true parameters within their intervals;
- fits on rescaled data map exactly onto each other (Section 6: κ unchanged,
  σ, r, s, τ and the knots scaled, the mean shifted and scaled, and the
  log-likelihood shifted by $-N \log b$);
- the alert score has mean 0 and standard deviation 1 on simulated data;
- a simulated velocity fade before arm placements is detected by the
  evaluation, and only for arm placements;
- the final game separates better than warning games when the fade
  accelerates.