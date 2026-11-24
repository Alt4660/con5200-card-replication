# AI Appendix

## Sections 1 and 2

Sections 1 and 2 were completed without AI.

Name: Leopold Diouf
Date: November 22nd 2026

Using AI in Sections 1 or 2 is an integrity violation. Do not edit the statement above; if it
is not true, talk to the instructor before you submit.

## How to file an entry

Sections 3 to 6 allow AI as a documented co-pilot under P.R.I.M.E. File one entry for each
piece of work you used it on, under the section it belongs to. Paste the full exchange: every
prompt as you sent it and every reply as it came back, including what you rejected. A summary
is not a transcript. If you cannot explain a line of what the model gave you, you cannot
submit it.

If you did not use AI in a section, write "No AI used in this section." under its heading.

Copy this block for each entry:

```
### Entry: TODO what it was for (for example, Exercise 1, not-yet-treated comparison)

- Model and version: TODO
- Date: TODO

**[Prep]** TODO: the role you gave the model.
**[Request]** TODO: what you asked for.
**[Iterate]** TODO: the constraints you added, and the version-2 prompt.
**[Mechanism Check]** TODO: what you asked it to explain, and whether the explanation held.
**[Evaluate]** TODO: how you checked the output against your own data or results.

**Full transcript**

TODO: paste the exchange here, unedited.

**What I changed, and why:** TODO, one line per change.
```

## Section 3: Robustness

### Exercise 1

Model : Claude Sonnet 5.5
Date: November 21st 2026
Entry 1

Section and exercise: Section 3 – Exercise 1: Inference that does not need a strong first stage

Outcome: Used Claude to develop an Anderson–Rubin confidence set and compare it with conventional and robust 2SLS confidence intervals. I adapted the code to use the existing model specification and data preparation in my notebook.

The P.R.I.M.E. prompt

[Prep]

I'm working on a robustness exercise for my assignment, which is based on David Card (1995) and his paper on the returns to education. I've already completed the main IV analysis in my notebook, and I now want to look at how reliable the IV estimate is when we take the possibility of weak instruments into account.

[Request]

I'm using educ as the endogenous variable, nearc4 as the instrument, and lwage as the dependent variable. I also want to keep exper and expersq as controls.

First, I'd like to estimate the usual 2SLS model and report the coefficient on education, its standard error and its 95% confidence interval. Could you calculate this using both conventional standard errors and heteroskedasticity-robust standard errors so I can compare the results?

[Iterate]

After that, I want to use the Anderson–Rubin method to build a confidence set for the education coefficient. This method should be useful because it doesn't rely as heavily on the instrument being strong. I'd like to test possible values of the coefficient between -0.10 and 0.60, using increments of 0.002. For each candidate value, adjust the dependent variable by subtracting that value multiplied by educ, then regress the adjusted outcome on nearc4 and the controls. Use HC1 robust standard errors and identify the candidate values for which the coefficient on nearc4 is not statistically significant at the 5% level.

Please organise the results into a DataFrame so I can compare the conventional 2SLS results, robust 2SLS results and Anderson–Rubin confidence set. Save the output as output/ex1_weak_iv_inference.csv.

[Mechanism Check]

I'd also like a brief explanation of why conventional IV confidence intervals can be misleading when an instrument is weak, and how the Anderson–Rubin method addresses this.

[Evaluate]

Finally, help me interpret whether the evidence for a positive return to education still holds when using the Anderson–Rubin approach, and what the results suggest about the reliability of the IV estimate.

Please write the Python code with comments explaining the main steps. Use the dataset and variables already defined in my notebook, and reuse the existing imports and setup wherever possible. Don't reload the data or start the analysis from scratch. I want the code to fit naturally into the notebook I've already been working on.

Verification log

1. The prompt, as sent

I'm working on a robustness exercise for my assignment, which is based on David Card (1995) and his paper on the returns to education. I've already completed the main IV analysis in my notebook, and I now want to look at how reliable the IV estimate is when we take the possibility of weak instruments into account.

I'm using educ as the endogenous variable, nearc4 as the instrument, and lwage as the dependent variable. I also want to keep exper and expersq as controls.

First, I'd like to estimate the usual 2SLS model and report the coefficient on education, its standard error and its 95% confidence interval. Could you calculate this using both conventional standard errors and heteroskedasticity-robust standard errors so I can compare the results?

After that, I want to use the Anderson–Rubin method to build a confidence set for the education coefficient. This method should be handy because it doesn't rely as heavily on the instrument being strong. I'd like to test possible values of the coefficient between -0.10 and 0.60, using increments of 0.002. For each candidate value adjust the dependent variable by subtracting that values multiplied by educ then regress the adjusted outcome on nearc4 and the controls. Use HC1 robust standard errors and identify the candidate values for which the coefficient on nearc4 is not statistically significant at the 5% level.

Please organise the results into a DataFrame so I can compare the conventional 2SLS results, robust 2SLS results and Anderson–Rubin confidence set. Save the output as output/ex1_weak_iv_inference.csv.

I'd also like a brief explanation of why conventional IV confidence intervals can be misleading when an instrument is weak, and how the Anderson–Rubin method addresses this.

Please write the Python code with comments explaining the main steps. Use the dataset and variables already defined in my notebook, and reuse the existing imports and setup wherever possible. Don't reload the data or start the analysis from scratch. I want the code to fit naturally into the notebook I've already been working on.

2. What it generated, in full

Claude explained that it had not been able to run the code on card.dta, but had tested it on simulated data with the same column names. It also pointed out that the main code used only exper and expersq, rather than the full controls in my headline specification.

Its generated code was:

# ---- Exercise 1: weak-instrument-robust inference (Anderson-Rubin) ----
# One extra import: 02_robustness doesn't import IV2SLS, but 01 does.
from linearmodels.iv import IV2SLS

# Variables for this exercise (df already exists in the notebook).
Y, D, Z = "lwage", "educ", "nearc4"

# Controls as specified: exper + expersq only.
# To match the Section 2 headline spec, use f"exper + expersq + {CONTROLS}" instead.
AR_CONTROLS = "exper + expersq"

# --- 1. 2SLS with conventional and heteroskedasticity-robust SEs ---
# educ is instrumented by nearc4, so the model is exactly identified.
iv_fml = f"{Y} ~ 1 + {AR_CONTROLS} + [{D} ~ {Z}]"
iv_conv = IV2SLS.from_formula(iv_fml, data=df).fit(cov_type="unadjusted")
iv_rob = IV2SLS.from_formula(iv_fml, data=df).fit(cov_type="robust")

def iv_row(label, res):
    """Coefficient, SE and 95% Wald CI on educ from a fitted 2SLS model."""
    lo, hi = res.conf_int(level=0.95).loc[D]
    return {"method": label, "coef": res.params[D], "se": res.std_errors[D],
            "ci_lower": lo, "ci_upper": hi, "ci_type": "Wald (normal)"}

# --- 2. Anderson-Rubin confidence set by grid inversion ---
# For each candidate beta0, test H0: beta = beta0. Under H0, (y - beta0*educ)
# is unrelated to the instrument, so nearc4 should have a zero coefficient
# when the adjusted outcome is regressed on nearc4 and the controls.
GRID = np.round(np.linspace(-0.10, 0.60, 351), 3)

work = df.copy()
pvals = []
for b0 in GRID:
    work["y_adj"] = work[Y] - b0 * work[D]
    fit = smf.ols(f"y_adj ~ {Z} + {AR_CONTROLS}",
                  data=work).fit(cov_type="HC1")
    pvals.append(fit.pvalues[Z])

ar = pd.DataFrame({"beta0": GRID, "p_value": pvals})
ar["in_set"] = ar["p_value"] >= 0.05

# The set can be disjoint or unbounded, so describe it as contiguous runs of grid points.
kept = ar.loc[ar["in_set"], "beta0"].to_numpy()
runs = np.split(kept, np.where(np.diff(kept) > 0.002 + 1e-9)[0] + 1) if len(kept) else []

ar_desc = " U ".join(f"[{r[0]:.3f}, {r[-1]:.3f}]" for r in runs) or "empty on grid"

# --- 3. Combine into one comparison table ---
ar_row = {"method": "Anderson-Rubin (HC1), grid", "coef": np.nan, "se": np.nan,
          "ci_lower": kept.min() if len(kept) else np.nan,
          "ci_upper": kept.max() if len(kept) else np.nan,
          "ci_type": "AR set: " + ar_desc}

ex1 = pd.DataFrame([iv_row("2SLS, conventional SE", iv_conv),
                    iv_row("2SLS, robust (HC) SE", iv_rob),
                    ar_row])
ex1["ci_width"] = ex1["ci_upper"] - ex1["ci_lower"]
ex1["n_obs"] = int(iv_conv.nobs)

ex1["grid_hits_edge"] = [False, False,
                         bool(len(kept) and (kept.min() == GRID[0] or
                                             kept.max() == GRID[-1]))]

ex1.to_csv(ROOT / "output" / "ex1_weak_iv_inference.csv", index=False)
ex1

Claude also explained that conventional IV confidence intervals can be misleading when the first stage is weak because the 2SLS estimator is a ratio of the reduced-form effect to the first-stage effect. If the first stage is close to zero, the ratio can have a skewed and heavy-tailed sampling distribution. This makes the usual normal approximation unreliable.

It explained that the Anderson–Rubin method instead tests candidate values by examining whether the adjusted outcome is related to the instrument. Since this test does not divide by the first-stage coefficient, it remains valid even when the instrument is weak. A potential downside is that the resulting confidence set can be wide, disconnected or unbounded.

Claude also noted that the first-stage F-statistic from my earlier analysis was 16.19, suggesting that the conventional and Anderson–Rubin intervals might be fairly similar.

3. What I changed, and why

a. Used the existing SIMPLE model specification. Claude's main version estimated the IV model using only exper and expersq as controls. I changed this to use the SIMPLE formula already defined in my notebook. This includes the additional controls for race, location, region and family background. I wanted the exercise to remain consistent with the specification I had already established rather than estimate a separate model with fewer controls.

b. Reused the existing dataset and setup. Claude's code assumed that df was already available and introduced its own model formula using the variable names Y, D and Z. I used the existing df, CONTROLS and SIMPLE objects from my setup instead. This meant I did not need to recreate the model specification or repeat the data preparation.

c. Used a separate function for the Anderson–Rubin test. Claude stored every p-value in a list and then created a separate DataFrame to identify the accepted values. I used an ar_pvalue() function to perform the regression for each candidate coefficient and return the p-value on nearc4. I then used a list comprehension to keep the values where the p-value was above 0.05. This made the calculation more direct and kept the code shorter.

d. Simplified the confidence-set calculation. Claude included extra code to identify separate runs of accepted values and check whether the set touched the edge of the grid. I instead used the minimum and maximum of the accepted values to report the confidence-set bounds. In my results all 119 accepted values were consecutive, so there was no need to report separate intervals.

e. Simplified the results table. Claude included additional columns for confidence-interval width, sample size and whether the confidence set reached the edge of the grid. I kept the table focused on the method, estimate, standard error and confidence-interval bounds. I also saved the CSV with index=False so that an additional DataFrame index would not be included in the output.

f. Used the grid already specified in my notebook. My code uses np.arange(-0.10, 0.60, 0.002), while Claude used a grid that included the final value of 0.60. This is a small implementation difference. The accepted values in my results were not close to the upper boundary, so excluding that final grid point did not affect the reported confidence set.

4. Verification and final results

I ran the final code using the Card dataset. The conventional and robust 2SLS estimates were both approximately 0.140. The conventional 95% confidence interval was approximately 0.036 to 0.244, while the robust interval was approximately 0.037 to 0.243.

The Anderson–Rubin confidence set ran from approximately 0.044 to 0.280. All 119 accepted grid points were consecutive, meaning the confidence set was not disconnected.

The results suggest that the estimated return to education remains positive even when using Anderson–Rubin inference. The Anderson–Rubin interval is slightly wider than the conventional and robust intervals, but it still excludes zero. The first-stage F-statistic of 16.19 also suggests that weak-instrument concerns do not substantially change the conclusion in this specification.

### Exercise 2

Model : Claude Sonnet 5.5
Date: November 21st 2026

Entry 2

Section and exercise: Section 3 – Exercise 2: Specification curve over the choices the paper leaves open

Outcome: Developed a specification curve using five parental-education groupings, two treatments of experience and two standard-error types. I adapted Claude's proposed groupings to the alternatives used in my notebook and produced the final plot and results table.

The P.R.I.M.E. prompt

[Prep]

I'm moving on to the second part of my assignment. For this exercise I want to create a specification curve to see how much the IV estimate of the return to education changes depending on the way I set up the model.

[Request]

I need to run 20 different IV specifications. These should come from combining five different ways of grouping parental education with two different ways of treating experience and two types of standard errors.

[Iterate]

For experience I want to compare treating exper and expersq as normal control variables with treating them as endogenous variables. In the second case they should be instrumented using age and agesq.

For the parental education part use the five different groupings required by the assignment. Use the variables already in my notebook for this and make sure each version is set up correctly.

For every specification I want to calculate the IV coefficient on educ along with its standard error and 95% confidence interval. Use both conventional and robust standard errors. Put everything into one DataFrame and make it clear which specification each result comes from. Save the results as output/ex2_specification_curve.csv.

[Mechanism Check]

I'd also like to understand what the specification curve is showing. Briefly explain why we're changing these parts of the model and what the difference is between treating experience as exogenous and instrumenting for it.

[Evaluate]

Once the results are calculated help me interpret how much the education coefficient changes across the 20 specifications. In particular I want to know whether the estimates stay positive and whether the results are particularly sensitive to the different modelling choices.

Please write the code with comments so I can follow what each part is doing. Use the dataset and setup that's already in my notebook. Don't reload the data or rewrite things that are already defined. I want the code to fit into what I've already done.

Verification log

1. The prompt, as sent

I'm moving on to the second part of my assignment. For this exercise I want to create a specification curve to see how much the IV estimate of the return to education changes depending on the way I set up the model.

I need to run 20 different IV specifications. These should come from combining five different ways of grouping parental education with two different ways of treating experience and two types of standard errors.

For experience I want to compare treating exper and expersq as normal control variables with treating them as endogenous variables. In the second case they should be instrumented using age and agesq.

For the parental education part use the five different groupings required by the assignment. Use the variables already in my notebook for this and make sure each version is set up correctly.

For every specification I want to calculate the IV coefficient on educ along with its standard error and 95% confidence interval. Use both conventional and robust standard errors. Put everything into one DataFrame and make it clear which specification each result comes from. Save the results as output/ex2_specification_curve.csv.

I'd also like to understand what the specification curve is showing. Briefly explain why we're changing these parts of the model and what the difference is between treating experience as exogenous and instrumenting for it. Once the results are calculated help me interpret how much the education coefficient changes across the 20 specifications. In particular I want to know whether the estimates stay positive and whether the results are particularly sensitive to the different modelling choices.

Please write the code with comments so I can follow what each part is doing. Use the dataset and setup that's already in my notebook. Don't reload the data or rewrite things that are already defined. I want the code to fit into what I've already done.

2. What it generated, in full

Claude explained that it could not access the repository data and therefore tested its code using simulated data with the same column names. It produced 20 rows in that test and said that one baseline specification matched earlier model objects to six decimal places.

Its proposed code was:

# ---- Exercise 2: specification curve (5 groupings x 2 experience x 2 SE = 20) ----
# Uses df, FAMILY_BASE, REGIONS, ROOT and IV2SLS from the notebook.

BASE_CONTROLS = f"black + south + smsa + smsa66 + {REGIONS}"

GROUPINGS = {
    "g1_none": dict(desc="no family controls at all", cuts=None,
                    labels=None, family=False),
    "g2_nointx": dict(desc="family vars, no interaction dummies", cuts=None,
                      labels=None, family=True),
    "g3_3cls": dict(desc="3 classes: <12 / 12 / >12 (baseline)",
                    cuts=[-1, 11, 12, 30],
                    labels=["lt12", "hs", "morehs"], family=True),
    "g4_3cls_b": dict(desc="3 classes: <12 / 12-15 / 16+",
                      cuts=[-1, 11, 15, 30],
                      labels=["lt12", "hs_some", "col"], family=True),
    "g5_4cls": dict(desc="4 classes: <12 / 12 / 13-15 / 16+",
                    cuts=[-1, 11, 12, 15, 30],
                    labels=["lt12", "hs", "some", "col"], family=True),
}

def build_controls(spec):
    """Return a copy of df with this grouping's interaction dummies, plus the control string."""
    d = df.drop(columns=[c for c in df.columns if c.startswith("pe_")]).copy()
    if not spec["family"]:
        return d, BASE_CONTROLS, 0
    if spec["cuts"] is None:
        return d, f"{BASE_CONTROLS} + {FAMILY_BASE}", 0

    mom = pd.cut(d["motheduc"], bins=spec["cuts"],
                 labels=spec["labels"]).astype(str)
    dad = pd.cut(d["fatheduc"], bins=spec["cuts"],
                 labels=spec["labels"]).astype(str)
    dums = pd.get_dummies(mom + "_" + dad, prefix="pe",
                          drop_first=True).astype(int)
    d = pd.concat([d, dums], axis=1)
    return d, f"{BASE_CONTROLS} + {FAMILY_BASE} + " + " + ".join(dums.columns), dums.shape[1]

EXPER = {
    "exogenous": lambda c: f"lwage ~ 1 + exper + expersq + {c} + [educ ~ nearc4]",
    "endogenous": lambda c: f"lwage ~ 1 + {c} + [educ + exper + expersq ~ nearc4 + age + agesq]",
}

SE_TYPES = {"conventional": "unadjusted", "robust": "robust"}

rows = []
for g_name, g in GROUPINGS.items():
    d, ctrl, k = build_controls(g)
    for e_name, fml in EXPER.items():
        model = IV2SLS.from_formula(fml(ctrl), data=d)
        for se_name, cov in SE_TYPES.items():
            res = model.fit(cov_type=cov)
            lo, hi = res.conf_int(level=0.95).loc["educ"]
            rows.append({"grouping": g_name, "grouping_desc": g["desc"],
                         "n_parent_dummies": k, "experience": e_name,
                         "se_type": se_name, "coef": res.params["educ"],
                         "se": res.std_errors["educ"], "ci_lower": lo,
                         "ci_upper": hi,
                         "first_stage_F": res.first_stage.diagnostics.loc["educ", "f.stat"],
                         "n_obs": int(res.nobs)})

spec = pd.DataFrame(rows)
spec.insert(0, "spec_id", np.arange(1, len(spec) + 1))
spec["spec_label"] = (spec["grouping"] + " | exper " + spec["experience"]
                      + " | " + spec["se_type"] + " SE")
assert len(spec) == 20

spec.to_csv(ROOT / "output" / "ex2_specification_curve.csv", index=False)
spec

Claude's explanation was that it used five groupings based on the variants it could identify from my earlier notes. It included a version with no family controls, a version with family variables but no interaction dummies, two three-class alternatives and one four-class alternative. It explicitly said that the four-class cutoffs were a guess because it could not see the assignment instructions.

It also explained the difference between treating experience as exogenous and endogenous. When experience is treated as exogenous, it is included as a normal control and nearc4 is the only excluded instrument. When experience is treated as endogenous, age and agesq are added as instruments for exper and expersq. Claude noted that potential experience is mechanically related to age and education, so treating it as exogenous may be questionable.

3. What I changed, and why

a. Changed the five parental-education groupings. Claude included a specification with no family controls at all. I did not use this version. Instead, my first specification left out the parental-education interaction dummies while retaining the underlying family controls. This was important because I wanted to isolate the effect of changing the way parental education was grouped, rather than removing family background controls altogether.

b. Used the five groupings implemented in my build() function. My notebook uses the following alternatives: no interaction dummies; three classes using the cutoffs below 12, exactly 12 and above 12; three classes using below 12, 12–15 and 16+; four classes using below 12, exactly 12, 13–15 and 16+; and four classes using below 10, 10–11, exactly 12 and above 12. Claude's final grouping used the same first three-class grouping but guessed the four-class alternatives differently. I replaced these with the cutoffs I had selected for my own analysis.

c. Reused the existing build() function. Claude wrote a separate build_controls() function that removed previously generated pe_ columns and then rebuilt the controls for each grouping. I used the build() function already defined in my notebook. This function takes the raw data and the selected cutoffs and labels, creates the parental-education interaction dummies and returns the corresponding control specification. Reusing it meant that the data preparation was handled consistently across the five alternatives.

d. Kept the same two treatments of experience. I retained Claude's basic comparison between treating exper and expersq as exogenous controls and treating them as endogenous variables instrumented by age and agesq. The difference between the two specifications is important because potential experience is related to age and education. Treating it as endogenous allows the model to account for this rather than assuming it is an ordinary exogenous control.

e. Kept both standard-error specifications. I retained the comparison between conventional and robust standard errors. These do not change the estimated IV coefficient itself, but they can change the estimated uncertainty around it. This allowed me to see whether the conclusions were sensitive to the standard-error correction as well as to the modelling choices.

f. Added the specification-curve plot. Claude's code created a table of the 20 estimates but did not include the final plot in its generated code. I added a plot that sorts the estimates from lowest to highest and displays their 95% confidence intervals. I also added reference lines for the paper's IV estimate of 0.132 and OLS estimate of 0.073. This makes it easier to compare the range of estimates visually and see whether the main result changes across the specifications.

g. Kept the output focused on the coefficient and standard error. My final DataFrame records the parental-education grouping, experience treatment, standard-error type, IV return and standard error. I then used the standard error to construct the plotted confidence intervals. Claude's version included additional columns such as the first-stage F-statistic, observation count and confidence-interval bounds. I kept the saved table more concise because the main purpose of the exercise was to compare the education coefficients across the 20 specifications.

4. Verification and final results

I ran the 20 specifications using the Card dataset. The estimated return to education ranged from 0.1281 to 0.1423. All 20 estimates were positive.

The paper's IV estimate of 0.132 falls within this range. The main difference came from how experience was treated. Treating experience as exogenous generally increased the education coefficient by around 0.01 compared with treating it as endogenous. The different parental-education groupings had a smaller effect, while changing from conventional to robust standard errors left the coefficient unchanged and only slightly affected the standard error.

Overall, the estimates stayed within a relatively narrow range. The modelling choices therefore did not substantially change the conclusion that education has a positive effect on wages. The results also show that the treatment of experience matters more than the parental-education grouping or standard-error choice in this exercise.


### Exercise 3

Model : Claude Sonnet 5.5
Date: November 21st 2026

Entry 3

Section and exercise: Section 3 – Exercise 3: Balance check on the instrument

Outcome: Used regressions to examine whether nearc4 is associated with predetermined characteristics. I replaced Claude's guessed list of variables with the seven characteristics used in my notebook and applied the existing data.

The P.R.I.M.E. prompt

[Prep]

For the third exercise I want to check whether my instrument nearc4 is balanced. Basically I want to see whether people who lived near a four-year college differ systematically from those who didn't in terms of characteristics that should have been determined before their education.

[Request]

I need to run separate regressions using nearc4 as the main explanatory variable and the seven predetermined characteristics listed in the assignment as the dependent variables.

For each regression include smsa66 and the regional dummy variables reg662 through reg669 as controls. Use HC1 robust standard errors.

[Iterate]

I'd like to put all the results into one DataFrame. Include the coefficient on nearc4 along with its standard error and p-value. Also include the 95% confidence interval and the number of observations for each regression. Save the results as output/ex3_instrument_balance.csv.

[Mechanism Check]

Once the regressions are done I'd like a short explanation of what this balance test is checking and why these characteristics matter when assessing whether nearc4 is a valid instrument.

[Evaluate]

Help me interpret the results too. I want to know which characteristics are significantly related to college proximity and what that might mean for the validity of the instrument.

Please write the code with comments so I can follow what is happening. Use the dataset and variables already defined in my notebook. Don't reload the data or repeat setup work that I've already done. I want this to fit into my existing code rather than look like a completely separate analysis.

Verification log

1. The prompt, as sent

For the third exercise I want to check whether my instrument nearc4 is balanced. Basically I want to see whether people who lived near a four-year college differ systematically from those who didn't in terms of characteristics that should have been determined before their education.

I need to run separate regressions using nearc4 as the main explanatory variable and the seven predetermined characteristics listed in the assignment as the dependent variables.

For each regression include smsa66 and the regional dummy variables reg662 through reg669 as controls. Use HC1 robust standard errors.

I'd like to put all the results into one DataFrame. Include the coefficient on nearc4 along with its standard error and p-value. Also include the 95% confidence interval and the number of observations for each regression. Save the results as output/ex3_instrument_balance.csv.

Once the regressions are done I'd like a short explanation of what this balance test is checking and why these characteristics matter when assessing whether nearc4 is a valid instrument. Help me interpret the results too. I want to know which characteristics are significantly related to college proximity and what that might mean for the validity of the instrument.

Please write the code with comments so I can follow what is happening. Use the dataset and variables already defined in my notebook. Don't reload the data or repeat setup work that I've already done. I want this to fit into my existing code rather than look like a completely separate analysis.

2. What it generated, in full

Claude explained that it had not seen the assignment's exact list of seven variables, so it made a best guess based on the Card dataset. It proposed the following list:

BALANCE_VARS = [
    "black", "fatheduc", "motheduc", "momdad14",
    "sinmom14", "step14", "libcrd14"
]

It then generated this code:

# ---- Exercise 3: is nearc4 balanced on predetermined characteristics? ----
# Uses df, REGIONS and ROOT from the notebook; no new imports needed.

BALANCE_VARS = [
    "black", "fatheduc", "motheduc", "momdad14",
    "sinmom14", "step14", "libcrd14"
]

missing_cols = [v for v in BALANCE_VARS if v not in df.columns]
assert not missing_cols, f"Not in df: {missing_cols}"

rows = []
for y in BALANCE_VARS:
    d = df.copy()
    # fatheduc/motheduc were mean-imputed in Section 2. Balance should be tested
    # on real values only, so drop the imputed rows for those two.
    if f"{y}_miss" in d.columns:
        d = d[d[f"{y}_miss"] == 0]

    fit = smf.ols(
        f"{y} ~ nearc4 + smsa66 + {REGIONS}", data=d
    ).fit(cov_type="HC1")

    lo, hi = fit.conf_int(alpha=0.05).loc["nearc4"]
    rows.append({
        "outcome": y,
        "coef_nearc4": fit.params["nearc4"],
        "se": fit.bse["nearc4"],
        "p_value": fit.pvalues["nearc4"],
        "ci_lower": lo,
        "ci_upper": hi,
        "mean_far": d.loc[d["nearc4"] == 0, y].mean(),
        "n_obs": int(fit.nobs)
    })

balance = pd.DataFrame(rows)
balance["sig_5pct"] = balance["p_value"] < 0.05
balance["sig_bonferroni"] = balance["p_value"] < 0.05 / len(balance)

balance.to_csv(ROOT / "output" / "ex3_instrument_balance.csv", index=False)
balance

Claude explained that the list of seven characteristics was only a guess because it could not see the assignment instructions. It also noted that the parent-education variables had been mean-imputed in my earlier data preparation, so it removed those imputed observations before running the balance regressions. It added the average for individuals not living near a college and a Bonferroni significance indicator to the output.

Its interpretation was that a balance test examines whether people living near a college differ in characteristics that were determined before their schooling decisions. If nearc4 is associated with parental education or family background, this could suggest that location is not random and that the instrument may be related to wages through channels other than education.

3. What I changed, and why

a. Replaced Claude's guessed list of characteristics. Claude selected black, fatheduc, motheduc, momdad14, sinmom14, step14 and libcrd14. I instead used the seven characteristics in my final notebook: KWW, IQ, fatheduc, motheduc, momdad14, sinmom14 and black. This was an important change because Claude explicitly said it had guessed the list. I used the characteristics I had selected for the assignment rather than relying on variables that were not part of my final exercise.

b. Used the raw data for the balance regressions. Claude used the prepared df and then removed observations with imputed parental education using the missing-value indicators. I used raw.dropna(subset=[var]) for each characteristic. This means each regression uses the original, non-missing observations for the characteristic being tested. It also avoids treating mean-imputed parental education values as if they were actual observations.

c. Kept the control variables consistent. I retained smsa66 and the regional dummy variables reg662 through reg669 in each regression. These controls account for differences in 1966 urban or rural location and region when testing whether college proximity predicts the predetermined characteristics.

d. Used a simpler results table. Claude included p-values, confidence intervals, a mean for individuals not living near a college and two significance indicators. My final table records the characteristic, number of observations, overall mean, coefficient on nearc4, robust standard error and t-statistic. I kept the output focused on the estimated differences and their statistical precision.

e. Used the actual sample sizes for each characteristic. The regressions use different numbers of observations because some characteristics have missing values. My code drops missing observations separately for each outcome rather than forcing all seven regressions to use the same sample. This preserves the available observations for each characteristic.

4. Verification and final results

I ran the seven regressions using HC1 robust standard errors. The results were:

| Characteristic | Observations | nearc4 coefficient | Robust SE | t-statistic |
| :--- | :---: | :---: | :---: | :---: |
| KWW | 2,963 | 0.676 | 0.374 | 1.809 |
| IQ | 2,061 | 0.348 | 0.808 | 0.431 |
| Father's education | 2,320 | 0.259 | 0.184 | 1.405 |
| Mother's education | 2,657 | -0.102 | 0.147 | -0.695 |
| Lived with both parents at 14 | 3,010 | -0.034 | 0.018 | -1.839 |
| Lived with a single mother at 14 | 3,010 | 0.021 | 0.013 | 1.574 |
| Black | 3,010 | 0.048 | 0.017 | 2.766 |

The clearest difference was race. Individuals living near a four-year college were approximately 4.8 percentage points more likely to be Black, and this relationship was statistically significant. The KWW score and the likelihood of having lived with both parents at age 14 were closer to conventional significance levels, while the other characteristics did not show statistically significant differences.

The race result also passed the Bonferroni adjustment for seven tests. This suggests that college proximity is not completely random, even after controlling for region and urban or rural location. However, race is included as a control in my main IV specification. The balance test therefore highlights the importance of controlling for predetermined differences, but it does not by itself prove that the instrument is invalid.

### Exercise 4

Model : Claude Sonnet 5.5
Date: November 21st 2026

Entry 4


Section and exercise: Section 3 – Exercise 4: How big a direct effect would undo the IV result

Outcome: Examined how sensitive the IV estimate is to a possible direct effect of college proximity on wages. I adapted Claude's sensitivity calculation to use my existing SIMPLE specification and added a direct re-estimation check.

The P.R.I.M.E. prompt

[Prep]

For the final exercise I want to look at how sensitive my IV results are to the exclusion restriction assumption.

[Request]

The concern is that living near a four-year college (nearc4) might have some direct effect on wages rather than affecting wages only through education. I want to see how much this would change my original IV estimate.

Use the main IV model I've already been working with in my notebook. The endogenous variable is educ and the instrument is nearc4. Keep the same controls and model specification as in my existing analysis.

[Iterate]

The exercise uses the following adjustment:

[
b(\delta) = b - \frac{\delta}{\pi}
]

Here b is the original IV estimate and pi is the first-stage coefficient on nearc4. Delta represents the possible direct effect of college proximity on wages.

I want to calculate the adjusted IV estimate over a range of possible delta values, including zero so I can compare everything with the original estimate. For each value calculate the adjusted estimate and its 95% confidence interval. Put the results into a DataFrame and save them as output/ex4_exclusion_sensitivity.csv.

I also need to calculate the value of delta at which the lower bound of the 95% confidence interval reaches zero. This should give an indication of how large a direct effect would need to be before the evidence for a positive return to education disappears.

[Mechanism Check]

Please explain how this adjustment works and what the delta threshold tells us about the exclusion restriction.

[Evaluate]

Help me interpret whether the original positive IV result is reasonably robust or whether even a relatively small direct effect could change the conclusion.

Please write the code with comments so I can understand each step. Use the dataset and model setup already in my notebook. Don't reload the data or repeat code that I've already written. I want this to fit naturally into my existing analysis.

Verification log

1. The prompt, as sent

For the final exercise I want to look at how sensitive my IV results are to the exclusion restriction assumption.

The concern is that living near a four-year college (nearc4) might have some direct effect on wages rather than affecting wages only through education. I want to see how much this would change my original IV estimate.

Use the main IV model I've already been working with in my notebook. The endogenous variable is educ and the instrument is nearc4. Keep the same controls and model specification as in my existing analysis.

The exercise uses the following adjustment:

[
b(\delta) = b - \frac{\delta}{\pi}
]

Here b is the original IV estimate and pi is the first-stage coefficient on nearc4. Delta represents the possible direct effect of college proximity on wages.

I want to calculate the adjusted IV estimate over a range of possible delta values, including zero so I can compare everything with the original estimate. For each value calculate the adjusted estimate and its 95% confidence interval. Put the results into a DataFrame and save them as output/ex4_exclusion_sensitivity.csv.

I also need to calculate the value of delta at which the lower bound of the 95% confidence interval reaches zero. This should give an indication of how large a direct effect would need to be before the evidence for a positive return to education disappears.

2. What it generated, in full

Claude explained that it interpreted the requested adjustment as a sensitivity analysis in which a possible direct effect of nearc4 is subtracted from the original IV estimate after scaling it by the first-stage coefficient. It used the existing iv, fs and ROOT objects from the earlier analysis.

Its generated code was:

# ---- Exercise 4: sensitivity to a direct effect of nearc4 on wages ----
# Uses iv (main IV model), fs (first stage) and ROOT from Section 2. No new imports.
# iv: educ, exper, expersq instrumented by nearc4, age, agesq, with CONTROLS.

b = iv.params["educ"]
se = iv.std_errors["educ"]
pi = fs.params["nearc4"]
ci_lo, ci_hi = iv.conf_int(level=0.95).loc["educ"]

# If nearc4 has a direct wage effect delta, the IV estimate converges to beta + delta/pi,
# so the adjusted estimate is b(delta) = b - delta/pi.
# Treating delta as known and pi as fixed, the whole CI shifts by the same amount.

# Break-even values, in closed form (lower bound = ci_lo - delta/pi = 0):
delta_star = pi * ci_lo
delta_zero = pi * b

# Grid of delta from -0.05 to +0.06 in steps of 0.005 (includes 0), plus the
# two break-evens.
grid = set(np.arange(-50, 61, 5) / 1000) | {round(delta_star, 6),
                                             round(delta_zero, 6)}
sens = pd.DataFrame({"delta": sorted(grid)})

sens["shift"] = -sens["delta"] / pi
sens["adj_coef"] = b + sens["shift"]
sens["ci_lower"] = ci_lo + sens["shift"]
sens["ci_upper"] = ci_hi + sens["shift"]
sens["ci_excludes_zero"] = sens["ci_lower"] > 0
sens["b_original"], sens["pi"], sens["se"] = b, pi, se

sens["note"] = ""
sens.loc[np.isclose(sens["delta"], 0), "note"] = "delta = 0 (original IV)"
sens.loc[np.isclose(sens["delta"], delta_star), "note"] = "CI lower bound = 0"
sens.loc[np.isclose(sens["delta"], delta_zero), "note"] = "point estimate = 0"

sens.to_csv(ROOT / "output" / "ex4_exclusion_sensitivity.csv", index=False)

print(f"b = {b:.4f}, pi = {pi:.4f}")
print(f"delta* (CI lower bound = 0): {delta_star:.4f}")
print(f"delta at which the point estimate = 0: {delta_zero:.4f}")
sens

Claude also explained that a positive delta represents a direct positive effect of college proximity on wages. This would mean that part of the wage difference attributed to education is actually due to college proximity itself, reducing the adjusted IV estimate.

It noted that the adjustment treats delta and the first-stage coefficient as fixed and shifts the original confidence interval by the same amount. It also pointed out that a fuller sensitivity analysis could account for uncertainty in the first stage, and that the result depends on which instruments are included in the first stage.

3. What I changed, and why

a. Used my existing SIMPLE specification. Claude's code used an existing iv object from the earlier analysis, where experience was also treated as endogenous and instrumented using age and agesq. I instead estimated the model using SIMPLE, which treats exper and expersq as exogenous controls and uses nearc4 as the instrument for educ. This kept the sensitivity exercise consistent with the simpler IV specification I had used in Exercise 1.

b. Re-estimated the model with robust standard errors. I fitted the IV model using cov_type="robust" and extracted the coefficient and standard error for education from that result. This meant the confidence interval used in the sensitivity analysis was based on the robust standard error from my chosen specification.

c. Calculated the first stage directly. Claude used the fs object from the earlier analysis to obtain the first-stage coefficient on nearc4. I instead ran the first-stage regression directly using educ as the dependent variable and nearc4, experience and the existing controls as explanatory variables. This ensured that the first-stage coefficient matched the model used for the sensitivity calculation.

d. Added a direct re-estimation check. One change I made was to check the sensitivity formula by subtracting a direct effect of 0.02 from lwage and estimating the IV model again. I then compared this re-estimated coefficient with the value obtained from the formula (b - \delta/\pi). Both approaches produced the same result, which gave me a way to verify that the adjustment had been implemented correctly.

e. Used a different range of delta values. Claude proposed a range from -0.05 to 0.06 in increments of 0.005, including the calculated break-even values. I used a range from -0.02 to 0.05 in increments of 0.005. This allowed me to focus on the range of direct effects I wanted to examine while including zero as the original IV estimate.

f. Calculated the threshold for reaching the OLS estimate as well. Claude calculated the value of delta at which the lower confidence-interval bound reaches zero and the value at which the point estimate itself reaches zero. I instead calculated the value at which the IV estimate falls to the OLS estimate, as well as the value at which the lower confidence-interval bound reaches zero. This allowed me to compare the adjusted IV estimate with the OLS result and assess when the positive IV estimate would no longer be statistically distinguishable from zero.

g. Added a sensitivity plot. I added a plot showing the adjusted IV estimate across the different delta values, with its 95% confidence interval shaded around it. I also included a horizontal reference line for the OLS estimate and another at zero. This made it easier to see how quickly the IV estimate and its statistical significance change as the assumed direct effect increases.

4. Verification and final results

I ran the final code using the Card dataset. The original IV estimate was approximately 0.140, with a first-stage coefficient of approximately 0.3336. The OLS estimate was approximately 0.0721.

The direct-effect adjustment was also checked by re-estimating the model after subtracting a direct effect of 0.02 from log wages. The re-estimated coefficient was 0.0801, which matched the result from the sensitivity formula.

The results showed that a direct effect of approximately 0.0124 would be enough to make the lower bound of the 95% confidence interval reach zero. A larger direct effect of approximately 0.0227 would bring the IV point estimate down to the OLS estimate.

This suggests that the positive IV estimate is somewhat sensitive to a direct effect of college proximity on wages. The point estimate remains positive for a range of direct effects, but a direct effect of only around 1.2% is enough to remove statistical significance at the 5% level. The exclusion restriction is therefore an important assumption behind the original IV result, and the analysis shows that it cannot be dismissed as a minor concern.

## Section 4: Does the finding hold?

No AI was used in this section

## Section 5: Peer audit

Model: CLaude Sonnet 5.5

Date: November 24th


## P.R.I.M.E. prompts and log of verification

\*The P.R.I.M.E. prompts below are structured into the five parts of the framework based on the instructions I gave Claude. They don’t exactly replicate the original messages. *The full conversation with Claude is provided separately in the appendix.*

##### Entry 1 – Reading the assignment directions

**[Prep]**

Can you help me with Sections 5 and 6 of my ECON 5200 midterm replication project? Sections 1 and 2 are to be done without AI.

**[Request]**

Read the assignment instructions and explain:

1. What Sections 5 and 6 are and how they are graded.
2. What deadlines are relevant and already past.
3. AI usage rules, including P.R.I.M.E. requirements, and the requirement to include the full transcript in `ai-appendix.md`.
4. Any instructions that are not clear to me and that I should clarify before I begin.

**[Iterate]**

Don’t start the real work yet. Wait for my confirmation that you understand the instructions. Then I’ll share my classmate’s repository and we’ll go through the audit step by step. Show me the commands you execute and the output.

**[Mechanism Check]**

Describe the exact requirements and grading criteria and point out any confusing parts of the instructions.

**[Evaluate]**

Tell me if you have enough details to start writing Section 5 and 6 or if you need to clarify something.

**What I changed and why**

I haven't changed my original request much. I just put it in the P.R.I.M.E. sections so you could see how I approached the assignment before I did the audit.

---

### Claude's response

I read all 9 pages. pdfinfo shows 9 pages and pdftotext returned 368 lines. A few filenames in code font (ai-appendix.md, requirements.txt) dropped out of the text layer, but the context makes them clear. I haven’t opened any linked resources yet, such as the sample README, the sample ai-appendix.md or the template repo.

**1. What Sections 5 and 6 require**

Section 5, Peer Audit (12 pts).

- You clone and run one classmate’s repo. It is assigned, and it is never a paper you replicated.
- You file a GitHub issue on that repo of about 300 words. It must say:
  - whether the repo runs from a clean checkout, and what broke if it didn’t;
  - one substantive methodological question;
  - one thing they did that you will adopt.
- You paste the issue URL in your Canvas submission comment.
- It is graded on the audit you file. The criterion is “Issue filed; reproducibility actually tested from a clean checkout; one substantive methodological question raised.” The brief also says a courteous audit that finds nothing scores below a rigorous one that finds a real problem.

Section 6, Your Own Repository (12 pts).

The grader runs your one command from a clean checkout, on a machine with none of your files.

You ship:

- `requirements.txt` with pinned versions;
- the data, or a `download_data.py`, and the README says which you chose and why, given the licence;
- a `README.md` naming the one command that reproduces the headline table, plus the expected runtime;
- `ai-appendix.md`, with P.R.I.M.E. entries and full transcripts for Sections 3 to 6, plus an explicit statement that Sections 1 and 2 were unaided.

The README must also say which sections used AI.

A repo whose clean checkout reproduces a different number from the report scores zero here.

The checkpoints must show in the commit history.

A single-commit history, or code that disagrees with the report, can trigger a 10-minute follow-up conversation.

**Grading scale.** Every rubric row is full marks, 60% (7.2 of 12), or zero. There is nothing finer-grained.

**2. Deadlines**

Today is Mon 23 Nov, 22:10 ET. The PDF gives no timezone, so I’m assuming Eastern.

| Deadline            | Item                                           | Status                    |
| ------------------- | ---------------------------------------------- | ------------------------- |
| Mon 26 Oct          | Paper choice posted                            | Past (28 days ago)        |
| Mon 9 Nov           | Sections 1–2 committed, with replication table | Past                      |
| Tue 10 Nov, 11:59pm | Repo public                                    | Past                      |
| Sat 14 Nov          | Audit pairs assigned                           | Past                      |
| Thu 19 Nov, 11:59pm | Project due (repo and report)                  | Past by about 3 days 22 h |
| Sun 22 Nov, 11:59pm | Audit issue due                                | Past by about 22 h        |

Everything is past due. Two points affect how much that matters:

- The Canvas header says “Due Sunday by 23:59”, which conflicts with the Thu 19 Nov project date in the body.
- The document says nothing about late penalties.

The brief says a missed checkpoint costs only part of the Your own repository row, and never the project.

**3. AI-use rules**

Your reading matches the document:

- Sections 1 and 2: no AI, for code or prose. Using it is an integrity violation. I won’t draft or edit anything for them.
- Sections 3 to 6: AI is allowed as a documented co-pilot under P.R.I.M.E., “on the same terms as every problem set”.
  - You file the full transcript in `ai-appendix.md`.
  - If you can’t explain a line, you can’t submit it.
  - The README states which sections used AI.
  - The course rule is “manual command first, expansion second”.

**4. Ambiguities and what I need from you**

1. Late submission. Which date governs, 19 Nov or Canvas’s Sunday? Does Canvas still accept the submission and the audit issue, and what is the late policy? That policy isn’t in this PDF. The “New Attempt” button suggests you may have already submitted something. Please check, and consider emailing the instructor.
2. Which commit to audit. The classmate’s repo may have changed after the 19 Nov due date. I suggest auditing the last commit on or before then and recording the hash in the issue. Is that what you want?
3. Does the auditor get the report PDF? The report goes to Canvas, not necessarily to the repo. Without it I can only check the repo’s own claimed number, not report against repo.
4. Where “no AI on Sections 1–2” ends. Running your Sections 1–2 code in a clean checkout for Section 6 puts me near code I must not touch. My proposal is that I run it and report failures, and you make any fix in Sections 1–2 files by hand. Do you agree? If you’re unsure, ask the instructor.
5. P.R.I.M.E. format and transcript. The entry format is in the sample ai-appendix.md link, which I haven’t read. Should I fetch it? Does this preliminary reading step count as part of the transcript? I can’t write to your repo, so you’d need to export the transcript yourself.
6. My sandbox limits. Its network access is allowlisted. If the classmate’s `download_data.py` needs a blocked host (for example `users.nber.org`), a failure here would be my environment, not their repo. I’ll flag it if that happens, and you can re-run on your own machine. I’ll run their code in an isolated directory and virtualenv, since it’s untrusted.
7. “Manual command first.” Do you want to run their README’s command yourself first, with me expanding from there? Or should I run it and you verify?
8. Section 6 inputs. I’d need your repo link and the data licence status. I’d also need to know which checkpoint commits exist, because I won’t backdate or fabricate history.

I’ll also only draft the audit issue, never post it. It goes out under your name, so you file it.

When you’ve confirmed or corrected the above, send the classmate’s repo link and the commit to audit, and we’ll start.

---

### Entry 2 – Doing the peer audit

**[Prep]**

I need help with the ECON 5200 project peer audit Section 5. The audit should be done on some specific commit of my classmate's repo.

**[Request]**

Audit this repo:

[https://github.com/atanyeri/Midterm_ReplicationStudy](https://github.com/atanyeri/Midterm_ReplicationStudy)

Use commit 4083235 (19 November, "Update README.md")

Check it runs from a clean checkout, find the problems and prepare a GitHub issue of 300 words including:

1. Whether the repository runs and what breaks
2. An important methodological question.
3. Something I could use on my own project.

**[Iterate]**

- Run the code in a separate folder and virtualenv.
- Don't change the repo, don't submit the issue. Just make a draft.
- If no report PDF is available, state that and check only the repository's own claims.
- Refer to the example `ai-appendix.md` for the format to use.
- Highlight issues that might be caused by network restrictions, not the repository.
- I will run the README instructions myself first so run them independently as well and let me compare the results.
- Show the commands and outputs as we go.

**[Mechanism Check]**

Look at how the README, Python scripts, requirements, data and notebook work together. Check that the project can be run from a clean checkout and that missing files or data can be provided to reproduce results.

See also the placebo analysis and standard errors, particularly the comparison of the real cutoff to the placebo cutoffs.

**[Evaluate]**

Evaluate the reproducibility of the repository and the convincingness of the results and methodology. Think about whether differences in standard errors influence the conclusions. Name one practice I could take on in my own project.

**What I changed and why**

I ran the repo myself, and made a few changes to Claude's original GitHub issue draft.

**1. Updated the results.**

I replaced the execution details of the execution of Claude with my own results from Google Colab. I used Python 3.13.18. It took about 12 minutes to install the requirements, the first run failed after about 5 seconds and the notebook was completed after 298 seconds when I manually provided the data set. I added the headline sample size of 93,899 observations as well.

**2. Specified Python version.**

I added that README says Python 3.12 but does not enforce it. Just to mention that I ran the code using Python 3.13.

**3. The question of methodology was improved.**

I made the question more specific by examining the variation in standard errors across placebo cutoffs. The ratios vary from 0.73 to about 3. This brings some doubt on the reliability of the ranking of the real cutoff. I also left the suggestions about comparing placebos with similar standard errors and about using honest intervals for a discrete running variable (Koles'ar and Rothe, 2018).

**4. Maintained suggested practice.**

I took Claude's suggestion to have a `REPORT_NUMBERS` dictionary in the notebook, so that the numbers in the final report are directly tied to the code. I decided to use this in my own repo too.

---

### Claude's response

The data downloaded fine (200 OK, 9.45 MB, a valid Stata release-118 file). The network wasn’t a barrier here, so I can run the pipeline in the workaround copy.

I audited commit 4083235 and drafted the issue below. I haven’t posted anything. It does not run from a clean checkout, because `download_data.py` was never committed.

**What I ran**

The environment was Python 3.12.3 on Linux, in a fresh venv outside the clone. Their pin is 3.12.9, so the patch release differs.

```
git clone https://github.com/atanyeri/Midterm_ReplicationStudy . && git checkout 4083235
python3.12 -m venv ../venv
../venv/bin/python -m pip install -r requirements.txt     # README step 1
../venv/bin/python reproduce.py                           # README step 2
```

| Step                                     | Result                                                                                                                                  |
| ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `pip install -r requirements.txt`        | Succeeded in 15 s. All nine pins installed at the exact versions.                                                                       |
| `python reproduce.py` in the clean clone | Failed after 1.5 s with `can't open file '.../download_data.py': [Errno 2] No such file or directory`, then `Could not fetch the data`. |
| History check                            | `git log --all -- download_data.py ReplicationMidtermReport.pdf data/` returned nothing, so neither file was ever committed.            |

The clean-checkout failure is not caused by my network allowlist, because the file doesn’t exist in the repo. You should get the same failure.

To test the rest, I copied the clone to a scratch directory and downloaded `hansen_dwi.dta` from the README’s URL. That fetch worked (200 OK, 9.45 MB, valid Stata file, sha256 `08158875…ecd1`).

- `reproduce.py` then finished in 115 s. The README says about 1 min 20 s, which is a minor gap, and my machine may just be slower.
- It printed the headline row −0.0221, matching the README.
- A script of mine checked 24 claimed numbers against the code-cell outputs, and all 24 matched. My checker missed the 0.240 placebo only because of a formatting slip in my regex, and the notebook prints it correctly.
- I found no randomness in the code.
- I couldn’t check the report, since the PDF isn’t in the repo.

**Draft issue (about 300 words)**

Audit of commit 4083235 (19 Nov, “Update README.md”)

1. Clean checkout: does not run as committed. Fresh clone, new Python 3.12 virtualenv; pip install -r requirements.txt succeeded (all nine pins resolved). python reproduce.py then stops after about 1.5 s: it calls download_data.py, which the README and reproduce.py both rely on but which is in no commit (git log --all -- download_data.py is empty). ReplicationMidtermReport.pdf, listed in the README layout, is also missing. The report was not available to me, so I checked only the repo’s own claims. Likely fix: git add download_data.py and the PDF.

To test the rest, I saved hansen_dwi.dta from the README’s URL into a scratch copy. The run then finished in 115 s (README: about 1 min 20 s) and matched the README: −0.0221 (SE 0.0041); 192 specifications from −0.0241 to −0.0072, median −0.0165. I found no randomness in the code, so numbers should not vary by machine.

2. Methodological question. In 3.1, 3 of 17 placebo cutoffs are significant at 5% (0.185, 0.205, 0.240; about 0.85 expected), and the 0.205 placebo (−0.0219) exceeds the real estimate (−0.0193). You conclude the clustered SEs understate uncertainty, yet also that the jump is “specific to the legal threshold” because the real |t| = 4.01 ranks first (p = 1/18). If the SEs are too small, the real cutoff’s t is inflated too. How does the conclusion hold after calibrating, for instance using only placebos with SEs close to the real one (the ratio reaches \~3 in the thin right tail), or honest intervals for a discrete running variable (Kolesár and Rothe 2018)?
3. What I will adopt. REPORT_NUMBERS: one dictionary filled as the notebook runs and printed in one cell, so every number in the report traces to code.

Minor: only direct dependencies are pinned, and the data URL tracks master; a checksum would catch drift.

**Before you file it**

1. Run it yourself first. Section 1 of the issue claims you ran it. If your clean-clone run differs from mine (a different error, or a different headline number once the data is in), replace that section with your results.
2. Pick the “adopt” item honestly. REPORT_NUMBERS is my suggestion. Another candidate is `reproduce.py`’s `ThisPythonKernel`, which runs the notebook on the interpreter that installed the requirements. Use whichever you would really adopt, and be ready to explain the question in section 2.
3. Not for the issue: all six commits are dated 19 Nov, so no earlier checkpoints show in their history. That concerns their Section 6 grade, not the audit. I left it out of the issue, and you can decide whether it belongs there.
4. Your `ai-appendix.md`: this conversation is the Section 5 transcript. The verification log needs the output you compare against mine, and at least one change you make to my draft.

Once you’ve run it, tell me whether your output matches mine and which “adopt” item you chose. I’ll adjust the issue accordingly.


## Section 6: Your own repository

Model: Claude sonnet 5.5

Date: 24th November

## Section 6 – P.R.I.M.E. prompt and log of verification

*The P.R.I.M.E. prompt below is structured into the five parts of the framework based on the instructions I gave Claude. It does not exactly replicate the original message. The full conversation with Claude is provided separately in the appendix.*

### Entry 1 – Checking my own repository

**[Prep]**

Hi, I'm working on Section 6 of my midterm replication project. I've attached the assignment instructions, so please read them in full first.

My repo is https://github.com/Alt4660/con5200-card-replication. it works.

**[Request]**

Can you:

1. Clone it into an empty folder, install from requirements.txt, and run it so i can compare against my run?
2. Check that the numbers it prints match my README and my report PDF?
3. Tell me if you find anything missing or wrong in the README

**[Iterate]**

Two rules: DO not suggets any edits to any notbooks and please don't change my repo, just show me what to change and I'll do it myself.

**[Mechanism Check]**

Please show me the commands you run and what they print, so I can check it.

**[Evaluate]**

Check whether the clean checkout runs, whether the results match the README and report PDF, and whether anything is missing or incorrect in the README.

**What I changed and why**

I didn't change much sicne the idea was just to have a comparison for when I ran my repo form a clean check out. I asked it to do the same and I compared the results against so that I could find any missing or incorrect README information. Claude found the same one as me being that the runtime was longer than originally stated when ran from a fresh cleanout. However I got 45 seconds runtime and it got approximately 30 seconds. I also made clear that I did not want any changes made to my repo or notebooks.

---

### Claude's response, in full

The clean checkout runs, and every number the README and your report PDF claim matches what it prints. The README has some gaps, listed in section 3. I changed nothing in your repo. I only cloned it into a scratch folder of mine.

**1. Clean clone, install, run**

I used an empty folder (`ls -A | wc -l` printed `0`) and your README’s own commands.

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Alt4660/con5200-card-replication repo
cd repo && git log --oneline | wc -l
python -m venv .venv && source .venv/bin/activate
python --version
pip install -r requirements.txt
python reproduce.py
```

The outputs were:

```text
git log --oneline | wc -l        → 31 commits
python --version                 → Python 3.13.18
pip install -r requirements.txt  → Successfully installed … (12 s, no errors)
python reproduce.py
running notebooks/01_positioning_and_replication.ipynb ...
  done in 8 s
running notebooks/02_robustness.ipynb ...
  done in 19 s

All 2 notebooks ran in 27 s.
```

**Headline table (`output/headline_table.csv`)**

| | paper_estimate | my_estimate | gap_pct |
|---|---:|---:|---:|
| Table 4 row 1: OLS return to education | 0.073 | 0.072086 | -1.252452 |
| Table 3 Panel B col 6: IV return to education | 0.132 | 0.131399 | -0.455192 |
| Table 3 Panel B col 2: first stage, nearc4 on education | 0.365 | 0.380286 | 4.188040 |

- **Runtime:** The whole run took 29 s including the data download, so your README’s “about 30 seconds” is accurate.
- **Warning:** A harmless nbformat `MissingIDFieldWarning` printed. It comes from the notebooks, and I’m not suggesting any notebook changes.
- **Repo state:** `git status` shows only untracked files generated by the run (`output/*.csv`, `output/*.png`). `card.dta` and `output/executed/` are gitignored as intended.

**2. Do the numbers match?**

I compared the README, the report PDF, and the run. Everything matches.

| Number | README | Report PDF | Clean run |
|---|---|---|---|
| OLS | 0.0721 (−1.25%) | 0.0721, diff −0.0009 | 0.072086 |
| IV | 0.1314 (−0.46%) | 0.1314, diff −0.0006 | 0.131399 |
| First stage (nearc4) | 0.3803 (+4.19%) | 0.3803, diff +0.0153 | 0.380286 |

The report’s other figures also reproduce exactly from the executed notebooks and CSVs:

- **Sample and standard errors:** N = 3,010 with 37 variables; OLS s.e. 0.0037; IV s.e. 0.0469.
- **F-statistics and simple IV:** first stage 0.3336, F = 16.19 (simple) and 8.41 (headline); simple IV 0.14.
- **Section 3.1 intervals:** conventional 0.036–0.244, robust 0.037–0.243, Anderson-Rubin 0.044–0.280.
- **Specification curve:** range 0.1281–0.1423 over 20 specs. The five endogenous-experience versions are 0.1281, 0.1314, 0.1306, 0.1325 and 0.1335.
- **Balance check:** race coefficient 0.048 with t = 2.766, and all others |t| < 1.9 (the largest is 1.839).
- **Sensitivity:** delta 0.0124 and 0.0227.

The data licence claim also holds. I fetched the `scunning1975/mixtape` LICENSE and it is an MIT licence.

**3. What’s missing or wrong in the README**

**a) Leftover template text in the Data section.** From “The template downloads by default…” down to “…and say so above.” is the instructor’s template wording, not your decision. It talks about “five files”, Dehejia-Wahba, and “if you decide…”. Delete that whole block. Then make the “Included or downloaded” bullet carry your reasoning:

```markdown
- **Included or downloaded:** Downloaded. The `scunning1975/mixtape` repository is under an MIT licence (copyright A. S. Cunningham), which would let me keep a copy if I kept the notice. I chose to download it so the repo stays small and the data stay at their source. `python reproduce.py` fetches it on the first run via `download_data.py`, and `.gitignore` excludes `data/*.dta`.
```

**b) Typo in the runtime bullet.** “With the dat download” should read “With the data download”. I’d also name the Python version you tested on, since the README says “3.11 or newer” and I only tested 3.13.

**c) One unpinned direct dependency (`requirements.txt`).** `reproduce.py` imports `jupyter_client` directly, but it isn’t pinned. pip pulled it in transitively as 8.10.0. Add one line using whatever `pip freeze | grep jupyter_client` shows in your venv.

**4. Other things I noticed**

- `ai-appendix.md` still says TODO under `## Section 6`. The rubric requires `ai-appendix.md` to be filed. This conversation needs a P.R.I.M.E. entry and a full transcript there.
- **Commit dates.** The last two commits are dated Tue 24 Nov, 11:41 and 11:42. The PDF’s checkpoint list says the project was due Thu 19 Nov, while its header says “Due Sunday”. I can’t tell which deadline or extension applies to you.
- **Report table header.** In the report’s replication table, the “Difference” header wraps to “Differenc / e”.
- **Template files.** `report/README.md` and `AUDIT.md` are still the template versions. `AUDIT.md` has stale dates, such as Mon 16 Nov. These are optional to tidy.

My clone is at `/home/claude/clean_run/repo` if you want to compare your own run against it.
