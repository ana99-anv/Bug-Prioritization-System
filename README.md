# Bug-Prioritization-System

 

I built this project to answer a question that seemed simple at first: can machine learning help engineering teams figure out which bugs actually need urgent attention?

It turned out to be a lot less simple than I expected and honestly, most of what I learned came from things not working the way I assumed they would. This README walks through that journey, not just the final numbers.

[**Try Live App **](https://bug-prioritization-system-xusnsps3advoappxvgfaipg.streamlit.app/)

The short version

I trained a model to predict bug priority (P1–P5) from ~10,000 historical Bugzilla bug reports. The first version technically "worked" with 66.75% accuracy but that number was hiding a real problem: the model was great at guessing the most common priority and terrible at catching the bugs that actually mattered. So I reframed the whole thing around a more useful question: is this bug urgent or not , and thereby spent most of the project figuring out how to tune that decision properly, rather than chasing a bigger accuracy number.

I also tested the model on a later time window to see if it would hold up outside the data it was trained on. It didn't hold up as well as I'd hoped, and that turned into one of the more interesting parts of the project.

The data

Bugzilla bug reports, cleaned down to about 10,000 records, with a summary, component, type, OS, version, and priority label.

Priority	Share of data
P1	3.0%
P2	11.6%
P3	65.7%
P4	2.7%
P5	17.1%

P3 dominates. That imbalance shapes almost every decision in this project, so it's worth stating up front instead of burying it.

What I tried first (and why it wasn't good enough): 

I built a 5-class XGBoost model on TF-IDF text features (from the bug summary) plus one-hot encoded categorical features (type, component, OS, version) and a couple of engineered numeric features (summary length, word count).

Accuracy: 66.75%. Looks fine on its own. Except just guessing the majority-class baseline gets you 65.65%. So the model was only 1.1 points better than doing nothing. And when I looked at per-class recall, P1 (the priority that actually matters most) was only being caught 15% of the time. The model was basically hiding behind the P3 majority class.

I tried class weighting to fix this. It helped , P1 recall jumped to 50% but then accuracy dropped to 50.85%. That's a real trade-off, not a free win, and I didn't feel good shipping a model that traded that much overall correctness for the recall gain. So I didn't use it as the final model. I'm including it here because I think showing what didn't make the cut is as useful as showing what did.

Reframing the problem

At some point I stepped back and asked what an engineering team actually needs from this. Nobody's day-to-day decision hinges on "is this exactly a P2 or exactly a P3." What they actually care about is: does this need urgent attention or not.

So I collapsed the problem:

P1 + P2 → Urgent
P3 + P4 + P5 → Normal

This is a product decision, not something the data told me to do. 

Urgent bugs made up about 14.5% of the test set, so the new baseline (predict "Normal" for everything) was 85.45% accuracy. A binary XGBoost model at the default 0.5 threshold got 85.9% accuracy is technically better than baseline , but caught only 12.37% of actual urgent bugs. High accuracy, almost useless in practice. This was the moment it really clicked for me that accuracy was the wrong thing to optimize for this problem.

Picking a threshold on purpose

Instead of using the default 0.5 cutoff, I swept through thresholds and looked at the precision/recall trade-off at each one.

| Threshold | Precision | Recall | % of Bugs Routed to Urgent |
|---:|---:|---:|---:|
| 0.50 | 57.1% | 12.4% | 3.2% |
| 0.30 | 45.4% | 37.5% | 12.0% |
| 0.20 | 37.5% | 47.1% | 18.3% |
| 0.10 | 24.7% | 78.0% | 46.0% |
| **0.097** | **24.4%** | **80.1%** | **47.9%** |

I set a target: catch at least 80% of true urgent bugs. That landed on a threshold of 0.097.

I also tried a different approach, assuming a missed urgent bug (false negative) costs 10x as much as an unnecessary review (false positive), and finding the threshold that minimizes that assumed cost. That came out to 0.08, with 87.3% recall.

I ended up going with 0.097 as my main choice. My reasoning: "catch 80% of urgent bugs" is something a team can actually understand and agree on, like an SLA. The 10:1 cost ratio was something I assumed, not something I measured, so I didn't feel comfortable making it the primary recommendation, but I kept it in to show what the trade-off looks like if you optimize for cost instead of recall directly.

Does this actually hold up on future bugs?

This is the part I almost skipped, and I'm glad I didn't.

Everything above uses a random train/test split, which assumes the test set looks statistically like the training set. That's not really true for a system meant to be deployed forward in time. So I re-ran the same pipeline using a chronological split — training on an earlier period, testing on a later one — to get a more honest sense of how this would behave in production.

| Metric | Random Split | Chronological Split |
|---|---:|---:|
| Accuracy | 66.75% | 61.70% |
| Macro-F1 | 0.326 | 0.303 |
| Urgent Recall @ Default 0.5 Threshold | 12.37% | 3.10% |
| Recalibrated Threshold for 80% Recall | 0.097 | 0.0658 |
| Precision at 80% Recall | 24.4% | 41.4% |

The accuracy/F1 drop is fairly mild on its own. What's actually alarming is the base rate of urgent bugs nearly tripled between the two periods — from about 10.4% of training data to 31.0% of the later test window — and the default threshold's recall collapsed to 3.1%. A threshold that worked fine at training time basically stopped working once the underlying data shifted.

One important note: I don't have a reliable bug creation date in this dataset, so I used updated_date as the closest available proxy for chronological order. That means the "future" window might include bugs that were reopened or re-triaged, not just newly filed ones — which could partly explain the base-rate jump (urgent bugs get touched more often because they're urgent). I couldn't fully verify this against the raw data, so I'm treating this result as an upper-bound estimate of drift, not a precise simulation of deployment.

The takeaway I actually trust: a threshold picked once during development isn't safe to leave fixed forever. It needs to be monitored and recalibrated as the data shifts, the same way you'd manage a fraud or spam threshold.

The app

I built a Streamlit app on top of this that has a home page, a dashboard with the threshold comparison table, a single-bug prediction page with an adjustable threshold slider, a batch-prediction page that takes a CSV and returns a downloadable file of predictions, and a model-insights page that shows the precision/recall trade-off and the cost assumptions explicitly, instead of hiding them in a notebook.

The threshold is a slider, not a fixed number baked into the code — because after seeing how much the "right" threshold moved under the temporal test, I didn't want to pretend there's one correct value.

Limitations: 
- Historical priorities came from human triage decisions, which may be inconsistent or biased the model inherits whatever patterns are in that history, good or bad.
- TF-IDF only captures word overlap, not meaning two bugs describing the same issue in different words might get very different treatment.
- The 10:1 false-negative/false-positive cost ratio is an assumption I made up as a reasonable starting point, not something calibrated from real operational data.
- "P1/P2 = Urgent" is a framing choice I made, not an objective fact about the data.
- The temporal split used updated_date, not a true creation date, for the reasons explained above.
- Predicted probabilities weren't explicitly calibrated, so the thresholds assume the model's raw probability outputs are reasonably trustworthy.
- This should be monitored after deployment bug types, components, and priority patterns can and will drift.

Where this leaves things

- The project moved from "predict the exact priority" to something more useful: a system that flags bugs as urgent or not, with a threshold that can be adjusted based on how much review capacity a team has and how costly a missed urgent bug actually is. The main finding, honestly, isn't a number, it's that the threshold matters more than the model, and that whatever threshold you pick needs to be revisited over time, not set once and forgotten.

This is built to be a decision-support tool, not an autonomous system. A human still makes the final call the model just helps make sure the right bugs get looked at first.

Next steps, if I kept going
- Calibrate the cost ratio with real data instead of an assumption.
- Get a real bug creation date and re-run the temporal validation properly.
- Add BERT-style embeddings to see if they pick up on semantic similarity that TF-IDF misses.
- Build the monitoring loop I keep describing but haven't actually implemented  tracking predicted vs. actual urgent rate over time and triggering re-calibration automatically.
