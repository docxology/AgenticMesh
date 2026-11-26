# Controlled check and archived run

In the controlled check, a synthetic environment call returns Medium with
certainty. The fresh agent begins from equal state probabilities. Its
posterior assigns two thirds to Medium and one sixth to each remaining state.
The check compares the computed posterior to those fractions and verifies
that the policy is normalized and the sampled action belongs to the declared
action set. It leaves the sampled action unconstrained within that set.

The committed January 2025 run contains 50 agent-step and 50 environment-step
events. These counts come from the event fields in the archived JSON lines.
Figure @fig:archived-belief-evolution is the repository's diagnostic plot from
the same dated run bundle. Its lines show the recorded belief components as
the synthetic agent stepped through observations.

![Low, Medium, and High belief probabilities across the archived demo.](../../output/figures/archived-belief-evolution.png){#fig:archived-belief-evolution width=95%}
