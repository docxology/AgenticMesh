# Reproduction and limits

The repository verification guide runs syntax checks and calls the arithmetic,
environment, and agent interfaces with controlled inputs. The evidence
staging script reads the committed demo log and copies the archived belief
plot byte for byte. It records the source paths, plot checksum, and event
counts in the build. Render and validation use a pinned Template checkout.

The example uses stochastic action and environment sampling. A fresh driver
run can produce different trajectories and figures. The controlled check
tests the initial belief calculation and the policy interface. The archived
run demonstrates diagnostics for one trajectory. Neither supports claims
about calibration, performance in another environment, or general planning
quality. The code and trace support inspection of the specific implemented
update.
