# Synthetic Active Inference model

The worked example draws on discrete Active Inference [@friston2017; @dacosta2020].
Its hidden states and observations share the names Low, Medium, and High.
The agent samples a choice among Decrease, Stay, and Increase. These choices
drive transitions in the local simulated environment.

Let $A$ denote observation likelihoods and $B_a$ the state-transition matrix
for action $a$. Column vectors index state beliefs. The prior is $D$ on the
first call and $B_a q$ on later calls, where $q$ is the previous posterior
and $a$ is the previous sampled action. For observation $o$, the state score
sums the negative log likelihood and negative log prior. A softmax of the
negated score gives the posterior, with a small constant inside logarithms.

For each prospective action, the code predicts a state distribution and
projects it through $A$ to predict observations. The implemented action score
subtracts predicted-minus-current state entropy from a preference cross-entropy.
The preference distribution is obtained by normalizing the exponentiated
scores $C$. The preference term is the cross-entropy implemented in the
repository. It omits the predicted observation entropy term that appears in
a full Kullback–Leibler divergence.

The habit vector $E$ initializes the policy prior. Each step combines the prior
with a softmax of the negated action score, normalizes the updated prior, then
combines it with the score again to form the sampled policy. The sampled
action and the belief, score, and policy arrays are included in the return
value. The default constructors configure the example. The adjacent YAML
serves as example configuration for the separately callable utilities.
