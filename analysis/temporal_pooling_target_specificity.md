# Is the temporal attention target-specific? (Reviewer 2 #2)

Answer from the code. No computation needed.

## Direct answer

**No. The temporal pooling is shared across all four targets.** It is
*per-input-channel*, not per-target. Any claim in the paper that the model
produces target-specific temporal importance must be weakened.

The variable attention, by contrast, **is** target-specific. So the correct
statement is: *the model produces one temporal importance profile per input
channel, shared across targets, and a separate variable-importance vector for
each target.*

## The code

The two queries are declared with different shapes
([src/xm_models/xai_meteoformer.py:188-190](src/xm_models/xai_meteoformer.py#L188-L190)):

```python
self.temporal_query = nn.Parameter(torch.randn(d_model) * 0.02)                      # (d_model,)
self.variable_query = nn.Parameter(torch.randn(self.n_targets, d_model) * 0.02)      # (n_targets, d_model)
```

`temporal_query` has **no target axis at all** — it is a single vector. Its use
([line 242-244](src/xm_models/xai_meteoformer.py#L242-L244)):

```python
scores = (z @ self.temporal_query) / math.sqrt(self.d_model)   # (BN, P)
alpha  = torch.softmax(scores, dim=-1)
h_t    = (alpha.unsqueeze(-1) * z).sum(dim=1)                  # (BN, d_model)
```

`z` is `(B*N, P, d_model)` where `N` is the number of **input channels** and `P`
the number of patches. So `alpha` is indexed by (sample, input channel, patch).
There is no target index anywhere in this computation, and `h_t` is produced
once and reused for every target.

`variable_query` does carry the target axis
([line 254](src/xm_models/xai_meteoformer.py#L254)):

```python
vs = torch.einsum("bnd,td->btn", h_t, self.variable_query)     # t = target
beta = torch.softmax(vs, dim=-1)                               # (B, n_targets, N)
```

so target-specificity enters **only** at the variable-mixing stage, operating on
representations that have already been pooled over time identically for all
targets.

The same holds for the exported explanations: `out["temp_attn"]` is
`alpha.view(B, N, num_patches)` and `out["rollout"]` likewise — both are
`(B, N, P)`, with no target dimension, while `out["var_attn"]` is
`(B, n_targets, N)`.

## What to change in the paper

- Replace any phrasing like "target-specific temporal attention" with
  "per-variable temporal attention, shared across the four targets".
- The dual-attention contribution is still intact: temporal importance is
  per-input-channel (which is more than a single global profile), and variable
  importance is genuinely per-target.
- If target-specific temporal importance is wanted as a *contribution*, it is a
  one-line change — make `temporal_query` a `(n_targets, d_model)` parameter and
  pool once per target — but that is a new model and needs retraining, so it
  belongs in future work, not in this resubmission.
