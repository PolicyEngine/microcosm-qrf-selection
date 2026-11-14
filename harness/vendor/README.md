# Vendored helpers

`qrfdecomp.py` and `forest_bias.py` are byte-identical copies of
`harness/qrfdecomp.py` and `harness/forest_bias.py` in
[PolicyEngine/qrf-moop-shortfall](https://github.com/PolicyEngine/qrf-moop-shortfall)
at commit `dba3940`, under that repository's MIT licence (`LICENSE`, copied
here). The same-tree harness (`harness/selection.py`) calls their
`weighted_bootstrap`, `bootstrap_counts`, `fill_catchments` and
`recipient_estimates` unchanged.

Check: `git -C <qrf-moop-shortfall clone> diff dba3940 -- harness/qrfdecomp.py harness/forest_bias.py`
prints nothing, and `cmp` against these copies passes.
