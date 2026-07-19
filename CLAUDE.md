# tedana

## Environment

Development uses the micromamba environment **`tedenv`** (python 3.12.13; numpy,
scipy, nibabel, nilearn, pandas, scikit-learn, mapca, matplotlib). `tedana` is
installed editable from this checkout, so source edits take effect immediately
with no reinstall.

Run everything through it:

    micromamba run -n tedenv pytest tedana/tests/test_decay.py -q
    micromamba run -n tedenv flake8 tedana/decay.py
    micromamba run -n tedenv black tedana/decay.py

Use `tedenv`, not `tedanapy`. Both environments have tedana and past sessions
used them interchangeably, but only `tedenv` has the editable install and the
lint toolchain (`flake8`, `black`, `isort`). This project does not use ruff.
