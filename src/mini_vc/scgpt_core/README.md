`model.py`, `dsbn.py`, and `grad_reverse.py` are unmodified copies from the
official `scgpt==0.2.4` wheel, downloaded from PyPI. The upstream project is
https://github.com/bowang-lab/scGPT and is MIT licensed. This small subset is
used because the upstream package imports many unrelated training integrations
at package import time. The checkpoint is downloaded separately from the
official whole-human link in that repository; it is not distributed here.
