---
description: Launch the LP-1 Startup Funding Analyzer Streamlit app locally and verify it serves.
---

# Run LP-1 Startup Funding Analyzer

## Prerequisite: Python 3.13 venv (NOT the global Python)

The machine's global Python 3.14 breaks Streamlit imports
(`TypeError: Metaclasses with custom tp_new are not supported`). Always use
the project venv:

```bash
cd "E:/school/Azubi Africa/LP-1-Project"
py -3.13 -m venv .venv                      # first time only
.venv/Scripts/pip.exe install -r requirements.txt
```

## Launch

```bash
.venv/Scripts/python.exe -m streamlit run app.py \
  --server.port=8501 --server.address=127.0.0.1 --server.headless=true
```

## Verify it serves

- `curl http://127.0.0.1:8501/_stcore/health` → `ok`
- `curl -s http://127.0.0.1:8501/` → HTTP 200, Streamlit index

## Drive the golden path (no browser needed)

Use Streamlit's built-in AppTest harness:

```bash
.venv/Scripts/python.exe - <<'EOF'
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("app.py", default_timeout=90).run()
assert not at.exception
at.sidebar.radio[0].set_value("Funding predictor").run()
at.button[0].click().run()
assert not at.exception, [str(e.value) for e in at.exception]
print([(m.label, m.value) for m in at.metric])
EOF
```

Expected: metrics like `Estimated funding $15.04M`, `Readiness probability 43%`,
`Description fit: Fintech` (values vary slightly with input).

## Notes

- Models in `models/` are committed, hash-verified, and trained with
  scikit-learn 1.8.x — the pinned `requirements.txt` must stay on 1.8.x.
- `verify_hash()` is case-insensitive on purpose: the old `.sha256`
  sidecars were written uppercase by PowerShell.
- The app loads models at startup; if `/health`-equivalent widgets are empty,
  check `models/*.pkl` integrity first.
