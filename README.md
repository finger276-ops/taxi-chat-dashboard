# Patch v18 — robust CSV normalization

Fixes upload crashes like:

`ValueError: Length of values (0) does not match length of index (...)`

Reason: in some CSV exports, one of the columns used for chat/author identification, such as `Профиль блога` or `Блог`, can be absent. The previous preprocessing code used scalar defaults, which produced empty lists during `zip(...)`.

Replace at minimum:

```text
src/preprocess.py
```

Then commit and push to GitHub. Streamlit Cloud will redeploy the app.

The patch keeps full datetime internally, but does not affect the visual date formatting implemented earlier.
