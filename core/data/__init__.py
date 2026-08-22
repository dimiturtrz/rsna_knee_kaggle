"""The data layer: what a study IS (`corpus`) and what it is LABELLED with (`labels`).

Below `kneemri`, which trains on it. `corpus` imports nothing outside itself and `core.paths`; a
trainer, a viewer and a Kaggle kernel all read this layer and none of them may be imported back into
it. That one-way edge is what import-linter enforces at the package level, and it is the reason this
lives under `core` rather than beside the models that consume it.
"""
