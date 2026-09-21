# data/

Data is **never committed** (see `.gitignore`); only this note and `.gitkeep`
placeholders are tracked. Phase 1 (synthetic simulation) needs no files here — the
`MoleculeLibrary` is generated in memory from a seed.

Phase 2 layout (when implemented):

- `raw/` — downloaded BindingDB TSV, unmodified, with a recorded download date/version.
- `processed/` — standardized SMILES, deduplicated, single endpoint (e.g. pKd), with
  assay info / publication dates / target IDs / measurement type preserved (spec §9-10).
- `splits/` — compound / Bemis-Murcko scaffold / temporal / target-holdout splits, each
  with a manifest documenting how it was made and verified leakage-free (spec §11).
