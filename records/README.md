# Export reviewed regions for training

Each region asks one question: is there a visible obstruction inside this box? Reviewers can revise the same region or mark it uncertain. A whole-scene report about a ramp stays in the scene history and does not become a tile label. Include manually selected clear areas and hard negatives as well as proposals, so candidate generation does not determine which patches receive labels.

Choose photos with permission for the intended dataset use. Assign every selected photo to a street group using a non-identifying code. Group adjacent viewpoints and duplicates together. Keep each group entirely within one split. Have independent adjudicators inspect every anchor/test region. Record their binary decisions only after resolution; omit an unresolved photo from evaluation. Do not use crowd majority or model outputs as truth.

Create an assignments JSON using actual photo and region IDs from the review desk. Example:

```json
{
  "1": {"street": "block-a", "split": "train"},
  "2": {"street": "block-b", "split": "anchor", "truth": {"3": 1}},
  "5": {"street": "block-c", "split": "test", "truth": {"8": 0, "9": 1}}
}
```

Supply a private random key of at least 16 bytes in `ATLAS_EXPORT_KEY`. A consistent key links the same reviewer across exports. Use a fresh destination directory.

```sh
python -m records.export assignments.json dataset --db data/atlas.sqlite3
python -m vision.train dataset/manifest.jsonl --checkpoint field-model.npz
```

The atomic export includes normalized JPEGs, boxes, pseudonymous reviewer votes, group codes and the assigned splits. Unclear observations become null; omitted observations remain missing. Photo titles and GPS coordinates are excluded. Image content can still identify a place or person. The exporter rejects exact normalized-image and street-group leakage; a curator must identify overlapping viewpoints and renamed groups.
