# Field evidence

The 29 locations in `sample-dc.json` are historical Project Sidewalk crowd annotations of obstacles in Washington, DC. Their imagery capture dates span April 2014 through October 2017. Within this small spatial sample the separate validation events comprise one AI agreement and one AI uncertainty vote. No human validation votes appear. The original annotations and the subsequent validation events represent different actions.

## Source and reproducibility

- Data provider: Project Sidewalk, Washington DC deployment, [Raw Labels v3 documentation](https://sidewalk-dc.cs.washington.edu/v3/api-docs/rawLabels).
- Metadata rights: Project Sidewalk describes public API label exports as CC0 in [its data policy](https://sidewalk-cdmx.cs.washington.edu/about). This snapshot includes geographic label metadata only. Street-view imagery has separate provider terms.
- Retrieved 2026-11-20. SHA-256 of the original response bytes: `c4e2bc06be7b16b9378df578e92c2818ea696e65f6d1eb799178791b04117119`. The upstream dataset may change.
- The query selects obstacle labels inside one bounding box, including labels with zero validation votes. It cannot estimate citywide obstacle prevalence. A label is a historical point observation. Collect separately consented images with matched boxes and geographically separated sites for visual evaluation.

From the repository root:

```sh
curl -fL 'https://sidewalk-dc.cs.washington.edu/v3/api/rawLabels?bbox=-77.035,38.900,-77.027,38.905&labelType=Obstacle&filetype=geojson' -o raw-dc.geojson
python -m data_tools.project_sidewalk raw-dc.geojson --city washington-dc --out dc.geojson
python -m unittest data_tools.test_project_sidewalk
```

The importer retains label ID, position, dates, region, street segment ID and votes stratified by validator type. It omits user IDs, free text, imagery links and photographs. A missing class in a vote dictionary represents zero votes. Multiple nearby labels can refer to the same physical object. Human and machine labels require independent adjudication before use as evaluation targets.
