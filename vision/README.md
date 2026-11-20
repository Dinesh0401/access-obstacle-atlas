# Localization research baseline

A wheelchair user might report an obstruction while a second person reports a visible ramp. Those are different attributes. Fit each physical attribute independently. This first model covers a single attribute, a visible obstruction in a reviewed region. It cannot infer an individual person's ability to travel through the scene.

## Model and learning objective

Each 16×16 RGB patch passes through two learned 3×3 convolution layers (8 and 12 channels) with ReLU, intermediate spatial pooling, global average pooling, and a Bernoulli output. Adam fits all convolution filters. At inference a 6×6 grid tiles the image. Boxes mark coarse cells and can split one object into several proposals. Predicting at multiple resolutions and comparing with object-level detection are open experiments.

Let the CNN output a logit f(x) for a latent presence label z. Estimate a separate confusion matrix C_r for each reporter from adjudicated anchor examples, with diagonal pseudo-count 2 and off-diagonal pseudo-count 1. Treat reports for a tile as conditionally independent given z. For observed votes v_r, compute

    q = sigmoid(f(x) + sum_r log(C_r[1,v_r] / C_r[0,v_r])).

The gradient of the negative marginal report log-likelihood with respect to f is sigmoid(f) − q. The trainer passes a detached q into binary cross-entropy differentiation. Missing/unsure votes contribute zero evidence. Skip a training patch when no usable vote exists. A fixed weak prior handles reporters absent from the anchor split. Repeated correlated votes can inflate apparent certainty; coordinated users and systematic visibility bias require separate study. Accurate independent anchor adjudication and reasonably stable reporter behavior are substantive assumptions.

The paired majority baseline uses identical images, initialization, steps, architecture and optimizer. Ties receive a soft target of 0.5. Both systems use vision alone on the test set. Average precision assesses rare-positive ranking, Brier score assesses probability error and the report includes recall/false positives at the prespecified threshold of 0.5.

## Reproduce the illustration

    OPENBLAS_NUM_THREADS=1 python -m vision.experiment

This writes demo.npz and results.json. The deterministic generator draws colored rectangles, seams and texture. It demonstrates the pipeline, loss and effects of noisy votes in a controlled setting. All accuracy measurements here refer to generated tiles. Reported synthetic street IDs identify disjoint batches; the generator does not simulate geographic shift. Real-street detection, object-box overlap, accessibility outcomes and probability calibration remain unmeasured.

## Train with independently labeled photographs

Create a JSON Lines manifest containing one row per reviewed region. Store images relative to the manifest. For each physical attribute use a separate manifest/checkpoint. Example row:

    {"image":"photos/crossing-06.jpg","box":[0.33,0.5,0.5,0.67],"street":"block-006","split":"train","votes":{"r1":1,"r2":0,"r3":null}}

Use region coordinates normalized to the EXIF-corrected image. Split into train, anchor and test **by street before tiling**. For anchor and test rows include numeric truth of 0 or 1 from independent adjudication. Record overlapping reports of the same attribute/region together, with one vote per reviewer. Generate negatives only from explicitly reviewed regions. Split capture bursts, overlapping intersections and duplicate photos together, including when filenames differ. The loader checks repeated exact street and image identifiers across splits.

    OPENBLAS_NUM_THREADS=1 python -m vision.train data/regions.jsonl --checkpoint trained.npz

For inference using a field checkpoint, load Network.load(path), tile with tile_image, and apply sigmoid to the model logits. Keep the bundled demo checkpoint independent from field experiments. For a credible portfolio claim, collect held-out neighborhoods, report the number of streets and positive regions per obstacle, repeat seeds, estimate paired street-clustered uncertainty, assess calibration and false negatives by street and lighting, and invite wheelchair users to review false negatives. The runner reports single-seed point estimates and a paired 95% percentile bootstrap interval for the difference in average precision, resampling entire test streets. This interval is conditional on the fitted checkpoints. Synthetic group IDs represent batches rather than physical neighborhoods.

## Measuring localization and switching checkpoints

The synthetic localization benchmark renders 20 independent 6×6 scenes with 84 object rectangles, ranks every cell and greedily matches by IoU, at most once per object. Its AP at IoU 0.1 was 0.539 and at IoU 0.5 was 0.000. Fixed cells identify vicinity much better than exact object extent in this benchmark. A trained box-regression or segmentation head and field-labelled boxes would be needed to evaluate accurate outlines.

Set ATLAS_MODEL_CONFIG to a JSON file to serve a different checkpoint. It must name the checkpoint relative to the configuration file, give its file SHA-256 digest and specify name, training_data, status, scope and threshold. The loader verifies the digest and fails closed on malformed configurations. For example, after training, supply these fields with actual values:

    {"checkpoint":"trained.npz","sha256":"<actual digest>","name":"Field pilot v1","training_data":"<collection and version>","status":"<evaluation status>","scope":"<supported obstacle and geography>","threshold":0.5}

Restart the server after changing a deployment's checkpoint or configuration.
