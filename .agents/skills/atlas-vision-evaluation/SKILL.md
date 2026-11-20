---
name: atlas-vision-evaluation
description: Implement or evaluate obstacle localization and crowd label fusion for Atlas when annotations are sparse, contradictory or geographically correlated.
---

# Evaluate physical obstacle evidence

- Define the annotation unit first: scene, localized region, observable property and observation time. Two claims about different properties can coexist. Preserve missing observations, abstentions and explicit negatives as distinct states.
- Split by street or collection group before training or tuning. Keep repeated captures, overlapping crops, and transformed copies together. Record seeds, manifest version, train/validation/test groups, positive counts and missing-label rates.
- Compare a visual baseline and an unweighted crowd baseline with the chosen fusion method. Keep the encoder, supervision budget, test observations and split fixed across ablations. If reliability is learned, use only training evidence and state the assumptions needed to identify annotator reliability.
- Report a rare-class metric with its denominator and a localization metric with the IoU rule. Report missed obstacles at a validation-selected threshold. Distinguish contradictory judgments from low image visibility. Identify unsupported classes and empty evaluation strata.
- Name the provenance of every checkpoint and result. Keep synthetic integration tests and field results clearly labeled. Verify that evaluation annotations cannot enter inference on held-out images unless the experiment explicitly tests crowd-assisted inference.
