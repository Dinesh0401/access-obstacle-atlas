# Access Obstacle Atlas

A Python computer-vision workbench for reviewing sidewalk barriers. A curb ramp can be visible while a parked scooter blocks its entrance. Atlas gives reviewers the image, the candidate regions, and the observations behind a disagreement.

[Open the public portfolio demo](https://access-obstacle-atlas.bubblybobcat3.chatgpt.site) · [Export reviewed regions for training](records/README.md)

The public site runs COCO-trained SSDLite MobileNet v2 on the selected photograph in a browser worker using the TensorFlow.js CPU backend. Review detected objects or draw a missing region, record the access impact, and revisit the observation on this device. Portfolio reviews stay in the current browser. The shared queue, training pipeline and patch-CNN experiment run in the Python service below. To serve the public interface locally, run `python portfolio/prepare_assets.py`, then `python -m http.server --directory portfolio 8000`.

## Try the review desk

Use Python 3.12. From the repository root:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the illustrated crossing, toggle its marked regions, then add a photo you have permission to share. Open the same scene under two reviewer aliases and record opposite verdicts about the same obstacle. Open **Community atlas** and filter the review queue for conflicting reports. The original illustration uses hand-marked regions. Uploaded photos run through the bundled CNN.

For the optional real-photo object detector, install prerequisites with `python -m pip install -r requirements-field.txt` and run `python -m vision.field --prepare`. On the review desk, select the field photograph and enable pretrained physical-object detections. A motorcycle box identifies a vehicle. A reviewer still needs to judge whether it blocks the guidance strip. The [field photograph credits](data_tools/field-example.md) document the source.

For a single-instance deployment with persistent storage:

```sh
docker build --build-arg FIELD_OBJECTS=1 -t access-obstacle-atlas .
docker run --rm -p 8501:8501 -v atlas-data:/app/data access-obstacle-atlas
```

The server shares submitted photos and optional coordinates with visitors to that instance. Image processing normalizes orientation and removes metadata. Visible people, addresses and license plates still require review before upload. Reviewer aliases are self-asserted and can be reused by other visitors. The service is intended for a trusted review group. It does not provide route guidance.

## What is being tested?

Can information about reviewer reliability improve visual learning when positive regions are uncommon and raters disagree? The paired experiment uses the same CNN, initialization and training images. The baseline trains on majority labels. The alternative uses independently adjudicated anchor regions to estimate how each annotator reports positive and negative examples. Both predict from pixels alone at test time. The alternative consumes additional anchor information; the comparison measures the combined value of that information and the fusion rule.

| System component | Implementation |
| --- | --- |
| Review desk, upload, evidence overlays, accessible text summary | `app.py`, `ui/` |
| Shared photo storage, atomic report revisions and conflict queue | `records/` |
| Two-layer patch CNN, gradient computation and localized inference | `vision/network.py`, `vision/api.py` |
| Annotator confusion matrices, marginal likelihood and baseline training | `vision/train.py` |
| Reproducible generated benchmark and model weights | `vision/experiment.py`, `vision/demo.npz`, `vision/results.json` |
| Historical Washington, DC obstacle points and importer | `data_tools/` |
| Reusable interface-review and model-evaluation instructions | `.agents/skills/` |

### Evidence so far

On the seeded procedural experiment, 38 of 250 test patches contained generated obstructions. At the fixed 0.5 threshold the majority-supervised CNN recovered 9 and the reliability-aware CNN recovered 19. Both produced zero false positives among the 212 negative test patches. Average precision was 0.844 and 0.887 respectively. This single synthetic run measures patch recognition. On a separate synthetic localization test with 84 objects in 20 scenes, average precision was 0.539 at IoU 0.1 and **0.000 at IoU 0.5**. The current grid finds candidate areas but does poorly at drawing tight object boxes. The generated objects are much simpler than real street scenes.

The 29 Project Sidewalk points in the field sample are historical observations made from Washington, DC imagery. They supply real examples for map exploration. They contain no street-image pixels or adjudicated region masks. The model has not been validated on those locations.

### Reproduce and extend

```sh
OPENBLAS_NUM_THREADS=1 python -m vision.experiment
python -m pip install pytest
python -m pytest -q
```

The [model and manifest guide](vision/README.md) defines the probability model, training format and limitations. The [field data guide](data_tools/README.md) records the source, retrieval command and usage terms.

A field evaluation needs consented photographs, independently reviewed positive and negative regions, and entire streets held out before training. Compare models on the same held-out locations, count rare positives and missed barriers, score region overlap, and record how often someone must inspect a proposal. Measure whether users can make a better decision from the evidence. Agreement with a synthetic label cannot answer that question.

### Invite another reviewer

In the Python review desk, open a submitted photo and select **Review a specific region**. Outline a region and save an observation. Copy the resulting page URL, which contains both the photo and region IDs. A second reviewer can open that URL on the same running instance, enter a different alias, and submit an independent observation. The Community atlas places conflicting reports first. Use a shared server with persistent storage when reviewers are on different devices.
