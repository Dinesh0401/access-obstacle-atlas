"""Public entry point for the Python street-image review desk."""

import json
from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image

from records import AtlasStore
from ui.scene import evidence_image, example_crossing
from ui.theme import apply_theme
from vision.api import MODEL_CARD, detect

st.set_page_config(page_title="Access Obstacle Atlas", page_icon="◩", layout="wide")
apply_theme()
store = AtlasStore()
OBSTACLES = (
    "Blocked curb ramp",
    "Missing curb ramp",
    "Broken surface",
    "Narrow passage",
    "Other obstruction",
)
DEMO_MARKS = [
    ("Parked scooter", (0.365, 0.31, 0.54, 0.525)),
    ("Curb ramp", (0.33, 0.48, 0.609, 0.577)),
]


def encode(image):
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def open_scene(photo_id):
    st.session_state.scene_id = photo_id
    st.session_state.next_page = "Review desk"
    st.rerun()


def add_photo():
    st.header("Add a street photo")
    st.write(
        "Choose a clear view of the crossing or footpath. Photos and reports are shared with other reviewers on this instance."
    )
    with st.form("photo-upload"):
        upload = st.file_uploader("Street photo", type=["jpg", "jpeg", "png", "webp"])
        title = st.text_input("Location or scene title", max_chars=120)
        locate = st.checkbox("Include coordinates on the atlas")
        east, north = st.columns(2)
        lat = east.number_input("Latitude", -90.0, 90.0, format="%.6f")
        lon = north.number_input("Longitude", -180.0, 180.0, format="%.6f")
        consent = st.checkbox(
            "I can share this photo and have removed identifying details."
        )
        if st.form_submit_button("Add to review queue", type="primary"):
            if upload is None or not title.strip() or not consent:
                st.error("Choose a photo, enter a title and confirm you can share it.")
            else:
                try:
                    ident = store.add_photo(
                        upload.getvalue(),
                        title,
                        lat if locate else None,
                        lon if locate else None,
                    )
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    open_scene(ident)


def show_reviews(photo_id):
    st.subheader("Community evidence")
    summaries = store.summaries(photo_id)
    if summaries:
        for item in summaries:
            st.markdown(f"**{item['obstacle']}** · {item['status']}")
            st.caption(
                f"{item['present']} present · {item['absent']} absent · {item['uncertain']} unclear"
            )
    else:
        st.caption("This scene is waiting for its first observation.")
    with st.form(f"report-{photo_id}"):
        category = st.selectbox("Obstacle", OBSTACLES)
        verdict = st.radio(
            "What can you see?", ["present", "absent", "uncertain"], horizontal=True
        )
        reviewer = st.text_input(
            "Reviewer alias",
            help="Use the same alias to revise your observation.",
            max_chars=80,
        )
        if st.form_submit_button("Save observation", type="primary"):
            try:
                store.report(photo_id, reviewer, category, verdict)
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.session_state.notice = "Observation saved."
                st.rerun()


@st.cache_data(max_entries=40, show_spinner=False)
def proposals(content):
    with Image.open(BytesIO(content)) as image:
        return detect(image.convert("RGB"))


def region_review(photo_id, marks):
    with st.expander("Inspect or annotate a specific region"):
        st.write(
            "Choose a model tile, revisit an existing region, or describe a missed obstacle with a custom box."
        )
        saved = store.region_summaries(photo_id)
        for row in saved:
            st.caption(
                f"Region {row['id']} · {row['status']} · {row['present']} present, {row['absent']} absent, {row['uncertain']} unclear"
            )
        boxes = {f"Saved region {r['id']}": r["box"] for r in saved}
        boxes.update(
            {f"Candidate {i} · {label}": box for i, (label, box) in enumerate(marks, 1)}
        )
        which = st.selectbox(
            "Region to inspect", ["Custom region", *boxes], key=f"region-{photo_id}"
        )
        if which == "Custom region":
            st.caption("Set boundaries as a percentage from the top or left edge.")
            horizontal = st.slider("Left and right (%)", 0, 100, (25, 75))
            vertical = st.slider("Top and bottom (%)", 0, 100, (25, 75))
            box = [
                horizontal[0] / 100,
                vertical[0] / 100,
                horizontal[1] / 100,
                vertical[1] / 100,
            ]
        else:
            box = boxes[which]
        return box


def vote_on_region(photo_id, box):
    with st.form(f"vote-region-{photo_id}"):
        st.write("Does this outlined region contain a visible obstruction?")
        verdict = st.selectbox("Region observation", ["uncertain", "present", "absent"])
        alias = st.text_input("Alias for region observation", max_chars=80)
        if st.form_submit_button("Save region observation"):
            if not alias.strip():
                st.error("Enter a reviewer alias.")
                return
            try:
                region_id = store.add_region(photo_id, box)
                store.report_region(region_id, alias, verdict)
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.session_state.notice = "Region observation saved."
                st.rerun()


def review(photos):
    st.title("Street review desk")
    st.write(
        "Inspect the crossing, locate barriers and record what the image actually shows."
    )
    choices = {
        None: "Illustrated example · Market crossing",
        **{p["id"]: f"{p['title']} · #{p['id']}" for p in photos},
    }
    current = st.session_state.get("scene_id")
    ids = list(choices)
    ident = st.selectbox(
        "Review scene",
        ids,
        index=ids.index(current) if current in ids else 0,
        format_func=choices.get,
    )
    st.session_state.scene_id = ident
    left, right = st.columns([1.8, 1], gap="large")
    with left:
        if ident is None:
            image = example_crossing()
            marks = DEMO_MARKS
            st.subheader("Market crossing")
            st.caption("Illustrated example · hand-marked regions")
        else:
            content = store.get_image(ident)
            image = Image.open(BytesIO(content)).convert("RGB")
            st.subheader(next(p["title"] for p in photos if p["id"] == ident))
            st.caption(
                f"Uploaded street photo · {MODEL_CARD['name']} · {MODEL_CARD['status']}"
            )
            with st.spinner("Locating candidate regions…"):
                results = proposals(content)
            marks = [(f"{r['label']} {r['score']:.2f}", r["box"]) for r in results]
        overlay = st.toggle("Show evidence regions", value=True)
        active_box = None
        if ident is not None and st.checkbox(
            "Review a specific region", key=f"inspect-{ident}"
        ):
            active_box = region_review(ident, marks)
        visible_marks = (
            [("Selected region", active_box)] if active_box is not None else marks
        )
        rendered = evidence_image(image, visible_marks if overlay else [])
        st.image(rendered, width="stretch")
        if ident is None:
            st.caption(
                "A scooter overlaps the entrance to a curb ramp beside the zebra crossing."
            )
        elif marks:
            st.caption(
                "Candidate locations in normalized image coordinates (left, top, right, bottom). Model scores are uncalibrated."
            )
            st.dataframe(
                [
                    {"Candidate": label, "Region": ", ".join(f"{v:.2f}" for v in box)}
                    for label, box in marks
                ],
                hide_index=True,
            )
        else:
            st.info(
                "The model returned no proposals. Accessibility still needs human review."
            )
        st.download_button(
            "Download current evidence view",
            encode(rendered),
            "crossing-review.png",
            "image/png",
        )
    with right:
        if ident is None:
            st.subheader("Read the scene")
            st.markdown(
                "**01　Parked scooter**  \nThe front wheel occupies the ramp entrance."
            )
            st.markdown("**02　Curb ramp**  \nThe ramp is visible behind the scooter.")
            st.divider()
            st.write(
                "Two reviewers can agree a ramp exists and disagree about whether someone can use it. Record each obstacle separately."
            )
            st.caption("Open Add photo to review an actual image.")
        else:
            if active_box is not None:
                vote_on_region(ident, active_box)
            show_reviews(ident)


def atlas(photos):
    st.title("Community atlas")
    st.write("Find a location or review conflicting accounts of the same obstacle.")
    counts = st.columns(3)
    counts[0].metric("Street photos", len(photos))
    counts[1].metric("Observations", sum(p["report_count"] for p in photos))
    counts[2].metric(
        "Scenes with conflict", sum(bool(p["conflict_count"]) for p in photos)
    )
    points = [
        {"lat": p["latitude"], "lon": p["longitude"]}
        for p in photos
        if p["latitude"] is not None and p["longitude"] is not None
    ]
    if points:
        st.map(points, latitude="lat", longitude="lon")
    historical = Path(__file__).parent / "data_tools" / "sample-dc.json"
    if historical.exists() and st.checkbox(
        "Explore historical Washington, DC observations"
    ):
        features = json.loads(historical.read_text())["features"]
        history = []
        for point in features:
            lon, lat = point["geometry"]["coordinates"]
            info = point["properties"]
            history.append(
                {
                    "Latitude": lat,
                    "Longitude": lon,
                    "Region": str(info.get("region", "")),
                    "Captured": str(info.get("capture_date", "")),
                }
            )
        st.caption(
            "Historical crowd observations from Project Sidewalk. Current street conditions have not been checked."
        )
        if history:
            st.map(history, latitude="Latitude", longitude="Longitude")
            st.dataframe(history, hide_index=True)
    st.subheader("Review queue")
    if not photos:
        st.info("No street photos yet. Add a photo to start the atlas.")
        return
    conflicts = st.checkbox("Only show conflicting reports")
    for photo in photos:
        if conflicts and not photo["conflict_count"]:
            continue
        name, action = st.columns([4, 1])
        name.text(photo["title"])
        if photo["latitude"] is not None:
            name.caption(
                f"Location · {photo['latitude']:.5f}, {photo['longitude']:.5f}"
            )
        name.caption(
            f"{photo['report_count']} observations · {photo['conflict_count']} disputed categories"
        )
        if action.button("Review", key=f"open-{photo['id']}"):
            open_scene(photo["id"])
    if conflicts and not any(p["conflict_count"] for p in photos):
        st.caption("No conflicting reports in the current queue.")


def research():
    st.title("Inside the experiment")
    st.subheader("Can a rare barrier survive disagreement?")
    st.write(
        "A missed curb or a temporary obstruction can change whether someone can use a crossing. The research question is whether retaining sparse, conflicting observations improves rare-obstacle localization on streets the model has never seen."
    )
    st.markdown("### What runs here")
    st.caption(f"{MODEL_CARD['name']} · {MODEL_CARD['status']}")
    st.text(f"Training data: {MODEL_CARD['training_data']}")
    st.text(f"Scope: {MODEL_CARD['scope']}")
    st.write(
        "The review desk displays candidate image regions from the active detector. The queue preserves present, absent and uncertain responses per obstacle. Each alias can revise its own observation."
    )
    results = json.loads(
        (Path(__file__).parent / "vision" / "results.json").read_text()
    )
    st.markdown("### Held-out synthetic tiles")
    st.caption(
        f"{results['test_tiles']} test tiles · {results['test_positives']} positive tiles · fixed seed {results['seed']}"
    )
    st.table(
        [
            {
                "Method": name.title(),
                "Average precision": round(values["average_precision"], 3),
                "Recall at 0.5": round(values["recall_at_0.5"], 3),
                "Brier score": round(values["brier_score"], 3),
            }
            for name, values in results["models"].items()
        ]
    )
    st.markdown("### What still needs field evidence")
    st.write(
        "Evaluate on independently adjudicated street photographs with held-out neighborhoods. Compare against majority-vote supervision, count missed rare obstacles, and measure review workload at matched recall. A model score alone cannot establish whether a crossing is usable."
    )


if "next_page" in st.session_state:
    st.session_state.navigation = st.session_state.pop("next_page")
page = st.radio(
    "Workspace",
    ["Review desk", "Community atlas", "Add photo", "Research"],
    horizontal=True,
    key="navigation",
)
if notice := st.session_state.pop("notice", None):
    st.success(notice)
if page == "Review desk":
    review(store.list_photos())
elif page == "Community atlas":
    atlas(store.list_photos())
elif page == "Add photo":
    add_photo()
else:
    research()
