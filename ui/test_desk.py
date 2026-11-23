"""Exercise the normal Streamlit path with a fresh persistent evidence store."""

from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from records import AtlasStore
from ui.scene import evidence_image


def test_upload_observe_and_reopen(tmp_path, monkeypatch):
    monkeypatch.setenv("ATLAS_DB", str(tmp_path / "atlas.sqlite3"))
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run()
    assert not app.exception
    assert app.title[0].value == "Street review desk"
    app.radio(key="navigation").set_value("Add photo").run()
    app.button[0].click().run()
    assert app.error
    buf = BytesIO()
    Image.new("RGB", (128, 200), "#eeeccc").save(buf, format="PNG")
    app.get("file_uploader")[0].upload("footpath.png", buf.getvalue(), "image/png")
    app.text_input[0].set_value("Station corner")
    app.checkbox[1].check()
    app.button[0].click().run()
    assert not app.exception
    assert app.title[0].value == "Street review desk"
    for alias, verdict in [("rider", "present"), ("walker", "absent")]:
        app.text_input[0].set_value(alias)
        app.radio[1].set_value(verdict)
        app.button[0].click().run()
        assert not app.exception
    assert AtlasStore().summaries(1)[0]["conflict"]
    app.radio(key="navigation").set_value("Community atlas").run()
    next(
        control
        for control in app.checkbox
        if control.label == "Only show conflicting reports"
    ).check().run()
    app.button[0].click().run()
    assert app.title[0].value == "Street review desk"
    assert any("conflicting" in item.value for item in app.markdown)


@pytest.mark.parametrize("dimensions", [(300, 150), (150, 300)])
def test_overlay_normalized_to_photo(dimensions):
    image = Image.new("RGB", dimensions, "white")
    result = evidence_image(image, [("obstruction", (0.25, 0.25, 0.75, 0.75))])
    assert result.getpixel((int(0.75 * image.width), int(0.75 * image.height))) == (
        255,
        208,
        102,
    )
    assert image.getpixel((0, 0)) == (255, 255, 255)


def test_region_and_historical_layer(tmp_path, monkeypatch):
    monkeypatch.setenv("ATLAS_DB", str(tmp_path / "atlas.sqlite3"))
    buf = BytesIO()
    Image.new("RGB", (140, 80), "gray").save(buf, "PNG")
    photo_id = AtlasStore().add_photo(buf.getvalue(), "Corner")
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run()
    app.selectbox[0].set_value(photo_id).run()
    app.checkbox[0].check().run()
    assert not app.exception
    app.text_input[0].set_value("rider")
    app.selectbox[-2].set_value("present")
    app.button[0].click().run()
    assert not app.exception
    assert AtlasStore().region_summaries(photo_id)[0]["present"] == 1
    app.radio(key="navigation").set_value("Community atlas").run()
    app.checkbox[0].check().run()
    assert not app.exception
    assert len(app.dataframe[0].value) > 0
    app.radio(key="navigation").set_value("Research").run()
    assert not app.exception


def test_shared_region_keeps_two_reviewers(tmp_path, monkeypatch):
    monkeypatch.setenv("ATLAS_DB", str(tmp_path / "atlas.sqlite3"))
    db = AtlasStore()
    buf = BytesIO()
    Image.new("RGB", (120, 80), "gray").save(buf, "PNG")
    photo = db.add_photo(buf.getvalue(), "Crossing")
    region = db.add_region(photo, [0.1, 0.2, 0.6, 0.7])
    db.report_region(region, "first", "present")
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"))
    app.query_params.update(photo=str(photo), region=str(region))
    app.run()
    assert not app.exception
    assert app.selectbox[1].value == f"Saved region {region}"
    app.text_input[0].set_value("second")
    app.selectbox[2].set_value("absent")
    app.button[0].click().run()
    assert not app.exception
    summary = db.region_summaries(photo)[0]
    assert (summary["present"], summary["absent"]) == (1, 1)
