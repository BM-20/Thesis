"""Tests for the pneumonia detection app.

Run with:  pytest

The app imports heavy native deps (torch, opencv, pydicom) at module load, so the
suite skips cleanly if they're absent. No trained ``model.pth`` is required: the
model is loaded lazily and none of these tests trigger a real inference.
"""
import io

import pytest

# Skip the whole suite unless the app's import-time dependencies are available.
pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("cv2")
pytest.importorskip("pydicom")
pytest.importorskip("PIL")

from PIL import Image
from werkzeug.utils import secure_filename

from pneumonia_api import app, allowed_file, transform


@pytest.fixture
def client():
    app.config.update(TESTING=True)
    with app.test_client() as test_client:
        yield test_client


def test_allowed_file_accepts_supported_extensions():
    assert allowed_file("scan.png")
    assert allowed_file("scan.JPG")      # case-insensitive
    assert allowed_file("scan.jpeg")
    assert allowed_file("scan.dcm")


def test_allowed_file_rejects_unsupported():
    assert not allowed_file("malware.exe")
    assert not allowed_file("archive.zip")
    assert not allowed_file("no_extension")


def test_transform_outputs_model_input_shape():
    img = Image.new("RGB", (512, 400), color=(120, 120, 120))
    tensor = transform(img)
    assert tuple(tensor.shape) == (3, 224, 224)
    assert tensor.is_floating_point()


def test_index_page_loads(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Pneumonia" in resp.data


def test_predict_without_file_is_handled(client):
    resp = client.post("/predict")
    assert resp.status_code == 200
    assert b"No file uploaded" in resp.data


def test_predict_rejects_non_image(client):
    data = {"file": (io.BytesIO(b"not an image"), "notes.txt")}
    resp = client.post("/predict", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    assert b"Invalid file type" in resp.data


def test_secure_filename_blocks_path_traversal():
    cleaned = secure_filename("../../etc/passwd")
    assert ".." not in cleaned and "/" not in cleaned
