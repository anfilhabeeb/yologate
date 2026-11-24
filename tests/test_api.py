import threading
import time
from http.server import ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.detector import DummyDetector
from app.main import create_app
from fake_camera import Scene, make_handler

import cv2


@pytest.fixture()
def fake_camera():
    """Real HTTP MJPEG server with white 'vehicles' moving left -> right."""
    scene = Scene(width=480, height=270, vehicles=2, speed=10)
    lock = threading.Lock()
    latest = {"jpg": cv2.imencode(".jpg", scene.frame())[1].tobytes()}
    stop = threading.Event()

    def produce():
        while not stop.is_set():
            scene.step()
            jpg = cv2.imencode(".jpg", scene.frame())[1].tobytes()
            with lock:
                latest["jpg"] = jpg
            time.sleep(1 / 30)

    threading.Thread(target=produce, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(scene, lock, latest, 30))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    stop.set()
    srv.shutdown()


@pytest.fixture()
def client(tmp_path):
    settings = Settings(data_dir=tmp_path, detector="dummy", max_fps=30, reconnect_delay=0.5)
    app = create_app(settings, detector=DummyDetector())
    with TestClient(app) as c:
        yield c


def wait_for(pred, timeout=20.0, interval=0.2):
    end = time.time() + timeout
    while time.time() < end:
        v = pred()
        if v:
            return v
        time.sleep(interval)
    raise AssertionError("condition not met in time")


def test_health_and_pages(client):
    assert client.get("/api/health").json()["status"] == "ok"
    assert "Vehicle Counter" in client.get("/").text
    assert client.get("/camera/1").status_code == 200
    assert client.get("/static/common.js").status_code == 200


def test_camera_crud_and_validation(client):
    r = client.post("/api/cameras", json={"name": "Gate", "url": "http://127.0.0.1:9/none", "enabled": False})
    assert r.status_code == 201
    cam = r.json()
    assert cam["status"]["state"] == "stopped"
    assert client.get("/api/cameras").json()[0]["name"] == "Gate"
    assert client.patch(f"/api/cameras/{cam['id']}", json={"name": "Gate 2"}).json()["name"] == "Gate 2"
    assert client.post("/api/cameras", json={"name": "", "url": "x"}).status_code == 422
    bad = [{"x1": 0.1, "y1": 0.1, "x2": 0.1, "y2": 0.1}]
    assert client.put(f"/api/cameras/{cam['id']}/lines", json=bad).status_code == 422
    out = [{"x1": 0.1, "y1": 0.1, "x2": 1.5, "y2": 0.1}]
    assert client.put(f"/api/cameras/{cam['id']}/lines", json=out).status_code == 422
    assert client.delete(f"/api/cameras/{cam['id']}").status_code == 204
    assert client.get(f"/api/cameras/{cam['id']}").status_code == 404


def test_unreachable_camera_reports_error(client):
    cam = client.post("/api/cameras", json={"name": "Dead", "url": "http://127.0.0.1:9/video"}).json()
    st = wait_for(lambda: (s := client.get(f"/api/cameras/{cam['id']}/stats").json())["state"] == "error" and s)
    assert st["last_error"]


@pytest.mark.parametrize("path,stream_type", [("/video", "auto"), ("/snapshot.jpg", "snapshot")])
def test_live_http_stream_counts_vehicles(client, fake_camera, path, stream_type):
    cam = client.post("/api/cameras", json={"name": "Fake", "url": fake_camera + path,
                                            "stream_type": stream_type}).json()
    cid = cam["id"]
    wait_for(lambda: client.get(f"/api/cameras/{cid}/stats").json()["state"] == "running")
    # vertical line in the middle of the frame; vehicles move left -> right ("out")
    lines = [{"name": "Middle", "x1": 0.5, "y1": 0.05, "x2": 0.5, "y2": 0.95}]
    cam = client.put(f"/api/cameras/{cid}/lines", json=lines).json()
    line_id = cam["lines"][0]["id"]
    st = wait_for(lambda: (s := client.get(f"/api/cameras/{cid}/stats").json())["total"] >= 2 and s, timeout=40)
    assert sum(st["counts"][line_id]["out"].values()) >= 2
    assert sum(st["counts"][line_id]["in"].values()) == 0

    events = client.get(f"/api/cameras/{cid}/events").json()
    assert events and events[0]["direction"] == "out" and events[0]["label"] == "car"

    snap = client.get(f"/api/cameras/{cid}/snapshot")
    assert snap.status_code == 200 and snap.content[:2] == b"\xff\xd8"

    # counts persist in the DB and survive a worker restart
    client.patch(f"/api/cameras/{cid}", json={"enabled": False})
    assert client.get(f"/api/cameras/{cid}/stats").json()["total"] >= 2

    assert client.post(f"/api/cameras/{cid}/reset").json()["total"] == 0
    assert client.get(f"/api/cameras/{cid}/events").json() == []
