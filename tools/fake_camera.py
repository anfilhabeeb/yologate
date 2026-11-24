#!/usr/bin/env python3
"""Synthetic traffic camera for demos and tests.

Serves an MJPEG stream (/video) and a JPEG snapshot (/snapshot.jpg) of
vehicles driving left-to-right across a road. With --sprite the vehicles are
drawn using a real vehicle image (so YOLO can detect them); without it they
are bright rectangles (enough for VC_DETECTOR=dummy).

    python tools/fake_camera.py --port 8081 --sprite tools/assets/bus.jpg
    # then add camera URL http://localhost:8081/video
"""
from __future__ import annotations

import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np


class Scene:
    def __init__(self, width=960, height=540, sprite=None, vehicles=3, speed=12, sprite_w=260):
        self.w, self.h = width, height
        self.speed = speed
        self.sprite = None
        if sprite is not None:
            sh, sw = sprite.shape[:2]
            self.sprite = cv2.resize(sprite, (sprite_w, int(sh * sprite_w / sw)))
        self.vehicles = vehicles
        self.t = 0

    def frame(self, t: int | None = None) -> np.ndarray:
        t = self.t if t is None else t
        img = np.full((self.h, self.w, 3), 60, np.uint8)
        cv2.rectangle(img, (0, int(self.h * 0.25)), (self.w, int(self.h * 0.95)), (90, 90, 90), -1)
        for x in range(0, self.w, 80):
            cv2.line(img, (x, int(self.h * 0.6)), (x + 40, int(self.h * 0.6)), (200, 200, 200), 3)
        span = self.w + 400
        for i in range(self.vehicles):
            x = (t * self.speed + i * span // self.vehicles) % span - 300
            if self.sprite is not None:
                sh, sw = self.sprite.shape[:2]
                y = int(self.h * 0.6) - sh // 2
                x0, x1 = max(0, x), min(self.w, x + sw)
                if x1 > x0:
                    img[y:y + sh, x0:x1] = self.sprite[:, x0 - x:x1 - x]
            else:
                y = int(self.h * (0.35 + 0.2 * (i % 3)))
                cv2.rectangle(img, (x, y), (x + 120, y + 60), (255, 255, 255), -1)
        cv2.putText(img, time.strftime("%H:%M:%S"), (10, self.h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1)
        return img

    def step(self):
        self.t += 1


def make_handler(scene: Scene, lock: threading.Lock, latest: dict, fps: float):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def do_GET(self):
            if self.path.startswith("/snapshot"):
                with lock:
                    jpg = latest["jpg"]
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpg)))
                self.end_headers()
                self.wfile.write(jpg)
                return
            if not self.path.startswith("/video"):
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    with lock:
                        jpg = latest["jpg"]
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode() + jpg + b"\r\n")
                    time.sleep(1.0 / fps)
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler


def load_sprite(path: str | None, crop: str | None):
    if not path:
        return None
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"Cannot read sprite {path}")
    if crop:
        x1, y1, x2, y2 = map(int, crop.split(","))
        img = img[y1:y2, x1:x2]
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--fps", type=float, default=15)
    ap.add_argument("--speed", type=int, default=12, help="pixels per frame")
    ap.add_argument("--vehicles", type=int, default=3)
    ap.add_argument("--sprite", help="image of a vehicle to animate")
    ap.add_argument("--crop", help="x1,y1,x2,y2 crop of the sprite image")
    ap.add_argument("--write", help="write N frames to this video file instead of serving")
    ap.add_argument("--frames", type=int, default=300)
    args = ap.parse_args()

    scene = Scene(sprite=load_sprite(args.sprite, args.crop), vehicles=args.vehicles, speed=args.speed)
    if args.write:
        vw = cv2.VideoWriter(args.write, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (scene.w, scene.h))
        for i in range(args.frames):
            vw.write(scene.frame(i))
        vw.release()
        print(f"wrote {args.frames} frames to {args.write}")
        return

    lock = threading.Lock()
    latest = {"jpg": cv2.imencode(".jpg", scene.frame())[1].tobytes()}

    def produce():
        while True:
            scene.step()
            jpg = cv2.imencode(".jpg", scene.frame())[1].tobytes()
            with lock:
                latest["jpg"] = jpg
            time.sleep(1.0 / args.fps)

    threading.Thread(target=produce, daemon=True).start()
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(scene, lock, latest, args.fps))
    print(f"Fake camera on http://{args.host}:{args.port}/video  (snapshot: /snapshot.jpg)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
