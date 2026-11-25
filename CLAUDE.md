# CLAUDE.md: project instructions for YOLOgate

> Claude Code loads this file automatically. It doubles as a **base prompt for new
> projects**: copy it to a new repo's root and rewrite section 2 for that project.

---

## 1. About me and my setup
- I build practical **AI, computer-vision, IoT and embedded** projects (Python, ESP32/Arduino, Raspberry Pi, web dashboards).
- **My PC:** Windows 10 Home, Intel i5-10210U (4 cores / 8 threads), 16 GB RAM, Intel UHD graphics (no NVIDIA GPU), **limited free disk space (about 5 GB)**.
- I also use **GitHub Codespaces** to try things out. Its network to pypi.org can be unreliable.
- I follow your instructions step by step, so give me **exact, copyable commands**: one command per code block, and say whether it's PowerShell (Windows) or bash (Linux/Codespaces).

## 2. Project brief (this project; rewrite for new projects)
- **Name:** YOLOgate
- **Goal:** count vehicles from live camera streams using YOLO and lines drawn on each camera view.
- **Inputs:** HTTP MJPEG streams, JPEG snapshot URLs, RTSP/HLS streams and video files.
- **Outputs:** web portal (dashboard, live annotated view, line editor), REST API, SQLite database of crossing events.
- **Must run on:** my Windows PC, a Linux server (Docker or systemd) and GitHub Codespaces.
- **Code map:**
  - `app/agent.py`: camera workers and supervisor
  - `app/detector.py`: YOLO through Ultralytics, ONNX through OpenCV, or a dummy detector
  - `app/tracker.py`: vehicle tracking
  - `app/geometry.py`: line-crossing maths
  - `app/main.py`: FastAPI routes
  - `app/static/`: the portal pages
  - `tools/fake_camera.py`: the demo camera
- **Run tests:** `pytest -v` (needs `pip install -r requirements-lite.txt pytest httpx`).
- **Run the demo:** `bash tools/demo.sh`, then open http://localhost:8000.
- **Out of scope for now:** authentication, GPU images, ARM Docker builds.

## 3. Engineering rules
1. **Lightweight first.** Prefer small dependencies that work on CPU (e.g. ONNX + OpenCV rather than PyTorch). Offer the heavy or GPU version as an option, not the default.
2. **Work offline where possible.** Bundle models and sample assets in the repo, or download them once during setup, never on every start.
3. **Expect a flaky network.** Install with `--timeout 120 --retries 10`, give a fallback (for example Debian `apt` packages when pypi.org fails), and never make startup depend on downloads.
4. **Stay modular and testable.** Keep pure logic (geometry, parsing, maths) separate from I/O (streams, hardware, HTTP). Detectors, sources and drivers go behind small interfaces, so tests can swap in fakes.
5. **Long-running services must be robust:** auto-reconnect, supervise and restart workers, keep only the latest frame or reading (no growing backlog), and persist important data (SQLite by default).
6. **Configure through environment variables** with sensible defaults, documented in a table in the README.
7. **Handle security honestly.** Never commit secrets. If there's no login or authentication, say so in the README and in your summary.
8. **Check licences** of models, datasets and libraries (for example, Ultralytics YOLO is AGPL-3.0) and tell me about anything that affects public or commercial use.

## 4. Always include
- **A demo or simulator** (for example a fake camera or sensor) so the app can be shown without real hardware.
- **Tests:**
  - unit tests for the core logic
  - integration tests against the real protocol (for example a real HTTP/MJPEG server)
  - at least one end-to-end test with the real model or engine
  - tests that skip themselves when heavy optional dependencies are missing
- **Startup options:**
  - `tools/demo.sh` (one command: start the app with demo data)
  - `start_windows.bat` (Windows, no Docker)
  - `Dockerfile`, lite by default, with an optional `FULL=1` build argument
  - `docker-compose.yml`
  - a systemd unit and install script for Linux
- **Codespaces:** a `.devcontainer/` setup that installs dependencies (with the apt fallback), auto-starts the demo, forwards and labels the port, and opens the browser.
- **GitHub Actions CI:**
  - tests on the full and lite installs
  - Docker build, then a smoke test (start the container and check `/api/health`)
  - publish to GHCR using a **fixed image name**, `ghcr.io/<owner>/<project>`, so it doesn't depend on the repository name
- **README:**
  - what it does, with screenshots
  - quick start for Windows, Linux, Docker and Codespaces
  - configuration table and API table
  - troubleshooting table
  - notes on security and licences

## 5. How to work
1. **Look first, then plan:** inspect the repo and environment, then give me a short plan.
2. **Build in small steps,** running the tests after each meaningful change.
3. **Verify for real, not just with tests:**
   - run the app with the demo
   - check it in a headless browser (Playwright) and look at the screenshots
   - check that the numbers match what's expected (for example "14 vehicles in 45 s matches the simulator rate")
4. **When something fails,** find the root cause, fix it, and add a regression test.
5. **Git:**
   - work on a feature branch, with clear commit messages explaining what changed and why
   - push when a stage is done
   - check CI and fix it if it's red
   - merge to `main` only when I ask
6. **Be honest in reports:** state what was tested and passed, what couldn't be tested (and why), and what I need to do myself (for example change a GitHub setting).

## 6. How to report back
- Start with the result in one or two sentences.
- Then **step-by-step instructions** with copyable commands and the output I should expect.
- Add a short **"If something goes wrong"** table: problem → fix.
- Explain error messages I paste in plain language: what it means, whether to worry, and the exact next command.
- Keep it concise, use tables for comparisons, and don't use jargon without a one-line explanation.

## 7. Definition of done
- [ ] All tests pass locally and in CI
- [ ] The app runs from a clean environment using only the README instructions
- [ ] Demo mode works with no real hardware
- [ ] The Docker image builds, passes the smoke test and is published
- [ ] Codespaces works with one click
- [ ] README is complete, with screenshots, configuration, API and troubleshooting
- [ ] Code is pushed, and I'm told what's left for me to do
