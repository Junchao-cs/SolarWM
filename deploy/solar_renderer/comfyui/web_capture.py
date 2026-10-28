"""Record a web page's 3D canvas (three.js or any WebGL content) as a structure video, with its sound.

    python deploy/solar_renderer/comfyui/web_capture.py --url https://example.com/game.html --out capture --frames 158

Runs in its own process (the node starts it), so a browser that hangs or crashes never takes ComfyUI with it. The page
is loaded in headless Chromium, its start button pressed (--click), an optional script run (--setup), and then every
1/24 s the largest canvas on the page is grabbed (it must be drawn with preserveDrawingBuffer, as most engines can be
asked to; the capture asks for it on every WebGL context). An optional key and mouse script plays it:

    seconds:Key[:hold]      a key pressed for hold seconds (KeyW, ArrowUp, Space, ShiftLeft, ...)
    seconds:down:Key        a key held from here ...
    seconds:up:Key          ... to here
    seconds:mouse:dx:dy[:over]   the mouse moved by (dx, dy) pixels over `over` seconds
    seconds:click[:hold]    the left mouse button pressed

--exact runs the page on a virtual clock that moves 1/24 s per recorded frame, so a machine that cannot grab 24 frames
a second still records the page at exactly 24 fps (the sound, which runs on the real clock, is then left out).

The sound the page makes through WebAudio is recorded from the first frame.

Writes <out>.npy (frames, uint8 [T, H, W, 3]) and, when the page made sound, <out>.webm.
"""  # noqa: E501

import argparse
import base64
import io
import json
import sys
import time

import numpy as np

FPS = 24

AUDIO_TAP = """(() => {
    // every node connected to a context's destination also feeds a recorder, started at frame 0
    const taps = new WeakMap();
    window.__tapStreams = [];
    const connect = AudioNode.prototype.connect;
    AudioNode.prototype.connect = function (target, ...rest) {
        if (target instanceof AudioDestinationNode) {
            const ctx = target.context;
            if (!taps.has(ctx)) { const t = ctx.createMediaStreamDestination(); taps.set(ctx, t); window.__tapStreams.push(t.stream); }
            connect.call(this, taps.get(ctx), ...rest);
        }
        return connect.call(this, target, ...rest);
    };
    window.__startAudio = () => {
        const s = window.__tapStreams[0];
        if (!s) return false;
        window.__chunks = [];
        window.__rec = new MediaRecorder(s, { mimeType: 'audio/webm;codecs=opus', audioBitsPerSecond: 160000 });
        window.__rec.ondataavailable = e => window.__chunks.push(e.data);
        window.__rec.start(1000);
        return true;
    };
    window.__stopAudio = () => new Promise(resolve => {
        if (!window.__rec) return resolve('');
        window.__rec.onstop = async () => {
            const buf = await new Blob(window.__chunks, { type: 'audio/webm' }).arrayBuffer();
            let bin = ''; const u = new Uint8Array(buf);
            for (let i = 0; i < u.length; i += 0x8000) bin += String.fromCharCode.apply(null, u.subarray(i, i + 0x8000));
            resolve(btoa(bin));
        };
        window.__rec.stop();
    });
})()"""  # noqa: E501

# frame-exact capture: the page's clocks (performance.now, Date.now, requestAnimationFrame) run on a
# virtual time that
# advances 1/24 s per recorded frame, however long a frame takes to grab
VIRTUAL_TIME = """(() => {
    let now = performance.now();
    const base = Date.now() - now;
    const realRaf = window.requestAnimationFrame.bind(window);
    let queued = [], id = 0;
    performance.now = () => now;
    Date.now = () => Math.round(base + now);
    window.requestAnimationFrame = cb => { queued.push([++id, cb]); return id; };
    window.cancelAnimationFrame = handle => { queued = queued.filter(([i]) => i !== handle); };
    window.__advance = ms => {
        now += ms;
        const run = queued; queued = [];
        for (const [, cb] of run) { try { cb(now); } catch (e) { console.error(e); } }
        return new Promise(resolve => realRaf(() => resolve(true)));  // let the frame reach the canvas
    };
})()"""  # noqa: E501

# every WebGL context keeps its last frame, so the canvas can be read back after the page drew it
# (most pages do not ask
# for this, and their canvas reads back black)
PRESERVE = """(() => {
    const getContext = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = function (type, attributes) {
        if (/^(webgl2?|experimental-webgl)$/.test(type)) attributes = { ...(attributes || {}), preserveDrawingBuffer: true };
        return getContext.call(this, type, attributes);
    };
})()"""  # noqa: E501

GRAB = """() => { const cs = [...document.querySelectorAll('canvas')];
    if (!cs.length) return null;
    const c = cs.sort((a, b) => b.width * b.height - a.width * a.height)[0];
    return c.toDataURL('image/jpeg', 0.92).split(',')[1]; }"""


def plan_of(script: str) -> list:
    """The key and mouse script as (seconds, action, argument), in time order."""
    plan = []
    for item in [x.strip() for x in script.replace("\n", ",").split(",") if x.strip()]:
        parts = item.split(":")
        t, kind = float(parts[0]), parts[1]
        if kind in ("down", "up"):
            plan.append((t, kind, parts[2]))
        elif kind == "mouse":
            dx, dy = float(parts[2]), float(parts[3])
            over = float(parts[4]) if len(parts) > 4 else 0.4
            steps = max(1, int(over * 30))
            plan += [(t + over * i / steps, "move", (dx / steps, dy / steps)) for i in range(steps)]
        elif kind == "click":
            hold = float(parts[2]) if len(parts) > 2 else 0.15
            plan += [(t, "mdown", None), (t + hold, "mup", None)]
        else:
            hold = float(parts[2]) if len(parts) > 2 else 0.12
            plan += [(t, "down", kind), (t + hold, "up", kind)]
    return sorted(plan, key=lambda e: e[0])


def default_browser_args() -> list:
    # headless Chromium draws WebGL on the GPU through ANGLE; on Linux servers EGL is the path that
    # has a GPU
    args = ["--ignore-gpu-blocklist", "--autoplay-policy=no-user-gesture-required"]
    if sys.platform.startswith("linux"):
        args += ["--use-gl=angle", "--use-angle=gl-egl"]
    return args


def capture(
    url,
    out,
    frames,
    width,
    height,
    keys="",
    click="",
    setup="",
    warm=1.0,
    browser_args=None,
    exact=False,
):
    import queue
    import threading

    from PIL import Image
    from playwright.sync_api import sync_playwright

    plan = plan_of(keys)
    shots = np.zeros((frames, height, width, 3), dtype=np.uint8)
    work: queue.Queue = queue.Queue()

    def writer() -> None:  # decoding off the grab loop, which has to keep the page's 24 fps clock
        while (item := work.get()) is not None:
            n, data = item
            image = Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")
            if image.size != (width, height):
                image = image.resize((width, height), Image.Resampling.LANCZOS)
            shots[n] = np.asarray(image)

    pen = threading.Thread(target=writer, daemon=True)
    pen.start()
    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=browser_args or default_browser_args())
        page = browser.new_page(viewport={"width": width, "height": height})
        page.add_init_script(PRESERVE)
        page.add_init_script(AUDIO_TAP)
        if exact:
            page.add_init_script(VIRTUAL_TIME)
        page.goto(url, wait_until="load")
        page.wait_for_timeout(1500)
        if click:
            page.click(click)
        if setup:
            page.evaluate(f"() => {{ {setup} }}")
        if exact:  # the warm-up on the virtual clock too
            for _ in range(int(warm * FPS)):
                page.evaluate(f"() => window.__advance({1000 / FPS})")
        else:
            page.wait_for_timeout(int(warm * 1000))
        mouse = [width / 2, height / 2]
        page.mouse.move(*mouse)
        audio = False
        t0 = time.time()
        for n in range(frames):
            if exact:
                clock = n / FPS
                if n:
                    page.evaluate(f"() => window.__advance({1000 / FPS})")
            else:
                due = t0 + n / FPS
                while time.time() < due - 0.003:
                    page.wait_for_timeout(2)
                clock = time.time() - t0
            while plan and clock >= plan[0][0]:
                _t, kind, arg = plan.pop(0)
                if kind == "down":
                    page.keyboard.down(arg)
                elif kind == "up":
                    page.keyboard.up(arg)
                elif kind == "move":
                    mouse = [mouse[0] + arg[0], mouse[1] + arg[1]]
                    page.mouse.move(*mouse)
                elif kind == "mdown":
                    page.mouse.down()
                elif kind == "mup":
                    page.mouse.up()
            data = page.evaluate(GRAB)
            if data is None:
                raise RuntimeError(f"{url} has no canvas to record")
            work.put((n, data))
            if (
                n == 0 and not exact
            ):  # the sound runs on the real clock: only a real-time capture keeps it
                audio = bool(page.evaluate("() => window.__startAudio && window.__startAudio()"))
        took = time.time() - t0
        sound = (
            page.evaluate("() => window.__stopAudio ? window.__stopAudio() : ''") if audio else ""
        )
        browser.close()
    work.put(None)
    pen.join()
    np.save(f"{out}.npy", shots)
    if sound:
        with open(f"{out}.webm", "wb") as f:
            f.write(base64.b64decode(sound))
    # a grab slower than the clock leaves the page's time running ahead of the frames: the clip
    # plays fast
    return {
        "frames": frames,
        "seconds": round(took, 2),
        "real_time": round(frames / FPS / took, 2),
        "sound": bool(sound),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--url", required=True)
    p.add_argument(
        "--out", required=True, help="path prefix of the .npy frames and the .webm sound"
    )
    p.add_argument("--frames", type=int, default=158)
    p.add_argument("--width", type=int, default=1344)
    p.add_argument("--height", type=int, default=768)
    p.add_argument("--keys", default="")
    p.add_argument("--click", default="", help="the selector of a start button to press first")
    p.add_argument("--setup", default="", help="JavaScript run once the page has started")
    p.add_argument(
        "--warm", type=float, default=1.0, help="seconds the page runs before the first frame"
    )
    p.add_argument(
        "--exact",
        action="store_true",
        help="frame-exact: the page runs on a virtual clock, 1/24 s per "
        "frame however slow the grab; no sound",
    )
    p.add_argument(
        "--browser-args",
        default="",
        help="Chromium flags, space-separated (default: GPU through ANGLE)",
    )
    a = p.parse_args()
    report = capture(
        a.url,
        a.out,
        a.frames,
        a.width,
        a.height,
        a.keys,
        a.click,
        a.setup,
        a.warm,
        a.browser_args.split() or None,
        a.exact,
    )
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
