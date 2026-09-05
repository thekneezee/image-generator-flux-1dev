"""FLUX Studio - a chat interface for a FLUX.1-dev Runpod Serverless endpoint.

The point of this UI is that you can *feel* the model working. Serverless
image generation has a long, opaque middle: the request queues, a GPU wakes,
34 GB of weights load, then 28 denoising steps run. A spinner throws all of
that away.

Instead the handler emits a progress update on every denoising step, and this
app polls /status and re-renders on each one. Gradio repaints the chat on
every yield from a generator, so one generation yields many times and the
user watches it move through real phases.

Everything talks to Runpod server-side. Runpod has no browser-safe key and no
per-origin allowlist, so a key in client JavaScript would let anyone drain the
account.
"""

from __future__ import annotations

import os
import random
import tempfile
import time
import uuid
from base64 import b64decode
from pathlib import Path
from typing import Any, Iterator

import gradio as gr
from dotenv import load_dotenv

from runpod_client import MockClient, RunPodClient, RunPodError, cost_for
from theme import CSS, build_theme

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
load_dotenv()  # also accept a .env beside app.py, e.g. on a Space

MOCK = os.environ.get("RUNPOD_MOCK", "0") == "1"
APP_PASSWORD = os.environ.get("APP_PASSWORD", "").strip()

# The live demo runs on the author's credit, so cap what one visitor can spend.
MAX_GENERATIONS_PER_SESSION = int(os.environ.get("MAX_GENERATIONS", "25"))

IMAGE_DIR = Path(tempfile.gettempdir()) / "flux-studio"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# Render and Hugging Face Spaces both inject the port to bind and expect the
# process to listen on 0.0.0.0. Running locally, neither is set.
HOSTED = bool(os.environ.get("RENDER") or os.environ.get("SPACE_ID") or os.environ.get("PORT"))
HOST = "0.0.0.0" if HOSTED else "127.0.0.1"
PORT = int(os.environ.get("PORT", "7860"))

ASPECTS = {
    "Square 1:1": (1024, 1024),
    "Landscape 3:2": (1216, 832),
    "Portrait 2:3": (832, 1216),
    "Wide 16:9": (1344, 768),
}

EXAMPLES = [
    "a red fox curled asleep in deep snow, dawn light through pine branches, photorealistic",
    "an overgrown Art Deco train station reclaimed by jungle, volumetric god rays, cinematic",
    "a hand-drawn cutaway diagram of an imaginary deep-sea submarine, ink and watercolour",
    "portrait of an elderly luthier in his workshop, warm rim light, shallow depth of field",
]


def build_client():
    if MOCK:
        return MockClient()
    return RunPodClient(
        api_key=os.environ.get("RUNPOD_API_KEY", ""),
        endpoint_id=os.environ.get("RUNPOD_ENDPOINT_ID", ""),
    )


CLIENT = build_client()


def new_session() -> dict[str, Any]:
    return {
        "job_id": None,
        "cancelled": False,
        "count": 0,
        "cost": 0.0,
        "gpu_seconds": 0.0,
        "last_seed": None,
        "gallery": [],
    }


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

BAR_WIDTH = 22


def _bar(percent: float) -> str:
    filled = int(round(BAR_WIDTH * max(0.0, min(100.0, percent)) / 100))
    return "█" * filled + "░" * (BAR_WIDTH - filled)


def render_status(
    status: str, progress: dict[str, Any] | None, elapsed: float, cold: bool
) -> str:
    """Turn a poll of job state into the line the user reads."""
    phase = (progress or {}).get("phase")

    if status == "IN_QUEUE":
        # Three tiers, because one status covers wildly different waits. A warm
        # worker dequeues in ~1-2s; a cold start is ~30s; but after a new
        # deployment Runpod may stage ~58 GB of weights onto a fresh host,
        # which takes tens of minutes. Calling all of that "cold start" leaves
        # the user watching an unexplained timer.
        if elapsed < 6:
            icon = "🟡"
            body = ("<b>Queued</b> <span class='muted'>&mdash; asking Runpod "
                    "for a GPU worker</span>")
        elif elapsed < 90:
            icon = "❄️"
            body = (
                "<b>Cold start</b> <span class='muted'>&mdash; a worker is booting "
                "and loading FLUX.1-dev (~34&nbsp;GB) into GPU memory. The next "
                "image will be much faster.</span>"
            )
        else:
            icon = "🏗️"
            body = (
                "<b>Provisioning a worker</b> <span class='muted'>&mdash; Runpod is "
                "placing this job on a host that has the model staged. After a new "
                "deployment that staging can take several minutes. You are not "
                "billed for the wait, and the job stays queued until a worker is "
                "ready.</span>"
            )
        return (
            f"<div class='genstatus'>{icon} {body}<br>"
            f"<span class='muted'>{elapsed:.0f}s elapsed</span></div>"
        )

    if phase == "denoising":
        step = progress.get("step", 0)
        total = progress.get("total_steps", 1)
        percent = progress.get("percent", 0)
        return (
            "<div class='genstatus'>🎨 <b>Denoising</b><br>"
            f"<span class='bar'>{_bar(percent)}</span> "
            f"<b>{step}/{total}</b> <span class='muted'>({percent}%)</span><br>"
            f"<span class='muted'>{elapsed:.0f}s elapsed</span></div>"
        )

    if phase == "decoding":
        return (
            "<div class='genstatus'>🖼️ <b>Decoding latents to pixels</b><br>"
            f"<span class='bar'>{_bar(100)}</span><br>"
            f"<span class='muted'>{elapsed:.0f}s elapsed</span></div>"
        )

    if phase == "encoding_prompt" or status == "IN_PROGRESS":
        return (
            "<div class='genstatus'>✍️ <b>Encoding your prompt</b> "
            "<span class='muted'>(T5-XXL + CLIP)</span><br>"
            f"<span class='muted'>{elapsed:.0f}s elapsed</span></div>"
        )

    return f"<div class='genstatus'>⏳ <b>{status}</b> &mdash; {elapsed:.0f}s</div>"


def render_meta(output: dict[str, Any], delay_ms: int | None, exec_ms: int | None,
                cost: float) -> str:
    params = output.get("parameters", {}) or {}
    gpu = output.get("gpu", "unknown GPU")
    seed = output.get("seed")
    gen_time = output.get("generation_time")
    source = output.get("weights_source", "?")

    bits = [
        f"<b>{params.get('width')}&times;{params.get('height')}</b>",
        f"{params.get('num_inference_steps')} steps",
        f"guidance {params.get('guidance')}",
        f"seed <b>{seed}</b>",
    ]
    timing = [f"<b>{gen_time}s</b> generating"]
    if delay_ms is not None:
        timing.append(f"{delay_ms / 1000:.1f}s queue/cold start")
    if exec_ms is not None:
        timing.append(f"{exec_ms / 1000:.1f}s worker time")

    notes = output.get("notes") or []
    note_html = ""
    if notes:
        note_html = "<br>⚠️ " + " · ".join(notes)

    return (
        f"<div class='meta'>{' · '.join(bits)}<br>"
        f"{' · '.join(timing)}<br>"
        f"on <b>{gpu}</b> · weights from <b>{source}</b> · "
        f"cost <b>${cost:.4f}</b>{note_html}</div>"
    )


def render_telemetry(health: dict[str, Any] | None, error: str | None = None) -> str:
    if error:
        return f"<div class='telemetry'><span class='muted'>{error}</span></div>"
    workers = (health or {}).get("workers", {})
    jobs = (health or {}).get("jobs", {})

    running = workers.get("running", 0)
    initializing = workers.get("initializing", 0)
    if running:
        dot, label = "dot-live", "generating"
    elif initializing:
        dot, label = "dot-init", "staging model"
    elif workers.get("idle") or workers.get("ready"):
        dot, label = "dot-live", "warm"
    else:
        dot, label = "dot-idle", "scaled to zero"

    def row(key: str, value: int, hot: bool = False) -> str:
        cls = "v zero" if not value else ("v hot" if hot else "v warm")
        return f"<div class='row'><span class='k'>{key}</span><span class='{cls}'>{value}</span></div>"

    return (
        "<div class='telemetry'>"
        f"<div style='margin-bottom:0.5rem'><span class='dot {dot}'></span><b>{label}</b></div>"
        + row("workers running", running, hot=True)
        + row("workers idle", workers.get("idle", 0))
        + row("initializing", initializing)
        + row("throttled", workers.get("throttled", 0))
        + row("unhealthy", workers.get("unhealthy", 0))
        + "<div style='height:0.45rem'></div>"
        + row("jobs in queue", jobs.get("inQueue", 0))
        + row("jobs running", jobs.get("inProgress", 0), hot=True)
        + row("completed", jobs.get("completed", 0))
        + row("failed", jobs.get("failed", 0))
        + "</div>"
    )


def render_cost(session: dict[str, Any]) -> str:
    return (
        "<div class='cost'>"
        f"<div class='big'>${session['cost']:.4f}</div>"
        f"<div class='sub'>{session['count']} image(s) · "
        f"{session['gpu_seconds']:.1f}s of GPU time this session</div>"
        "</div>"
    )


def save_image(b64: str, fmt: str) -> Path:
    path = IMAGE_DIR / f"{uuid.uuid4().hex}.{'jpg' if fmt == 'jpeg' else 'png'}"
    path.write_bytes(b64decode(b64))
    return path


# ---------------------------------------------------------------------------
# Core interaction
# ---------------------------------------------------------------------------


def generate(
    prompt: str,
    history: list[dict[str, Any]],
    aspect: str,
    steps: int,
    guidance: float,
    seed_text: str,
    lock_seed: bool,
    negative: str,
    image_format: str,
    session: dict[str, Any],
) -> Iterator[tuple]:
    history = list(history or [])
    session = dict(session or new_session())

    prompt = (prompt or "").strip()
    if not prompt:
        yield history, session, render_cost(session), gr.update(), "", gr.update()
        return

    if session["count"] >= MAX_GENERATIONS_PER_SESSION:
        history.append({"role": "user", "content": prompt})
        history.append(
            {
                "role": "assistant",
                "content": (
                    f"⚠️ This demo allows {MAX_GENERATIONS_PER_SESSION} generations "
                    "per session, to keep the author's Runpod credit from being "
                    "drained. Reload the page to start a new session."
                ),
            }
        )
        yield history, session, render_cost(session), gr.update(), "", gr.update()
        return

    width, height = ASPECTS.get(aspect, (1024, 1024))
    try:
        seed = int(seed_text) if str(seed_text).strip() not in {"", "-1"} else -1
    except ValueError:
        seed = -1

    payload = {
        "prompt": prompt,
        "negative_prompt": (negative or "").strip(),
        "width": width,
        "height": height,
        "num_inference_steps": int(steps),
        "guidance": float(guidance),
        "seed": seed,
        "image_format": image_format,
    }

    history.append({"role": "user", "content": prompt})
    history.append({"role": "assistant", "content": "🟡 Submitting…"})
    yield history, session, render_cost(session), gr.update(), "", gr.update()

    started = time.time()
    session["cancelled"] = False

    try:
        job_id = CLIENT.submit(payload)
        session["job_id"] = job_id

        last_update = None
        for update in CLIENT.poll(job_id):
            last_update = update
            if session.get("cancelled"):
                break
            history[-1]["content"] = render_status(
                update.status, update.progress, time.time() - started, cold=True
            )
            yield history, session, render_cost(session), gr.update(), "", gr.update()

        if session.get("cancelled") or (last_update and last_update.status == "CANCELLED"):
            history[-1]["content"] = "🛑 Cancelled."
            session["job_id"] = None
            yield history, session, render_cost(session), gr.update(), "", gr.update()
            return

        update = last_update
        if update is None or update.status != "COMPLETED":
            history[-1]["content"] = _explain_terminal(update)
            session["job_id"] = None
            yield history, session, render_cost(session), gr.update(), "", gr.update()
            return

        output = update.output or {}
        if "error" in output:
            history[-1]["content"] = (
                f"❌ <b>{output.get('error_type', 'Error')}</b><br>{output['error']}"
            )
            session["job_id"] = None
            yield history, session, render_cost(session), gr.update(), "", gr.update()
            return

        images = output.get("images") or []
        if not images:
            history[-1]["content"] = "❌ The worker returned no image."
            session["job_id"] = None
            yield history, session, render_cost(session), gr.update(), "", gr.update()
            return

        # Bill against worker execution time - the part actually charged.
        exec_seconds = (update.execution_ms or 0) / 1000.0
        cost = cost_for(output.get("gpu"), exec_seconds)
        session["cost"] += cost
        session["gpu_seconds"] += exec_seconds
        session["count"] += 1
        session["last_seed"] = output.get("seed")
        session["job_id"] = None

        paths = [save_image(img["image"], img.get("format", "png")) for img in images]
        session["gallery"] = [str(p) for p in paths] + session["gallery"]

        history[-1] = {"role": "assistant", "content": {"path": str(paths[0])}}
        for extra in paths[1:]:
            history.append({"role": "assistant", "content": {"path": str(extra)}})
        history.append(
            {
                "role": "assistant",
                "content": render_meta(output, update.delay_ms, update.execution_ms, cost),
            }
        )

        # With the seed locked, write the resolved seed back so the next
        # prompt is a controlled comparison rather than a new roll.
        seed_out = gr.update(value=str(output.get("seed"))) if lock_seed else gr.update()
        yield history, session, render_cost(session), session["gallery"], "", seed_out

    except RunPodError as exc:
        history[-1]["content"] = f"❌ {exc}"
        session["job_id"] = None
        yield history, session, render_cost(session), gr.update(), "", gr.update()
    except Exception as exc:  # noqa: BLE001
        history[-1]["content"] = f"❌ Unexpected error: {type(exc).__name__}: {exc}"
        session["job_id"] = None
        yield history, session, render_cost(session), gr.update(), "", gr.update()


def _explain_terminal(update) -> str:
    if update is None:
        return "❌ Lost track of the job."
    if update.status == "FAILED":
        detail = ""
        if isinstance(update.output, dict):
            detail = f"<br>{update.output.get('error', '')}"
        return f"❌ <b>Job failed on the worker.</b>{detail}"
    if update.status == "TIMED_OUT":
        return (
            "⏱️ <b>Timed out.</b> The job exceeded the endpoint's execution timeout. "
            "Try fewer steps or a smaller image."
        )
    return f"❌ Job ended as <b>{update.status}</b>."


def cancel_job(session: dict[str, Any]) -> dict[str, Any]:
    session = dict(session or new_session())
    job_id = session.get("job_id")
    session["cancelled"] = True
    if job_id:
        try:
            CLIENT.cancel(job_id)
        except RunPodError:
            pass
    return session


def wake_gpu(session: dict[str, Any]) -> Iterator[str]:
    """Pay the cold start up front, so the first real prompt feels instant."""
    yield "<div class='genstatus'>🔌 Waking a worker…</div>"
    try:
        job_id = CLIENT.submit({"warmup": True, "prompt": "warmup"})
        started = time.time()
        for update in CLIENT.poll(job_id, timeout=1800):
            yield (
                "<div class='genstatus'>🔌 <b>Waking a worker</b><br>"
                f"<span class='muted'>{update.status} · "
                f"{time.time() - started:.0f}s</span></div>"
            )
        yield (
            "<div class='genstatus'>✅ <b>Worker warm.</b> "
            "<span class='muted'>Next image skips the cold start.</span></div>"
        )
    except RunPodError as exc:
        yield f"<div class='genstatus'>❌ {exc}</div>"


def refresh_telemetry() -> str:
    try:
        return render_telemetry(CLIENT.health())
    except RunPodError as exc:
        return render_telemetry(None, error=str(exc))
    except Exception:  # noqa: BLE001
        return render_telemetry(None, error="telemetry unavailable")


def randomise_seed() -> str:
    return str(random.randint(0, 2**32 - 1))


def reuse_last_seed(session: dict[str, Any]) -> str:
    seed = (session or {}).get("last_seed")
    return str(seed) if seed is not None else "-1"


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def build_ui() -> gr.Blocks:
    badge = (
        "<span class='pill pill-mock'>MOCK MODE</span>"
        if MOCK
        else "<span class='pill pill-live'>LIVE ON RUNPOD</span>"
    )
    endpoint_label = "no endpoint" if MOCK else CLIENT.endpoint_id

    with gr.Blocks(theme=build_theme(), css=CSS, title="FLUX Studio") as demo:
        session = gr.State(new_session())

        gr.HTML(
            "<div id='app-header'>"
            "<div><div class='title'>FLUX Studio</div>"
            "<div class='subtitle'>FLUX.1-dev &middot; 12B rectified-flow transformer "
            f"&middot; Runpod Serverless &middot; <code>{endpoint_label}</code></div></div>"
            f"<div>{badge}</div></div>"
        )

        if MOCK:
            gr.HTML(
                "<div class='mock-banner'>⚠️ <b>MOCK MODE</b> &mdash; no GPU is being "
                "used and no image model is running. Responses are simulated and the "
                "pictures are placeholder patterns, <b>not</b> generations of your "
                "prompt. This mode exists so the interface can be explored without "
                "Runpod credentials or cost. Set <code>RUNPOD_MOCK=0</code> for real "
                "FLUX.1-dev output.</div>"
            )

        with gr.Row():
            # ---- conversation -------------------------------------------
            with gr.Column(scale=3):
                chat = gr.Chatbot(
                    type="messages",
                    height=560,
                    show_label=False,
                    show_copy_button=True,
                    avatar_images=(None, None),
                    placeholder=(
                        "<div style='text-align:center;opacity:0.65'>"
                        "<h2>Describe an image</h2>"
                        "<p>The first request wakes a GPU and loads 34 GB of weights, "
                        "so it takes a minute. After that, a few seconds each.</p></div>"
                    ),
                )

                with gr.Row():
                    prompt_box = gr.Textbox(
                        placeholder="a red fox curled asleep in deep snow, dawn light…",
                        show_label=False,
                        scale=8,
                        lines=2,
                        max_lines=6,
                        autofocus=True,
                    )
                    with gr.Column(scale=1, min_width=120):
                        go = gr.Button("Generate", variant="primary")
                        stop = gr.Button("Cancel", variant="stop", size="sm")

                gr.Examples(examples=EXAMPLES, inputs=prompt_box, label="Try one")

                with gr.Accordion("Session gallery", open=False):
                    gallery = gr.Gallery(
                        show_label=False,
                        columns=4,
                        height=240,
                        object_fit="cover",
                        preview=True,
                    )

            # ---- instrument panel ---------------------------------------
            with gr.Column(scale=1, min_width=290):
                gr.Markdown("### Live endpoint")
                telemetry = gr.HTML(render_telemetry(None, "connecting…"))
                warm_btn = gr.Button("🔌 Pre-warm the GPU", size="sm")
                gr.HTML(
                    "<div class='hint'>Workers shut down after 5s idle, so the next "
                    "prompt pays a ~30s cold start while 34&nbsp;GB of weights load. "
                    "This pays that cost now, so your next image returns in seconds.</div>"
                )
                warm_status = gr.HTML("")

                gr.Markdown("### Session cost")
                cost_panel = gr.HTML(render_cost(new_session()))

                with gr.Accordion("Generation settings", open=False):
                    aspect = gr.Radio(
                        list(ASPECTS), value="Square 1:1", label="Aspect ratio"
                    )
                    steps = gr.Slider(
                        4, 50, value=28, step=1, label="Inference steps",
                        info="More steps, more detail, more cost. 28 is the FLUX default.",
                    )
                    guidance = gr.Slider(
                        0.0, 10.0, value=3.5, step=0.1, label="Guidance",
                        info="How literally to follow the prompt. FLUX likes 3.5.",
                    )
                    with gr.Row():
                        seed_box = gr.Textbox(
                            value="-1", label="Seed", scale=3,
                            info="The number that picks the starting noise. Same seed "
                                 "+ same prompt = the exact same image, every time. "
                                 "-1 rolls a new one each generation.",
                        )
                        dice = gr.Button("🎲", scale=1, min_width=44)
                    lock_seed = gr.Checkbox(
                        value=False,
                        label="Lock the seed after generating",
                        info="Keeps the next image on the same noise, so changing a "
                             "word in the prompt shows only that word's effect.",
                    )
                    reuse = gr.Button("Reuse last seed", size="sm")
                    negative = gr.Textbox(
                        label="Negative prompt", lines=2, value="",
                        placeholder="blurry, low quality, watermark",
                        info="FLUX.1-dev is guidance-distilled, so this switches on "
                             "true CFG and roughly doubles generation time.",
                    )
                    image_format = gr.Radio(
                        ["png", "jpeg"], value="png", label="Format"
                    )

        # ---- wiring -----------------------------------------------------
        inputs = [prompt_box, chat, aspect, steps, guidance, seed_box,
                  lock_seed, negative, image_format, session]
        outputs = [chat, session, cost_panel, gallery, prompt_box, seed_box]

        go.click(generate, inputs, outputs)
        prompt_box.submit(generate, inputs, outputs)
        stop.click(cancel_job, [session], [session])
        dice.click(randomise_seed, None, [seed_box])
        reuse.click(reuse_last_seed, [session], [seed_box])
        warm_btn.click(wake_gpu, [session], [warm_status])

        # Poll /health so the panel shows workers scaling 0 -> 1 -> 0 live.
        gr.Timer(3.0).tick(refresh_telemetry, None, [telemetry])
        demo.load(refresh_telemetry, None, [telemetry])

    return demo


if __name__ == "__main__":
    ui = build_ui()
    auth = ("reviewer", APP_PASSWORD) if APP_PASSWORD else None
    if auth:
        print(f"Password gate enabled (user 'reviewer').")
    else:
        print("No APP_PASSWORD set - running without a password gate.")
    ui.launch(
        auth=auth,
        allowed_paths=[str(IMAGE_DIR)],
        # A hosting platform (Render, a HF Space) must reach the app on all
        # interfaces and on the port it assigns. Locally, bind loopback only so
        # the dev server is not exposed to the network.
        server_name=os.environ.get("GRADIO_SERVER_NAME", HOST),
        server_port=PORT,
        show_api=False,
    )
