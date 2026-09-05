"""Visual identity for the chat UI.

Stock Gradio is instantly recognisable, which reads as a prototype. A small
amount of theming makes it read as a product instead - which matters here,
because the interface is the deliverable.
"""

from __future__ import annotations

import gradio as gr

# Runpod's brand purple, used sparingly as the single accent colour.
ACCENT = gr.themes.Color(
    c50="#f2f0ff", c100="#e6e2ff", c200="#cdc5ff", c300="#b0a4ff",
    c400="#9289fe", c500="#7c6cfd", c600="#5f4cfe", c700="#4a37e0",
    c800="#3a2bb0", c900="#2b2080", c950="#1a1450",
)

NEUTRAL = gr.themes.Color(
    c50="#f7f7f8", c100="#ececee", c200="#d9d9de", c300="#b8b8c0",
    c400="#8c8c99", c500="#6b6b78", c600="#4e4e5a", c700="#3a3a45",
    c800="#26262e", c900="#18181d", c950="#0f0f13",
)


def build_theme() -> gr.themes.Base:
    return gr.themes.Soft(
        primary_hue=ACCENT,
        neutral_hue=NEUTRAL,
        # Must be Font objects, not bare strings: Gradio compares themes by
        # font identity and a plain str has no .name.
        font=(gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"),
        font_mono=(gr.themes.GoogleFont("JetBrains Mono"), "Consolas", "monospace"),
    ).set(
        body_background_fill="*neutral_950",
        body_background_fill_dark="*neutral_950",
        block_background_fill="*neutral_900",
        block_background_fill_dark="*neutral_900",
        block_border_width="1px",
        block_border_color="*neutral_800",
        block_label_background_fill="*neutral_900",
        block_radius="12px",
        button_primary_background_fill="*primary_600",
        button_primary_background_fill_hover="*primary_500",
        button_primary_text_color="#ffffff",
        input_background_fill="*neutral_950",
        input_border_color="*neutral_800",
    )


CSS = """
:root { color-scheme: dark; }

#app-header {
  display: flex; align-items: center; justify-content: space-between;
  gap: 1rem; padding: 0.35rem 0.15rem 0.9rem;
  border-bottom: 1px solid var(--neutral-800); margin-bottom: 0.9rem;
}
#app-header .title { font-size: 1.22rem; font-weight: 650; letter-spacing: -0.015em; }
#app-header .subtitle { font-size: 0.8rem; color: var(--neutral-400); margin-top: 0.12rem; }

.pill {
  display: inline-flex; align-items: center; gap: 0.35rem;
  padding: 0.2rem 0.6rem; border-radius: 999px;
  font-size: 0.72rem; font-weight: 600; letter-spacing: 0.02em;
  font-family: var(--font-mono); white-space: nowrap;
}
.pill-live  { background: rgba(34,197,94,0.13);  color: #4ade80; border: 1px solid rgba(34,197,94,0.3); }
.pill-mock  { background: rgba(250,204,21,0.12); color: #facc15; border: 1px solid rgba(250,204,21,0.3); }

/* Live worker telemetry ------------------------------------------------- */
.telemetry { font-family: var(--font-mono); font-size: 0.76rem; line-height: 1.9; }
.telemetry .row { display: flex; justify-content: space-between; gap: 0.75rem; }
.telemetry .k { color: var(--neutral-400); }
.telemetry .v { font-weight: 600; }
.telemetry .v.zero { color: var(--neutral-500); font-weight: 400; }
.telemetry .v.hot  { color: #4ade80; }
.telemetry .v.warm { color: #facc15; }

.dot { display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:0.4rem; }
.dot-idle { background: var(--neutral-600); }
.dot-live { background: #4ade80; box-shadow: 0 0 7px #4ade80; }
.dot-init { background: #facc15; animation: pulse 1.4s ease-in-out infinite; }
@keyframes pulse { 0%,100% { opacity: 1 } 50% { opacity: 0.35 } }

/* Cost meter ------------------------------------------------------------ */
.cost { font-family: var(--font-mono); }
.cost .big { font-size: 1.5rem; font-weight: 700; letter-spacing: -0.02em; }
.cost .sub { font-size: 0.72rem; color: var(--neutral-400); }

/* Generation status inside the chat ------------------------------------- */
.genstatus { font-family: var(--font-mono); font-size: 0.82rem; line-height: 1.75; }
.genstatus .bar { letter-spacing: 0.06em; color: var(--primary-400); }
.genstatus .muted { color: var(--neutral-400); }

/* Image metadata caption ------------------------------------------------ */
.meta { font-family: var(--font-mono); font-size: 0.72rem; color: var(--neutral-400); }
.meta b { color: var(--neutral-200); font-weight: 600; }

.mock-banner {
  background: rgba(250,204,21,0.09); border: 1px solid rgba(250,204,21,0.42);
  border-radius: 10px; padding: 0.7rem 0.9rem; margin-bottom: 0.9rem;
  font-size: 0.82rem; line-height: 1.5; color: #fde68a;
}
.mock-banner code { background: rgba(0,0,0,0.35); padding: 0.05rem 0.3rem; border-radius: 4px; }

.hint { font-size: 0.72rem; line-height: 1.45; color: var(--neutral-400); margin: -0.3rem 0 0.5rem; }

footer { display: none !important; }
"""
