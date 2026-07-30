"""
Gradio UI color presets for chat.py / evaluation.py.

Set environment variable COVE_UI_THEME to one of:
  default     — original white (good for screen)
  warm_gray   — cool gray page, soft cards (common in paper figures)
  ivory       — warm off-white, book/paper feel
  soft_blue   — very light blue-gray, distinct from white background
  sage        — muted green-gray, easy on the eyes in print

Example:
  COVE_UI_THEME=ivory python chat.py

Legacy env names (CVEE_UI_THEME, CVINF_UI_THEME, etc.) are still read if the COVE_* name is unset.
"""
from __future__ import annotations

from dataclasses import dataclass

from cove_paths import env_str

UI_THEME_ENV = "COVE_UI_THEME"
UI_THEME_ENV_LEGACY = ("CVEE_UI_THEME", "CVINF_UI_THEME")


@dataclass(frozen=True)
class _Theme:
    page: str
    card: str
    output: str
    hero_grad_1: str
    hero_grad_2: str
    hero_grad_3: str
    hero_text: str
    hero_muted: str
    hero_border: str
    card_border: str
    hero_shadow: str


_THEMES: dict[str, _Theme] = {
    "default": _Theme(
        page="#ffffff",
        card="#ffffff",
        output="#ffffff",
        hero_grad_1="#eff6ff",
        hero_grad_2="#ecfeff",
        hero_grad_3="#f8fafc",
        hero_text="#0f172a",
        hero_muted="#475569",
        hero_border="#dbeafe",
        card_border="#e5e7eb",
        hero_shadow="rgba(15, 23, 42, 0.06)",
    ),
    "warm_gray": _Theme(
        page="#e8edf2",
        card="#f1f5f9",
        output="#f1f5f9",
        hero_grad_1="#e2e8f0",
        hero_grad_2="#e8edf3",
        hero_grad_3="#f1f5f9",
        hero_text="#0f172a",
        hero_muted="#475569",
        hero_border="#cbd5e1",
        card_border="#cbd5e1",
        hero_shadow="rgba(15, 23, 42, 0.08)",
    ),
    "ivory": _Theme(
        page="#f5f0e8",
        card="#faf6ef",
        output="#faf6ef",
        hero_grad_1="#f0e8dc",
        hero_grad_2="#f5efe4",
        hero_grad_3="#faf6ef",
        hero_text="#1c1917",
        hero_muted="#57534e",
        hero_border="#d6d3d1",
        card_border="#d6d3d1",
        hero_shadow="rgba(28, 25, 23, 0.07)",
    ),
    "soft_blue": _Theme(
        page="#e8eef5",
        card="#f0f4f8",
        output="#f0f4f8",
        hero_grad_1="#dbe7f2",
        hero_grad_2="#e4edf5",
        hero_grad_3="#f0f4f8",
        hero_text="#0f172a",
        hero_muted="#475569",
        hero_border="#bfdbfe",
        card_border="#cbd5e1",
        hero_shadow="rgba(15, 23, 42, 0.07)",
    ),
    "sage": _Theme(
        page="#e9ede9",
        card="#f2f4f2",
        output="#f2f4f2",
        hero_grad_1="#dde5dd",
        hero_grad_2="#e5ebe5",
        hero_grad_3="#f2f4f2",
        hero_text="#1a1f1a",
        hero_muted="#4a524a",
        hero_border="#c5ccc5",
        card_border="#c5ccc5",
        hero_shadow="rgba(26, 31, 26, 0.07)",
    ),
}


def current_theme_name() -> str:
    name = env_str(UI_THEME_ENV, *UI_THEME_ENV_LEGACY) or "default"
    name = name.lower()
    if name in _THEMES:
        return name
    return "default"


def _t() -> _Theme:
    return _THEMES[current_theme_name()]


def build_gradio_css(*, max_width: str, with_chat_extras: bool) -> str:
    """Build injected Gradio CSS for the active COVE_UI_THEME."""
    t = _t()
    base = f"""
.gradio-container {{max-width: {max_width} !important; margin: 0 auto !important; background: {t.page} !important;}}
.hero {{
  padding: 22px 26px;
  border-radius: 18px;
  background: linear-gradient(135deg, {t.hero_grad_1} 0%, {t.hero_grad_2} 50%, {t.hero_grad_3} 100%);
  color: {t.hero_text};
  border: 1px solid {t.hero_border};
  box-shadow: 0 10px 30px {t.hero_shadow};
  margin-bottom: 18px;
}}
.hero h1 {{margin: 0 0 6px 0; font-size: 28px;}}
.hero p {{margin: 0; color: {t.hero_muted};}}
.section-card {{
  border: 1px solid {t.card_border};
  border-radius: 16px;
  padding: 16px;
  background: {t.card} !important;
}}
.section-card .form,
.section-card .block,
.section-card .gradio-row,
.section-card .wrap,
.section-card .gap,
.section-card .panel {{
  background: {t.card} !important;
}}
.primary-btn,
.primary-btn button {{
  width: 100% !important;
  font-weight: 800 !important;
  font-size: 16px !important;
  color: #ffffff !important;
  background: #2563eb !important;
  border: 1px solid #1d4ed8 !important;
}}
.primary-btn button:hover {{
  background: #1d4ed8 !important;
}}
"""
    if with_chat_extras:
        base += f"""
#chat-start-btn,
#zkllm-start-btn,
#chat-start-btn button,
#zkllm-start-btn button {{
  width: 100% !important;
  min-height: 52px !important;
  font-size: 18px !important;
  font-weight: 900 !important;
  color: #ffffff !important;
  background: #2563eb !important;
  border: 2px solid #1d4ed8 !important;
  border-radius: 12px !important;
  box-shadow: 0 8px 18px rgba(37, 99, 235, 0.22) !important;
}}
#chat-start-btn button:hover,
#zkllm-start-btn button:hover {{
  background: #1d4ed8 !important;
}}
.large-output textarea,
.large-output .wrap,
.large-output .prose {{
  background: {t.output} !important;
  color: #111827 !important;
  font-size: 17px !important;
  line-height: 1.6 !important;
}}
.large-output textarea {{
  min-height: 340px !important;
}}
.full-width textarea,
.full-width input {{
  width: 100% !important;
}}
.status-output textarea {{
  background: {t.output} !important;
  color: #111827 !important;
  font-size: 14px !important;
  line-height: 1.45 !important;
}}
"""
    else:
        base += """
.full-width-btn,
.full-width-btn button {
  width: 100% !important;
  min-height: 48px !important;
  font-size: 16px !important;
  font-weight: 800 !important;
  background: #0f766e !important;
  border: 2px solid #0f766e !important;
  color: #ffffff !important;
  border-radius: 12px !important;
}
.full-width-btn button:hover {
  background: #115e59 !important;
}
"""
    return base.strip() + "\n"
