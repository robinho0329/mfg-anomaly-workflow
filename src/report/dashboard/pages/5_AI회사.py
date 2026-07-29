"""④ 대시보드 — ⑤ AI 회사 오피스 페이지. (소유: mfg-reporter)

사규(AI_COMPANY.md)의 34명 편성과 하루 진행을 시각화한 오피스를 임베드한다.

왜 임베드인가:
    오피스는 33ms마다 캔버스를 다시 그리고 A* 로 인물을 움직이며, 대표 승인에서
    `await` 로 멈춘다. Streamlit 은 상호작용마다 스크립트를 재실행하는 모델이라
    이 셋을 네이티브로 옮길 수 없다. 브라우저 안에서 원본 JS 를 그대로 돌린다.

데이터 경로:
    오피스는 iframe srcdoc 안에서 돌아 상대경로 fetch 가 막힌다. 그래서 후보를
    `window.__CANDIDATES` 로 주입한다. 후보 파일은 깃허브 Actions 가 매일 아침
    ai-company-office 저장소에 커밋한 것을 읽는다(로컬 사본이 있으면 그것을 우선).
"""

import json
import urllib.error
import urllib.request

import streamlit as st
import streamlit.components.v1 as components

# ── sys.path 부트스트랩: 레포 루트(config)·대시보드 폴더(_lib) 보장 (로컬·클라우드 공통) ──
import sys as _sys
from pathlib import Path as _Path
for _anc in _Path(__file__).resolve().parents:
    if (_anc / "_lib.py").exists() and str(_anc) not in _sys.path:
        _sys.path.insert(0, str(_anc))
    if (_anc / "config" / "settings.py").exists() and str(_anc) not in _sys.path:
        _sys.path.insert(0, str(_anc))

from _lib import dash_header, inject_css, render_footer, render_sidebar  # noqa: E402

OFFICE_RAW = "https://raw.githubusercontent.com/robinho0329/ai-company-office/main"
LOCAL_OFFICE = _Path(__file__).resolve().parents[3].parent / "ai-company-office"
FETCH_TIMEOUT = 15


def _fetch(url: str) -> str | None:
    """원격 텍스트 한 건. 실패하면 None(호출측에서 폴백)."""
    try:
        with urllib.request.urlopen(url, timeout=FETCH_TIMEOUT) as res:
            return res.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def load_office() -> tuple[str | None, str]:
    """오피스 HTML — 로컬 사본 우선, 없으면 깃허브 원본. (본문, 출처)."""
    local = LOCAL_OFFICE / "index.html"
    if local.exists():
        return local.read_text(encoding="utf-8"), "로컬"
    html = _fetch(f"{OFFICE_RAW}/index.html")
    return (html, "깃허브") if html else (None, "실패")


@st.cache_data(ttl=1800, show_spinner=False)
def load_candidates() -> tuple[dict | None, str]:
    """과제 후보 — 로컬 산출물 우선, 없으면 깃허브(Actions 가 매일 커밋). (내용, 출처)."""
    for path in (LOCAL_OFFICE / "candidates.json",
                 _Path(__file__).resolve().parents[3] / "data" / "discovery" / "candidates.json"):
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8")), "로컬"
            except json.JSONDecodeError:
                pass
    raw = _fetch(f"{OFFICE_RAW}/candidates.json")
    if raw:
        try:
            return json.loads(raw), "깃허브"
        except json.JSONDecodeError:
            pass
    return None, "실패"


st.set_page_config(page_title="AI 회사 오피스", page_icon="🏢", layout="wide")
inject_css()
render_sidebar()
dash_header(
    "🏢 AI 회사 오피스",
    "사규(AI_COMPANY.md) 34명 편성 · 하루 16단계 · 채택/승인 2게이트",
)

st.info(
    "**무엇이 진짜이고 무엇이 대본인지** — 기획 후보와 그 근거 수치·파일 경로는 "
    "실제 저장소를 스캔해 뽑은 실측입니다. 직원 대사와 하루 진행, 상태 표시는 대본이며 "
    "실행 중인 작업의 상태가 아닙니다.",
    icon="ℹ️",
)

office_html, office_src = load_office()
cands, cand_src = load_candidates()

if office_html is None:
    st.error(
        "오피스 화면을 불러오지 못했습니다. 깃허브 원본에 접근할 수 없고 로컬 사본도 없습니다.\n\n"
        f"원본: {OFFICE_RAW}/index.html"
    )
    render_footer()
    st.stop()

# ── 후보 요약 — 임베드 밖에서도 무엇을 보는지 알 수 있게 ──
if cands and cands.get("projects"):
    projects = cands["projects"]
    cols = st.columns(min(len(projects), 4) or 1)
    for col, p in zip(cols, projects):
        col.metric(
            p["project"]["name"][:22],
            f"{len(p['A']) + len(p['B'])}건",
            f"신규 {len(p['A'])} · 개선 {len(p['B'])}",
        )
    st.caption(
        f"후보 출처: {cand_src} · 생성 {cands.get('generated', '-')[:16]} · "
        f"스캔 저장소 {len(cands.get('scanned_repos', []))}개 — "
        "질문을 발명하지 않으며, 근거 문장과 파일 경로가 없는 후보는 만들지 않습니다."
    )
else:
    st.warning(
        "후보 파일을 읽지 못해 오피스가 **내장 예시**로 진행됩니다. "
        "`python -m src.report.discover` 로 생성하거나, "
        "ai-company-office 저장소의 Discover Candidates 워크플로우를 실행하세요.",
        icon="⚠️",
    )

# ── 후보 주입 — iframe 안에서는 상대경로 fetch 가 막힌다 ──
payload = json.dumps(cands, ensure_ascii=False) if cands else "null"
injected = office_html.replace(
    "<script>",
    f"<script>window.__CANDIDATES = {payload};</script>\n<script>",
    1,
)

components.html(injected, height=880, scrolling=True)

with st.expander("이 화면은 어떻게 동작하나"):
    st.markdown(
        f"""
- **왜 임베드인가** — 오피스는 33ms마다 캔버스를 다시 그리고 A\\*로 34명을 움직이며,
  대표 승인 지점에서 `await`로 멈춥니다. Streamlit은 상호작용마다 스크립트를
  재실행하므로 이 셋을 네이티브로 옮길 수 없습니다. 브라우저 안에서 원본 JS를 그대로 돌립니다.
- **후보는 어디서** — 깃허브 Actions가 매일 08:30 KST에 보유 저장소를 스캔해
  `candidates.json`을 커밋합니다. 이 페이지는 그걸 읽어 `window.__CANDIDATES`로 주입합니다.
- **채택하면** — 오피스가 작업 지시서를 만듭니다. 지시창의 **지시서** 버튼으로 꺼내
  복사한 뒤 작업 세션에 붙여넣으면 실제로 진행됩니다. 이 화면은 코드를 고치지 못합니다.
- **현재 소스** — 오피스 HTML: `{office_src}` · 후보: `{cand_src}`
- **원본 저장소** — [ai-company-office](https://github.com/robinho0329/ai-company-office)
"""
    )

render_footer()
