"""TEP 공정 P&ID(SVG) 생성기. (소유: mfg-reporter)

ISA 스타일 계기 버블·제어밸브·번호 스트림을 배치해 Tennessee Eastman Process의
파이핑·계장도를 벡터로 그린다. 산출: src/report/dashboard/assets/tep_pid.svg

실행: python -m scripts.gen_tep_pid
"""
from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "src" / "report" / "dashboard" / "assets" / "tep_pid.svg"

# 색상(인더스트리얼 테마)
EQ = "#34495E"       # 장치 외곽
EQ_FILL = "#f4f6f8"
PIPE = "#4a5a68"
INSTR = "#2C7BE5"    # 계기(측정) 버블
VALVE = "#E67E22"    # 제어밸브(XMV)
PROD = "#27AE60"
CW = "#17A2B8"       # 냉각수/스팀
INK = "#1c2733"
MUT = "#7b8a9a"

_parts: list[str] = []


def add(s: str) -> None:
    _parts.append(s)


def pipe(pts, color=PIPE, w=2.2, dash="", arrow=True):
    d = " ".join(f"{'M' if i == 0 else 'L'} {x},{y}" for i, (x, y) in enumerate(pts))
    da = f' stroke-dasharray="{dash}"' if dash else ""
    mk = ' marker-end="url(#arrow)"' if arrow else ""
    add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{w}"{da}{mk}/>')


def bubble(cx, cy, fn, tag, color=INSTR):
    """ISA 필드 계기 버블: 원 + 기능문자 + 태그번호."""
    add(f'<circle cx="{cx}" cy="{cy}" r="16" fill="#fff" stroke="{color}" stroke-width="1.6"/>')
    add(f'<line x1="{cx-16}" y1="{cy}" x2="{cx+16}" y2="{cy}" stroke="{color}" stroke-width="0.7"/>')
    add(f'<text x="{cx}" y="{cy-3}" text-anchor="middle" font-size="9.5" font-weight="700" fill="{INK}">{fn}</text>')
    add(f'<text x="{cx}" y="{cy+11}" text-anchor="middle" font-size="8.5" fill="{MUT}">{tag}</text>')


def cvalve(cx, cy, tag, vert=False):
    """제어밸브(보타이) + 다이어프램 액추에이터 + XMV 태그."""
    if not vert:
        add(f'<path d="M {cx-11},{cy-9} L {cx-11},{cy+9} L {cx},{cy} Z" fill="{VALVE}"/>')
        add(f'<path d="M {cx+11},{cy-9} L {cx+11},{cy+9} L {cx},{cy} Z" fill="{VALVE}"/>')
    else:
        add(f'<path d="M {cx-9},{cy-11} L {cx+9},{cy-11} L {cx},{cy} Z" fill="{VALVE}"/>')
        add(f'<path d="M {cx-9},{cy+11} L {cx+9},{cy+11} L {cx},{cy} Z" fill="{VALVE}"/>')
    # 액추에이터(다이어프램)
    add(f'<line x1="{cx}" y1="{cy}" x2="{cx}" y2="{cy-16}" stroke="{VALVE}" stroke-width="1.4"/>')
    add(f'<path d="M {cx-7},{cy-16} q7,-7 14,0 z" fill="none" stroke="{VALVE}" stroke-width="1.4"/>')
    add(f'<text x="{cx}" y="{cy-24}" text-anchor="middle" font-size="8.5" font-weight="700" fill="{VALVE}">{tag}</text>')


def vessel(x, y, w, h, name, sub=""):
    """수직 용기(원통) — 상/하 반타원."""
    rx, ry = w / 2, 9
    cx = x + rx
    add(f'<path d="M {x},{y+ry} A {rx},{ry} 0 0 1 {x+w},{y+ry} L {x+w},{y+h-ry} '
        f'A {rx},{ry} 0 0 1 {x},{y+h-ry} Z" fill="{EQ_FILL}" stroke="{EQ}" stroke-width="1.8"/>')
    add(f'<ellipse cx="{cx}" cy="{y+ry}" rx="{rx}" ry="{ry}" fill="{EQ_FILL}" stroke="{EQ}" stroke-width="1.8"/>')
    add(f'<text x="{cx}" y="{y+h/2}" text-anchor="middle" font-size="12" font-weight="800" fill="{INK}">{name}</text>')
    if sub:
        add(f'<text x="{cx}" y="{y+h/2+15}" text-anchor="middle" font-size="9" fill="{MUT}">{sub}</text>')


def column(x, y, w, h, name, sub=""):
    """스트리퍼 컬럼 — 트레이 표시."""
    cx = x + w / 2
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{EQ_FILL}" stroke="{EQ}" stroke-width="1.8"/>')
    for k in range(1, 5):
        ty = y + h * k / 5
        add(f'<line x1="{x+6}" y1="{ty}" x2="{x+w-6}" y2="{ty}" stroke="{EQ}" stroke-width="0.8" opacity="0.5"/>')
    add(f'<text x="{cx}" y="{y+h/2}" text-anchor="middle" font-size="12" font-weight="800" fill="{INK}" '
        f'transform="rotate(-90 {cx},{y+h/2})">{name}</text>')
    if sub:
        add(f'<text x="{cx}" y="{y+h+14}" text-anchor="middle" font-size="9" fill="{MUT}">{sub}</text>')


def hx(x, y, w, h, name):
    """열교환기(응축기) — 튜브 표시."""
    cx = x + w / 2
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{EQ_FILL}" stroke="{EQ}" stroke-width="1.8"/>')
    add(f'<path d="M {x+8},{y+h/2} l 12,-{h/3} l 20,{2*h/3} l 20,-{2*h/3} l 20,{2*h/3} l 12,-{h/3}" '
        f'fill="none" stroke="{EQ}" stroke-width="1" opacity="0.55"/>')
    add(f'<text x="{cx}" y="{y+h+14}" text-anchor="middle" font-size="11" font-weight="800" fill="{INK}">{name}</text>')


def compressor(cx, cy, r, tag):
    add(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{EQ_FILL}" stroke="{EQ}" stroke-width="1.8"/>')
    add(f'<path d="M {cx-r},{cy-r*0.7} L {cx+r},{cy-r*0.35} M {cx-r},{cy+r*0.7} L {cx+r},{cy+r*0.35}" '
        f'stroke="{EQ}" stroke-width="1.2"/>')
    add(f'<text x="{cx}" y="{cy+r+13}" text-anchor="middle" font-size="10" font-weight="700" fill="{INK}">{tag}</text>')


def stream_no(x, y, n):
    add(f'<circle cx="{x}" cy="{y}" r="9.5" fill="#fff" stroke="{PIPE}" stroke-width="1.2"/>')
    add(f'<text x="{x}" y="{y+3.3}" text-anchor="middle" font-size="9" font-weight="800" fill="{PIPE}">{n}</text>')


def feed(x, y, text):
    add(f'<text x="{x}" y="{y+4}" text-anchor="end" font-size="10.5" font-weight="700" fill="{INK}">{text}</text>')


def label(x, y, text, size=10, color=MUT, weight=400, anchor="middle"):
    add(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" font-weight="{weight}" fill="{color}">{text}</text>')


# ── 캔버스 ────────────────────────────────────────────────────────────
W, H = 1200, 660
add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">')
add('<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto" '
    f'markerUnits="strokeWidth"><path d="M0,0 L7,3 L0,6 Z" fill="{PIPE}"/></marker></defs>')
add(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>')
label(60, 30, "Tennessee Eastman Process — 계통 P&amp;ID (개략)", 15, INK, 800, "start")

# ── 공급 A·D·E (stream 1·2·3) → 반응기 헤더 ──────────────────────────
feeds = [("A 공급", 175, "FC", "3", "XMV3", "1"),
         ("D 공급", 235, "FC", "1", "XMV1", "2"),
         ("E 공급", 295, "FC", "2", "XMV2", "3")]
for name, fy, fn, ftag, xmv, sn in feeds:
    feed(70, fy, name)
    pipe([(78, fy), (110, fy)])
    bubble(126, fy, fn, ftag)
    pipe([(142, fy), (176, fy)], arrow=False)
    cvalve(192, fy, xmv)
    pipe([(208, fy), (250, fy)], arrow=False)
    stream_no(230, fy - 13, sn)
# 수직 헤더 → 반응기
pipe([(250, 175), (250, 295)], arrow=False)
pipe([(250, 235), (305, 235)])

# ── 반응기 ────────────────────────────────────────────────────────────
vessel(305, 150, 88, 215, "반응기", "Reactor")
RCX = 349
bubble(322, 178, "PI", "7")        # 반응기 압력
bubble(376, 205, "TI", "9")        # 반응기 온도
bubble(376, 300, "LI", "8")        # 반응기 액위
# 반응기 냉각수(자켓) + XMV10
label(300, 400, "CW", 9, CW, 700, "end")
pipe([(270, 396), (305, 340)], color=CW, w=1.8)
cvalve(288, 385, "XMV10")
label(360, 392, "반응기 냉각수", 8.5, CW)

# ── 반응기 → 응축기 (stream 7) ───────────────────────────────────────
pipe([(349, 150), (349, 112), (470, 112)])
stream_no(430, 100, "7")

# ── 응축기 ────────────────────────────────────────────────────────────
hx(470, 92, 120, 42, "응축기 Condenser")
# 응축기 냉각수 + XMV11
cvalve(530, 60, "XMV11")
pipe([(530, 70), (530, 92)], color=CW, w=1.8, arrow=False)
label(568, 58, "냉각수", 8.5, CW)

# ── 응축기 → 분리기 (stream 8) ───────────────────────────────────────
pipe([(590, 112), (660, 112), (660, 150)])
stream_no(636, 100, "8")

# ── 기액분리기 ────────────────────────────────────────────────────────
vessel(618, 150, 88, 200, "", "")
SCX = 662
# 2행 라벨은 수동
add(f'<text x="{SCX}" y="245" text-anchor="middle" font-size="12" font-weight="800" fill="{INK}">기액분리기</text>')
add(f'<text x="{SCX}" y="260" text-anchor="middle" font-size="9" fill="{MUT}">Separator</text>')
bubble(634, 178, "PI", "13")
bubble(690, 205, "TI", "11")
bubble(690, 300, "LI", "12")

# ── 분리기 증기 → 압축기 → 재순환/퍼지 ───────────────────────────────
pipe([(662, 150), (662, 96)], arrow=False)
compressor(662, 74, 22, "압축기 K-1")
# 압축기 일 계기
bubble(710, 74, "JI", "20")
# 재순환(stream 8 recycle): 압축기 → 좌측 → 반응기 상부 헤더
cvalve(590, 74, "XMV5")
pipe([(640, 74), (606, 74)], arrow=False)
pipe([(574, 74), (250, 74), (250, 175)], dash="7,5", arrow=True)
label(430, 66, "재순환 가스 (stream 8)", 9, MUT)
stream_no(300, 74, "8")
# 퍼지(stream 9): 압축기 상단 → XMV6 → AI → 배출
pipe([(662, 52), (662, 30)], arrow=False)
cvalve(720, 30, "XMV6")
pipe([(662, 30), (704, 30)], arrow=False)
bubble(760, 30, "AI", "R", INSTR)
pipe([(776, 30), (820, 30)])
label(846, 34, "퍼지 Purge", 10, INK, 700, "start")
stream_no(690, 20, "9")

# ── 분리기 액 → 스트리퍼 (stream 10) ─────────────────────────────────
pipe([(662, 350), (662, 400), (760, 400)], arrow=False)
cvalve(715, 400, "XMV7")
pipe([(760, 400), (858, 400), (858, 300)], arrow=True)
stream_no(792, 388, "10")

# ── 공급 A·B·C (stream 4) → 스트리퍼 하부 ────────────────────────────
feed(560, 560, "A·B·C 공급")
pipe([(568, 560), (600, 560)])
bubble(616, 560, "FC", "4")
pipe([(632, 560), (664, 560)], arrow=False)
cvalve(680, 560, "XMV4")
pipe([(696, 560), (830, 560), (830, 470)], arrow=True)
stream_no(760, 548, "4")
pipe([(830, 470), (862, 470)])

# ── 스트리퍼 ──────────────────────────────────────────────────────────
column(862, 190, 74, 250, "스트리퍼", "")
STX = 899
bubble(946, 220, "PI", "16")
bubble(946, 300, "TI", "18")
bubble(946, 400, "LI", "15")
# 재비등 스팀 + XMV9
label(860, 470, "스팀", 9, CW, 700, "end")
cvalve(899, 470, "XMV9")
pipe([(899, 458), (899, 440)], color=CW, w=1.8, arrow=True)
# 스트리퍼 오버헤드(stream 5) → 재순환(상부)
pipe([(899, 190), (899, 150), (760, 150)], dash="7,5")
label(820, 142, "오버헤드 재순환 (stream 5)", 8.5, MUT)
stream_no(870, 165, "5")

# ── 스트리퍼 바닥 → 제품 (stream 11) ─────────────────────────────────
pipe([(936, 415), (990, 415)], arrow=False)
cvalve(1006, 415, "XMV8")
pipe([(1022, 415), (1060, 415)], arrow=False)
bubble(1090, 415, "AI", "P", INSTR)
pipe([(1106, 415), (1150, 415)])
add(f'<rect x="1074" y="450" width="96" height="34" rx="6" fill="#d7f0df" stroke="{PROD}" stroke-width="1.8"/>')
add(f'<text x="1122" y="471" text-anchor="middle" font-size="11" font-weight="800" fill="{INK}">제품 G · H</text>')
pipe([(1122, 415), (1122, 450)], color=PROD, arrow=True)
stream_no(1046, 403, "11")

# ── 범례 ──────────────────────────────────────────────────────────────
ly = 612
add(f'<line x1="60" y1="{ly}" x2="{W-60}" y2="{ly-1}" stroke="#e6ebf0" stroke-width="1"/>')
add(f'<circle cx="80" cy="{ly+18}" r="9" fill="#fff" stroke="{INSTR}" stroke-width="1.4"/>')
label(96, ly + 22, "계기 버블(측정: FI/PI/TI/LI/AI 등 XMEAS)", 9, INK, 400, "start")
add(f'<path d="M 400,{ly+11} L 400,{ly+25} L 409,{ly+18} Z M 418,{ly+11} L 418,{ly+25} L 409,{ly+18} Z" fill="{VALVE}"/>')
label(426, ly + 22, "제어밸브(조작변수 XMV 1~11)", 9, INK, 400, "start")
add(f'<circle cx="680" cy="{ly+18}" r="9" fill="#fff" stroke="{PIPE}" stroke-width="1.2"/>')
label(696, ly + 22, "번호 = 공정 스트림(1~11)", 9, INK, 400, "start")
add(f'<line x1="900" y1="{ly+18}" x2="930" y2="{ly+18}" stroke="{MUT}" stroke-width="2" stroke-dasharray="7,5"/>')
label(936, ly + 22, "재순환 라인", 9, INK, 400, "start")

add("</svg>")
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(_parts), encoding="utf-8")
print(f"생성: {OUT} ({OUT.stat().st_size/1024:.1f} KB)")
