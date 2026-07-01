"""TEP 공정 P&ID(SVG) 생성기. (소유: mfg-reporter)

ISA 스타일 계기 버블·제어밸브·번호 스트림을 배치해 Tennessee Eastman Process
(Downs & Vogel, 1993)의 파이핑·계장도를 벡터로 그린다.
산출: src/report/dashboard/assets/tep_pid.svg

토폴로지: 공급(A·D·E) + 재순환(압축기 8 · 스트리퍼 오버헤드 5) → 반응기 피드(6)
→ 반응기(내부 냉각코일) → 응축기 → 기액분리기 → {압축기→재순환 / 퍼지 9}
· 분리기액(10) → 스트리퍼(+A·B·C 공급 4) → 제품 G·H(11).

실행: python -m scripts.gen_tep_pid
"""
from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "src" / "report" / "dashboard" / "assets" / "tep_pid.svg"

EQ, EQ_FILL, PIPE = "#34495E", "#f4f6f8", "#4a5a68"
INSTR, VALVE, PROD, CW = "#2C7BE5", "#E67E22", "#27AE60", "#17A2B8"
INK, MUT = "#1c2733", "#7b8a9a"

_parts: list[str] = []


def add(s: str) -> None:
    _parts.append(s)


def pipe(pts, color=PIPE, w=2.2, dash="", arrow=True):
    d = " ".join(f"{'M' if i == 0 else 'L'} {x},{y}" for i, (x, y) in enumerate(pts))
    da = f' stroke-dasharray="{dash}"' if dash else ""
    mk = ' marker-end="url(#arrow)"' if arrow else ""
    add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{w}"{da}{mk}/>')


def bubble(cx, cy, fn, tag, color=INSTR, leader=None):
    """ISA 필드 계기 버블. leader=(x,y)면 파이프까지 리더선."""
    if leader:
        add(f'<line x1="{cx}" y1="{cy}" x2="{leader[0]}" y2="{leader[1]}" stroke="{color}" stroke-width="0.9" stroke-dasharray="2,2"/>')
    add(f'<circle cx="{cx}" cy="{cy}" r="15.5" fill="#fff" stroke="{color}" stroke-width="1.6"/>')
    add(f'<line x1="{cx-15.5}" y1="{cy}" x2="{cx+15.5}" y2="{cy}" stroke="{color}" stroke-width="0.7"/>')
    add(f'<text x="{cx}" y="{cy-3}" text-anchor="middle" font-size="9" font-weight="700" fill="{INK}">{fn}</text>')
    add(f'<text x="{cx}" y="{cy+10.5}" text-anchor="middle" font-size="8" fill="{MUT}">{tag}</text>')


def cvalve(cx, cy, tag, tagpos="top"):
    """제어밸브(보타이) + 다이어프램 액추에이터 + XMV 태그."""
    add(f'<path d="M {cx-11},{cy-9} L {cx-11},{cy+9} L {cx},{cy} Z" fill="{VALVE}"/>')
    add(f'<path d="M {cx+11},{cy-9} L {cx+11},{cy+9} L {cx},{cy} Z" fill="{VALVE}"/>')
    add(f'<line x1="{cx}" y1="{cy}" x2="{cx}" y2="{cy-15}" stroke="{VALVE}" stroke-width="1.4"/>')
    add(f'<path d="M {cx-7},{cy-15} q7,-7 14,0 z" fill="none" stroke="{VALVE}" stroke-width="1.4"/>')
    if tagpos == "top":
        add(f'<text x="{cx}" y="{cy-23}" text-anchor="middle" font-size="8.5" font-weight="700" fill="{VALVE}">{tag}</text>')
    else:
        add(f'<text x="{cx+15}" y="{cy+3}" text-anchor="start" font-size="8.5" font-weight="700" fill="{VALVE}">{tag}</text>')


def vessel(x, y, w, h, name, sub=""):
    rx, ry = w / 2, 9
    cx = x + rx
    add(f'<path d="M {x},{y+ry} A {rx},{ry} 0 0 1 {x+w},{y+ry} L {x+w},{y+h-ry} '
        f'A {rx},{ry} 0 0 1 {x},{y+h-ry} Z" fill="{EQ_FILL}" stroke="{EQ}" stroke-width="1.8"/>')
    add(f'<ellipse cx="{cx}" cy="{y+ry}" rx="{rx}" ry="{ry}" fill="{EQ_FILL}" stroke="{EQ}" stroke-width="1.8"/>')
    if name:
        add(f'<text x="{cx}" y="{y+h/2}" text-anchor="middle" font-size="12" font-weight="800" fill="{INK}">{name}</text>')
    if sub:
        add(f'<text x="{cx}" y="{y+h/2+15}" text-anchor="middle" font-size="9" fill="{MUT}">{sub}</text>')


def column(x, y, w, h, name):
    cx = x + w / 2
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{EQ_FILL}" stroke="{EQ}" stroke-width="1.8"/>')
    for k in range(1, 5):
        ty = y + h * k / 5
        add(f'<line x1="{x+6}" y1="{ty}" x2="{x+w-6}" y2="{ty}" stroke="{EQ}" stroke-width="0.8" opacity="0.5"/>')
    add(f'<text x="{cx}" y="{y+h/2}" text-anchor="middle" font-size="12" font-weight="800" fill="{INK}" '
        f'transform="rotate(-90 {cx},{y+h/2})">{name}</text>')


def hx(x, y, w, h, name):
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


def coil(cx, y0, y1, loops=7, r=10):
    """반응기 내부 냉각 코일(서펜타인)."""
    step = (y1 - y0) / loops
    d = f"M {cx},{y0}"
    for i in range(loops):
        yy = y0 + step * (i + 1)
        sweep = 1 if i % 2 == 0 else 0
        d += f" A {r},{step/2:.1f} 0 0 {sweep} {cx},{yy:.1f}"
    add(f'<path d="{d}" fill="none" stroke="{CW}" stroke-width="1.7" opacity="0.85"/>')


def stream_no(x, y, n):
    add(f'<circle cx="{x}" cy="{y}" r="9.5" fill="#fff" stroke="{PIPE}" stroke-width="1.2"/>')
    add(f'<text x="{x}" y="{y+3.3}" text-anchor="middle" font-size="9" font-weight="800" fill="{PIPE}">{n}</text>')


def feed(x, y, text):
    add(f'<text x="{x}" y="{y+4}" text-anchor="end" font-size="10.5" font-weight="700" fill="{INK}">{text}</text>')


def label(x, y, text, size=10, color=MUT, weight=400, anchor="middle"):
    add(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" font-weight="{weight}" fill="{color}">{text}</text>')


# ── 캔버스 ────────────────────────────────────────────────────────────
W, H = 1270, 720
add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">')
add('<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto" '
    f'markerUnits="strokeWidth"><path d="M0,0 L7,3 L0,6 Z" fill="{PIPE}"/></marker></defs>')
add(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>')
label(60, 32, "Tennessee Eastman Process — 계통 P&amp;ID (개략)", 15, INK, 800, "start")

# ── 공급 A·D·E (stream 1·2·3) → 헤더 ─────────────────────────────────
JX, JY = 288, 300           # 반응기 피드 믹싱 junction
feeds = [("A 공급", 240, "FC", "3", "XMV3", "1"),
         ("D 공급", 300, "FC", "1", "XMV1", "2"),
         ("E 공급", 360, "FC", "2", "XMV2", "3")]
for name, fy, fn, ftag, xmv, sn in feeds:
    feed(78, fy, name)
    pipe([(86, fy), (116, fy)])
    bubble(132, fy, fn, ftag)
    pipe([(148, fy), (182, fy)], arrow=False)
    cvalve(198, fy, xmv)
    pipe([(214, fy), (250, fy)], arrow=False)
    stream_no(236, fy - 13, sn)
pipe([(250, 240), (250, 360)], arrow=False)   # 수직 헤더
pipe([(250, 300), (JX, JY)], arrow=False)

# ── 반응기 피드(stream 6): 공급 + 재순환 믹싱 → 반응기 ───────────────
add(f'<circle cx="{JX}" cy="{JY}" r="3.5" fill="{PIPE}"/>')   # junction 점
bubble(320, 258, "FI", "6", leader=(320, JY))
bubble(352, 258, "AI", "6", leader=(352, JY))                 # 반응기 피드 조성분석기
stream_no(304, JY - 13, "6")
pipe([(JX, JY), (372, JY)])                                   # → 반응기

# ── 반응기 (내부 냉각코일) ───────────────────────────────────────────
RX, RW = 372, 96
vessel(RX, 200, RW, 220, "반응기", "Reactor")
RCX = RX + RW / 2
coil(RCX + 26, 224, 400, loops=8, r=9)                        # 내부 냉각 코일
bubble(RX + 20, 226, "PI", "7")
bubble(RX + RW + 6, 262, "TI", "9", leader=(RX + RW, 262))
bubble(RX + RW + 6, 356, "LI", "8", leader=(RX + RW, 356))
# 냉각수: 입구(하단, XMV10) · 출구온도(TI-21)
label(318, 452, "냉각수", 8.5, CW, 700, "end")
cvalve(340, 440, "XMV10")
pipe([(324, 440), (RCX + 26, 400)], color=CW, w=1.8)
bubble(322, 214, "TI", "21", color=CW, leader=(RCX + 26, 224))
label(360, 470, "반응기 냉각수(XMV10) · 출구온도 TI-21", 8.5, CW)

# ── 반응기 → 응축기 (stream 7) ───────────────────────────────────────
pipe([(RCX, 200), (RCX, 150), (528, 150)])
stream_no(486, 138, "7")

# ── 응축기 ────────────────────────────────────────────────────────────
hx(528, 128, 128, 44, "응축기 Condenser")
cvalve(588, 96, "XMV11")
pipe([(588, 106), (588, 128)], color=CW, w=1.8, arrow=False)
bubble(636, 96, "TI", "22", color=CW, leader=(614, 108))
label(592, 118, "냉각수", 8, CW, 700, "start")

# ── 응축기 → 분리기 ──────────────────────────────────────────────────
pipe([(656, 150), (726, 150), (726, 200)])

# ── 기액분리기 ────────────────────────────────────────────────────────
SX, SW = 682, 92
vessel(SX, 200, SW, 190, "", "")
SCX = SX + SW / 2
add(f'<text x="{SCX}" y="292" text-anchor="middle" font-size="12" font-weight="800" fill="{INK}">기액분리기</text>')
add(f'<text x="{SCX}" y="307" text-anchor="middle" font-size="9" fill="{MUT}">Separator</text>')
bubble(SX + 20, 226, "PI", "13")
bubble(SX + SW + 6, 262, "TI", "11", leader=(SX + SW, 262))
bubble(SX + SW + 6, 348, "LI", "12", leader=(SX + SW, 348))

# ── 분리기 증기 → 압축기 → 재순환/퍼지 ───────────────────────────────
KX, KY = SCX, 108
pipe([(SCX, 200), (SCX, KY + 22)], arrow=False)
compressor(KX, KY, 23, "압축기 K-1")
bubble(KX + 48, KY, "JI", "20")
# 퍼지(stream 9): 압축기 상단 → XMV6 → AI → 배출
pipe([(KX, KY - 23), (KX, 44)], arrow=False)
cvalve(KX + 58, 44, "XMV6")
pipe([(KX, 44), (KX + 42, 44)], arrow=False)
bubble(KX + 100, 44, "AI", "9", leader=(KX + 84, 44))
pipe([(KX + 116, 44), (KX + 170, 44)])
label(KX + 178, 48, "퍼지 Purge", 10, INK, 700, "start")
stream_no(KX + 26, 34, "9")
# 재순환 가스(stream 8): 압축기 → XMV5 → 좌측 → junction
cvalve(KX - 66, KY, "XMV5", tagpos="side")
pipe([(KX - 23, KY), (KX - 55, KY)], arrow=False)
pipe([(KX - 77, KY), (JX, KY), (JX, JY)], dash="7,5")
stream_no(360, KY, "8")
label(470, KY - 8, "재순환 가스 (stream 8)", 9, MUT)

# ── 분리기 액 → 스트리퍼 (stream 10) ─────────────────────────────────
pipe([(SCX, 390), (SCX, 445), (812, 445)], arrow=False)
bubble(760, 415, "FI", "14", leader=(760, 445))
cvalve(828, 445, "XMV7")
pipe([(844, 445), (912, 445), (912, 320)])
stream_no(792, 433, "10")

# ── 공급 A·B·C (stream 4) → 스트리퍼 하부 ────────────────────────────
feed(556, 620, "A·B·C 공급")
pipe([(564, 620), (596, 620)])
bubble(612, 620, "FC", "4")
pipe([(628, 620), (662, 620)], arrow=False)
cvalve(678, 620, "XMV4")
pipe([(694, 620), (872, 620), (872, 430)])
stream_no(792, 608, "4")
pipe([(872, 430), (914, 430)], arrow=False)

# ── 스트리퍼 (재비등 스팀) ───────────────────────────────────────────
STX, STW = 914, 80
column(STX, 210, STW, 258, "스트리퍼")
STCX = STX + STW / 2
bubble(STX + STW + 6, 246, "PI", "16", leader=(STX + STW, 246))
bubble(STX + STW + 6, 330, "TI", "18", leader=(STX + STW, 330))
bubble(STX + STW + 6, 420, "LI", "15", leader=(STX + STW, 420))
# 재비등 스팀 + XMV9 + FI-19
label(886, 500, "스팀", 9, CW, 700, "end")
cvalve(STCX, 500, "XMV9")
pipe([(STCX, 488), (STCX, 468)], color=CW, w=1.8)
bubble(STCX - 44, 500, "FI", "19", color=CW, leader=(STCX - 11, 500))
# 오버헤드 재순환(stream 5) → 반응기 피드 (상단, junction 합류)
pipe([(STCX, 210), (STCX, 66), (JX + 14, 66)], dash="7,5", arrow=False)
# junction 근처에서 재순환 라인(8)과 합류
pipe([(JX + 14, 66), (JX, 66), (JX, KY)], dash="7,5", arrow=False)
stream_no(STCX - 60, 66, "5")
label(700, 58, "스트리퍼 오버헤드 재순환 (stream 5)", 8.5, MUT)

# ── 스트리퍼 바닥 → 제품 (stream 11) ─────────────────────────────────
pipe([(STX + STW, 452), (1004, 452)], arrow=False)
bubble(986, 418, "FI", "17", leader=(986, 452))
cvalve(1024, 452, "XMV8")
pipe([(1040, 452), (1074, 452)], arrow=False)
bubble(1106, 452, "AI", "11", leader=(1090, 452))
pipe([(1122, 452), (1160, 452)], arrow=False)
add(f'<rect x="1112" y="486" width="96" height="34" rx="6" fill="#d7f0df" stroke="{PROD}" stroke-width="1.8"/>')
add(f'<text x="1160" y="507" text-anchor="middle" font-size="11" font-weight="800" fill="{INK}">제품 G · H</text>')
pipe([(1160, 452), (1160, 486)], color=PROD)
stream_no(1058, 440, "11")

# ── 범례 ──────────────────────────────────────────────────────────────
ly = 665
add(f'<line x1="60" y1="{ly}" x2="{W-60}" y2="{ly}" stroke="#e6ebf0" stroke-width="1"/>')
add(f'<circle cx="80" cy="{ly+22}" r="9" fill="#fff" stroke="{INSTR}" stroke-width="1.4"/>')
label(96, ly + 26, "계기 버블 — 측정변수 XMEAS (FI·PI·TI·LI·AI·JI)", 9, INK, 400, "start")
add(f'<path d="M 430,{ly+15} L 430,{ly+29} L 439,{ly+22} Z M 448,{ly+15} L 448,{ly+29} L 439,{ly+22} Z" fill="{VALVE}"/>')
label(456, ly + 26, "제어밸브 — 조작변수 XMV 1~11", 9, INK, 400, "start")
add(f'<circle cx="720" cy="{ly+22}" r="9" fill="#fff" stroke="{PIPE}" stroke-width="1.2"/>')
label(736, ly + 26, "번호 = 공정 스트림 1~11", 9, INK, 400, "start")
add(f'<line x1="920" y1="{ly+22}" x2="952" y2="{ly+22}" stroke="{MUT}" stroke-width="2" stroke-dasharray="7,5"/>')
label(958, ly + 26, "재순환 라인", 9, INK, 400, "start")
add(f'<path d="M 1050,{ly+22} q10,-8 20,0" fill="none" stroke="{CW}" stroke-width="1.7"/>')
label(1078, ly + 26, "냉각수/스팀", 9, INK, 400, "start")

add("</svg>")
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(_parts), encoding="utf-8")
print(f"생성: {OUT} ({OUT.stat().st_size/1024:.1f} KB)")
