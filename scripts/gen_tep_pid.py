"""TEP 공정 P&ID(SVG) 생성기 — ISA 엔지니어링 계장도 스타일. (소유: mfg-reporter)

일반 산업 P&ID 관례(제어 loop별 계기 버블 FE·FT·FC·FCV, 신호선, 컨트롤러 버블,
알람 버블)를 Tennessee Eastman Process(Downs & Vogel, 1993)에 적용한 흑백 계장도.
산출: src/report/dashboard/assets/tep_pid.svg

제어 loop: 4개 공급 유량 · 반응기(압력·액위·온도) · 분리기 액위 · 스트리퍼(액위·스팀)
· 재순환/퍼지 · 응축기 냉각수. 조작변수(밸브)=XMV 1~11, 측정=XMEAS.

실행: python -m scripts.gen_tep_pid
"""
from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "src" / "report" / "dashboard" / "assets" / "tep_pid.svg"

INK = "#1f2933"      # 도면 선/글자(엔지니어링 블랙)
FILL = "#ffffff"
SIG = "#8a97a3"      # 계기 신호선(점선)
PROD = "#2f855a"     # 제품(유일한 강조색)

_p: list[str] = []


def add(s: str) -> None:
    _p.append(s)


def pipe(pts, w=2, arrow=True):
    d = " ".join(f"{'M' if i == 0 else 'L'} {x},{y}" for i, (x, y) in enumerate(pts))
    mk = ' marker-end="url(#arw)"' if arrow else ""
    add(f'<path d="{d}" fill="none" stroke="{INK}" stroke-width="{w}"{mk}/>')


def sig(pts):
    """계기 신호선(가는 점선)."""
    d = " ".join(f"{'M' if i == 0 else 'L'} {x},{y}" for i, (x, y) in enumerate(pts))
    add(f'<path d="{d}" fill="none" stroke="{SIG}" stroke-width="1" stroke-dasharray="4,3"/>')


def instr(cx, cy, fn, tag, dcs=False, r=17):
    """ISA 계기 버블. dcs=True면 공유표시(컨트롤러) — 가운데 가로줄."""
    add(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{FILL}" stroke="{INK}" stroke-width="1.4"/>')
    if dcs:
        add(f'<line x1="{cx-r}" y1="{cy}" x2="{cx+r}" y2="{cy}" stroke="{INK}" stroke-width="1.4"/>')
    add(f'<text x="{cx}" y="{cy-3}" text-anchor="middle" font-size="8.5" font-weight="700" fill="{INK}">{fn}</text>')
    add(f'<text x="{cx}" y="{cy+9}" text-anchor="middle" font-size="8" fill="{INK}">{tag}</text>')


def loop(x, y, fn_field, fn_ctrl, tag, valve_xy, r=16, up=True):
    """유량류 제어 loop: 파이프 위 오리피스 → 트랜스미터 → 컨트롤러 → (신호) → 밸브.

    x,y = 파이프 위 측정점. valve_xy = 제어밸브 위치.
    """
    # 오리피스(FE) 플레이트
    add(f'<line x1="{x-4}" y1="{y-8}" x2="{x-4}" y2="{y+8}" stroke="{INK}" stroke-width="1.4"/>')
    add(f'<line x1="{x+4}" y1="{y-8}" x2="{x+4}" y2="{y+8}" stroke="{INK}" stroke-width="1.4"/>')
    dy = -1 if up else 1
    ty = y + dy * 40      # 트랜스미터
    cy = y + dy * 78      # 컨트롤러
    sig([(x, y), (x, ty + dy * -r if up else ty - r)])
    instr(x, ty, fn_field, tag, r=r)          # 트랜스미터(field)
    sig([(x, ty + dy * r), (x, cy + dy * -r if up else cy - r)])
    instr(x, cy, fn_ctrl, tag, dcs=True, r=r)  # 컨트롤러(DCS)
    # 컨트롤러 → 밸브 신호
    vx, vy = valve_xy
    sig([(x, cy + dy * r if up else cy + r), (x, cy + dy * (r + 12)),
         (vx, cy + dy * (r + 12)), (vx, vy - 15)])


def cvalve(cx, cy, tag, tagdx=0, tagdy=-22):
    """제어밸브(보타이) + 다이어프램 액추에이터 + XMV 태그."""
    add(f'<path d="M {cx-11},{cy-9} L {cx-11},{cy+9} L {cx},{cy} Z" fill="{FILL}" stroke="{INK}" stroke-width="1.4"/>')
    add(f'<path d="M {cx+11},{cy-9} L {cx+11},{cy+9} L {cx},{cy} Z" fill="{FILL}" stroke="{INK}" stroke-width="1.4"/>')
    add(f'<line x1="{cx}" y1="{cy}" x2="{cx}" y2="{cy-15}" stroke="{INK}" stroke-width="1.3"/>')
    add(f'<path d="M {cx-7},{cy-15} q7,-7 14,0 z" fill="{FILL}" stroke="{INK}" stroke-width="1.3"/>')
    add(f'<text x="{cx+tagdx}" y="{cy+tagdy}" text-anchor="middle" font-size="8.5" font-weight="700" fill="{INK}">{tag}</text>')


def vessel(x, y, w, h, name, sub=""):
    rx, ry = w / 2, 9
    cx = x + rx
    add(f'<path d="M {x},{y+ry} A {rx},{ry} 0 0 1 {x+w},{y+ry} L {x+w},{y+h-ry} '
        f'A {rx},{ry} 0 0 1 {x},{y+h-ry} Z" fill="{FILL}" stroke="{INK}" stroke-width="1.8"/>')
    add(f'<ellipse cx="{cx}" cy="{y+ry}" rx="{rx}" ry="{ry}" fill="{FILL}" stroke="{INK}" stroke-width="1.8"/>')
    if name:
        add(f'<text x="{cx}" y="{y+h-14}" text-anchor="middle" font-size="11" font-weight="800" fill="{INK}">{name}</text>')
    if sub:
        add(f'<text x="{cx}" y="{y+h-2}" text-anchor="middle" font-size="8.5" fill="{INK}">{sub}</text>')


def column(x, y, w, h, name, sub=""):
    cx = x + w / 2
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="13" fill="{FILL}" stroke="{INK}" stroke-width="1.8"/>')
    for k in range(1, 5):
        ty = y + h * k / 5
        add(f'<line x1="{x+6}" y1="{ty}" x2="{x+w-6}" y2="{ty}" stroke="{INK}" stroke-width="0.8" opacity="0.5"/>')
    add(f'<text x="{cx}" y="{y+h/2}" text-anchor="middle" font-size="11" font-weight="800" fill="{INK}" '
        f'transform="rotate(-90 {cx},{y+h/2})">{name}</text>')
    if sub:
        add(f'<text x="{cx}" y="{y+h+13}" text-anchor="middle" font-size="8.5" fill="{INK}">{sub}</text>')


def hx(x, y, w, h, name):
    cx = x + w / 2
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" fill="{FILL}" stroke="{INK}" stroke-width="1.8"/>')
    add(f'<path d="M {x+8},{y+h/2} l 12,-{h/3} l 20,{2*h/3} l 20,-{2*h/3} l 20,{2*h/3} l 12,-{h/3}" '
        f'fill="none" stroke="{INK}" stroke-width="1" opacity="0.55"/>')
    add(f'<text x="{cx}" y="{y+h+13}" text-anchor="middle" font-size="10.5" font-weight="800" fill="{INK}">{name}</text>')


def compressor(cx, cy, r, tag):
    add(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{FILL}" stroke="{INK}" stroke-width="1.8"/>')
    add(f'<path d="M {cx-r},{cy-r*0.7} L {cx+r},{cy-r*0.35} M {cx-r},{cy+r*0.7} L {cx+r},{cy+r*0.35}" '
        f'stroke="{INK}" stroke-width="1.2"/>')
    add(f'<text x="{cx}" y="{cy+r+12}" text-anchor="middle" font-size="9.5" font-weight="700" fill="{INK}">{tag}</text>')


def coil(cx, y0, y1, loops=8, r=9):
    step = (y1 - y0) / loops
    d = f"M {cx},{y0}"
    for i in range(loops):
        yy = y0 + step * (i + 1)
        sw = 1 if i % 2 == 0 else 0
        d += f" A {r},{step/2:.1f} 0 0 {sw} {cx},{yy:.1f}"
    add(f'<path d="{d}" fill="none" stroke="{INK}" stroke-width="1.4" opacity="0.7"/>')


def stream_no(x, y, n):
    add(f'<polygon points="{x},{y-10} {x+10},{y} {x},{y+10} {x-10},{y}" fill="{FILL}" stroke="{INK}" stroke-width="1.1"/>')
    add(f'<text x="{x}" y="{y+3.3}" text-anchor="middle" font-size="8.5" font-weight="800" fill="{INK}">{n}</text>')


def feed(x, y, text):
    add(f'<text x="{x}" y="{y+4}" text-anchor="end" font-size="10" font-weight="700" fill="{INK}">{text}</text>')


def label(x, y, text, size=9.5, weight=400, anchor="middle", color=INK):
    add(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" font-weight="{weight}" fill="{color}">{text}</text>')


# ── 캔버스 ────────────────────────────────────────────────────────────
W, H = 1320, 780
add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">')
add('<defs><marker id="arw" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto" '
    f'markerUnits="strokeWidth"><path d="M0,0 L7,3 L0,6 Z" fill="{INK}"/></marker></defs>')
add(f'<rect x="0" y="0" width="{W}" height="{H}" fill="{FILL}"/>')
label(58, 30, "TENNESSEE EASTMAN PROCESS — P&amp;ID (계장도)", 14, 800, "start")
label(58, 46, "ISA 계기 loop(측정 XMEAS · 컨트롤러 · 신호선) + 제어밸브(조작 XMV 1~11)", 9, 400, "start", SIG)

# ── 공급 유량 loop A·D·E (stream 1·2·3) — 계단식 배치(겹침 방지) ──────
JX, JY = 300, 380
feeds = [("A 공급", 310, "3", "1", 175),
         ("D 공급", 380, "1", "2", 210),
         ("E 공급", 450, "2", "3", 245)]
for name, fy, xmv, sn, lx in feeds:
    feed(70, fy, name)
    pipe([(78, fy), (300, fy)], arrow=False)
    cvalve(120, fy, f"XMV{xmv}")
    loop(lx, fy, "FT", "FC", sn, (120, fy), up=True)
    stream_no(284, fy - 14, sn)
pipe([(300, 310), (300, 450)], arrow=False)

# ── 반응기 피드(stream 6): 믹싱 + 조성분석기 ─────────────────────────
add(f'<circle cx="{JX}" cy="{JY}" r="3.4" fill="{INK}"/>')
pipe([(JX, JY), (384, JY)])
stream_no(322, JY - 14, "6")
instr(346, 318, "AT", "6")          # 반응기 피드 조성 분석기
sig([(346, 335), (346, JY)])

# ── 반응기 (내부 냉각코일 + 온도 loop) ───────────────────────────────
RX, RW = 384, 96
vessel(RX, 210, RW, 250, "반응기", "Reactor")
RCX = RX + RW / 2
coil(RCX + 24, 236, 442, loops=9, r=9)
# 압력 loop (PT→PC)
instr(RX + 18, 232, "PI", "7")
# 액위 loop
instr(RX + RW + 22, 300, "LT", "8"); sig([(RX + RW, 300), (RX + RW + 6, 300)])
instr(RX + RW + 22, 336, "LC", "8", dcs=True)
# 온도 loop → 냉각수 밸브 XMV10
instr(RX + RW + 22, 392, "TT", "9"); sig([(RX + RW, 392), (RX + RW + 6, 392)])
instr(RX + RW + 22, 428, "TC", "9", dcs=True)
cvalve(346, 500, "XMV10")
sig([(RX + RW + 22, 444), (RX + RW + 22, 470), (346, 470), (346, 485)])
pipe([(330, 500), (RCX + 24, 442)], w=1.6)
label(300, 512, "냉각수", 8.5, 700, "end")
instr(330, 250, "TI", "21"); sig([(330, 233), (RCX + 24, 240)])   # CW 출구온도

# ── 반응기 → 응축기 (stream 7) ───────────────────────────────────────
pipe([(RCX, 210), (RCX, 158), (548, 158)])
stream_no(505, 146, "7")

# ── 응축기 + 냉각수 온도 loop ────────────────────────────────────────
hx(548, 136, 128, 44, "응축기 Condenser")
cvalve(608, 104, "XMV11")
pipe([(608, 114), (608, 136)], w=1.6)
instr(656, 104, "TC", "22", dcs=True); sig([(656, 121), (630, 128)])
sig([(656, 87), (656, 76), (608, 76), (608, 89)])

# ── 응축기 → 분리기 ──────────────────────────────────────────────────
pipe([(676, 158), (748, 158), (748, 210)])

# ── 기액분리기 + 액위 loop ───────────────────────────────────────────
SX, SW = 704, 92
vessel(SX, 210, SW, 200, "", "")
SCX = SX + SW / 2
label(SCX, 300, "기액분리기", 11, 800)
label(SCX, 314, "Separator", 8.5, 400)
instr(SX + 18, 234, "PI", "13")
instr(SX + SW + 22, 260, "TI", "11"); sig([(SX + SW, 260), (SX + SW + 6, 260)])
instr(SX + SW + 22, 356, "LT", "12"); sig([(SX + SW, 356), (SX + SW + 6, 356)])
instr(SX + SW + 22, 392, "LC", "12", dcs=True)

# ── 분리기 증기 → 압축기 → 재순환/퍼지 ───────────────────────────────
KX, KY = SCX, 116
pipe([(SCX, 210), (SCX, KY + 22)], arrow=False)
compressor(KX, KY, 22, "압축기 K-1")
instr(KX + 46, KY, "JI", "20")
# 퍼지(9): XMV6 + 분석기 AT
pipe([(KX, KY - 22), (KX, 54)], arrow=False)
cvalve(KX + 56, 54, "XMV6")
pipe([(KX, 54), (KX + 40, 54)], arrow=False)
instr(KX + 96, 54, "AT", "9"); sig([(KX + 96, 71), (KX + 80, 60)])
pipe([(KX + 113, 54), (KX + 165, 54)])
label(KX + 173, 58, "퍼지 Purge", 9.5, 700, "start")
stream_no(KX + 26, 42, "9")
sig([(KX + 96, 37), (KX + 96, 28), (KX + 56, 28), (KX + 56, 39)])   # AT→XMV6
# 재순환 가스(8)
cvalve(KX - 62, KY, "XMV5", tagdx=20, tagdy=3)
pipe([(KX - 22, KY), (KX - 51, KY)], arrow=False)
pipe([(KX - 73, KY), (JX, KY), (JX, JY)], arrow=False)
stream_no(372, KY, "8")

# ── 분리기 액 → 스트리퍼 (stream 10) + 밸브(액위제어 XMV7) ───────────
pipe([(SCX, 410), (SCX, 470), (846, 470)], arrow=False)
instr(792, 440, "FT", "14"); sig([(792, 470), (792, 457)])
cvalve(862, 470, "XMV7")
sig([(SX + SW + 22, 408), (SX + SW + 22, 452), (862, 452), (862, 455)])  # LC→XMV7
pipe([(878, 470), (946, 470), (946, 340)])
stream_no(824, 458, "10")

# ── 공급 A·B·C (stream 4) → 스트리퍼 하부 ────────────────────────────
feed(560, 690, "A·B·C 공급")
pipe([(568, 690), (760, 690)], arrow=False)
cvalve(636, 690, "XMV4")
loop(700, 690, "FT", "FC", "4", (636, 690), up=False)
pipe([(760, 690), (906, 690), (906, 452)])
stream_no(824, 678, "4")
pipe([(906, 452), (948, 452), (948, 340)], arrow=False)

# ── 스트리퍼 + 액위/스팀 loop ────────────────────────────────────────
STX, STW = 950, 82
column(STX, 230, STW, 262, "스트리퍼", "Stripper")
STCX = STX + STW / 2
instr(STX + STW + 22, 266, "PI", "16"); sig([(STX + STW, 266), (STX + STW + 6, 266)])
instr(STX + STW + 22, 342, "TI", "18"); sig([(STX + STW, 342), (STX + STW + 6, 342)])
instr(STX + STW + 22, 430, "LT", "15"); sig([(STX + STW, 430), (STX + STW + 6, 430)])
instr(STX + STW + 22, 466, "LC", "15", dcs=True)
# 스팀 loop(FT→FC→XMV9)
cvalve(STCX, 528, "XMV9", tagdx=22, tagdy=3)
pipe([(STCX, 516), (STCX, 492)], w=1.6)
instr(STCX - 46, 528, "FT", "19"); sig([(STCX - 29, 528), (STCX - 11, 528)])
label(STCX - 70, 532, "스팀", 8.5, 700, "end")

# ── 오버헤드 재순환(stream 5) → 반응기 피드 ──────────────────────────
pipe([(STCX, 230), (STCX, 74), (JX + 12, 74)], arrow=False)
pipe([(JX + 12, 74), (JX, 74), (JX, KY)], arrow=False)
stream_no(STCX - 58, 74, "5")

# ── 스트리퍼 바닥 → 제품 (stream 11) + 액위제어 XMV8 ─────────────────
pipe([(STX + STW, 476), (1044, 476)], arrow=False)
instr(1022, 442, "FT", "17"); sig([(1022, 476), (1022, 459)])
cvalve(1064, 476, "XMV8")
sig([(STX + STW + 22, 482), (STX + STW + 22, 500), (1064, 500), (1064, 491)])  # LC→XMV8
pipe([(1080, 476), (1112, 476)], arrow=False)
instr(1144, 476, "AT", "11"); sig([(1144, 493), (1128, 483)])
pipe([(1161, 476), (1198, 476)], arrow=False)
add(f'<rect x="1150" y="510" width="94" height="32" rx="6" fill="#e3f3ea" stroke="{PROD}" stroke-width="1.8"/>')
add(f'<text x="1197" y="530" text-anchor="middle" font-size="10.5" font-weight="800" fill="{INK}">제품 G · H</text>')
pipe([(1197, 476), (1197, 510)])
stream_no(1096, 464, "11")

# ── 범례 ──────────────────────────────────────────────────────────────
ly = 724
add(f'<line x1="58" y1="{ly}" x2="{W-58}" y2="{ly}" stroke="#dfe4ea" stroke-width="1"/>')
instr(78, ly + 22, "FT", "", r=11); label(96, ly + 26, "필드 계기(측정)", 9, 400, "start")
instr(250, ly + 22, "FC", "", dcs=True, r=11); label(268, ly + 26, "컨트롤러(가로줄=공유표시)", 9, 400, "start")
add(f'<path d="M 470,{ly+15} L 470,{ly+29} L 479,{ly+22} Z M 488,{ly+15} L 488,{ly+29} L 479,{ly+22} Z" fill="{FILL}" stroke="{INK}" stroke-width="1.3"/>')
label(498, ly + 26, "제어밸브 XMV", 9, 400, "start")
add(f'<line x1="640" y1="{ly+22}" x2="676" y2="{ly+22}" stroke="{SIG}" stroke-width="1" stroke-dasharray="4,3"/>')
label(682, ly + 26, "계기 신호선", 9, 400, "start")
add(f'<polygon points="820,{ly+13} 830,{ly+22} 820,{ly+31} 810,{ly+22}" fill="{FILL}" stroke="{INK}" stroke-width="1.1"/>')
label(838, ly + 26, "공정 스트림 번호", 9, 400, "start")

add("</svg>")
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(_p), encoding="utf-8")
print(f"생성: {OUT} ({OUT.stat().st_size/1024:.1f} KB)")
