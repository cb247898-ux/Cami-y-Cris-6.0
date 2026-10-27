# -*- coding: utf-8 -*-
"""
VoltLab Pro — Simulador educativo de circuitos AC adaptado para Streamlit.

Para ejecutar localmente:
    1. Instalar dependencias: pip install streamlit numpy matplotlib
    2. Ejecutar: streamlit run VoltLab_Pro_Plus_Unidades_Graficas.py
"""

import io
import math
import cmath
import re
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


# ---------- Utilidades de fasores ----------

def fmt_num(value, digits=4):
    if abs(value) < 1e-10:
        value = 0.0
    return f"{value:.{digits}g}"


def polar(z):
    return f"{abs(z):.4f} ∠ {math.degrees(cmath.phase(z)):.3f}°"


def rectangular(z):
    sign = "+" if z.imag >= 0 else "−"
    return f"{z.real:.4f} {sign} j{abs(z.imag):.4f}"


def angle(z):
    return math.degrees(cmath.phase(z))


def angle_phrase(a):
    sign = "+" if a >= 0 else "−"
    return f"{sign} {abs(a):.3f}°"


def parse_quantity(raw, dimension):
    """Acepta prefijos/sufijos SI: 4.7kΩ, 20mH, 100uF, 12V, 60Hz, 1.2rad."""
    text = str(raw).strip().replace("µ", "u").replace("μ", "u")
    text = text.replace("Ω", "ohm").replace("°", "deg").replace(",", ".")
    text = text.replace(" ", "")
    match = re.fullmatch(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)(.*)", text)
    if not match:
        raise ValueError(
            f"Valor no válido: '{raw}'. Ejemplos: 4.7kΩ, 20mH, 100uF, 12V, 60Hz o 1.2rad."
        )
    number, tail = float(match.group(1)), match.group(2)
    if not math.isfinite(number):
        raise ValueError("Introduce valores numéricos finitos.")

    if dimension == "phase":
        low = tail.lower()
        if low in ("", "deg", "degree", "degrees", "grado", "grados"):
            return number
        if low in ("rad", "radian", "radians"):
            return math.degrees(number)
        raise ValueError("La fase acepta grados (deg, °) o radianes (rad).")

    units = {
        "R": ("ohms", "ohm", "Ω"),
        "L": ("henries", "henry", "h"),
        "C": ("farads", "farad", "f"),
        "V": ("volts", "volt", "v"),
        "f": ("hertz", "hz"),
    }
    prefixes = {
        "p": 1e-12, "n": 1e-9, "u": 1e-6, "m": 1e-3,
        "": 1.0, "k": 1e3, "K": 1e3, "M": 1e6, "G": 1e9
    }

    unit_found = None
    lower_tail = tail.lower()
    for unit in sorted(units[dimension], key=len, reverse=True):
        if lower_tail.endswith(unit.lower()):
            unit_found = unit
            prefix = tail[:-len(unit)]
            break
    else:
        prefix = tail

    if prefix not in prefixes:
        raise ValueError(
            f"Unidad/prefijo no reconocido en '{raw}'. Prefijos admitidos: p, n, u/µ, m, k, M y G."
        )
    return number * prefixes[prefix]


def format_eng(value, unit="", sig=4):
    """Devuelve una magnitud con prefijo de ingeniería SI."""
    value = float(value)
    if not math.isfinite(value):
        return f"{value} {unit}".strip()
    if abs(value) < 1e-30:
        return f"0 {unit}".strip()
    prefixes = {-12: "p", -9: "n", -6: "µ", -3: "m", 0: "",
                3: "k", 6: "M", 9: "G", 12: "T"}
    exponent = max(-12, min(12, int(math.floor(math.log10(abs(value)) / 3) * 3)))
    return f"{value / (10 ** exponent):.{sig}g} {prefixes[exponent]}{unit}".strip()


def safe_complex_div(a, b):
    if abs(b) < 1e-12:
        raise ValueError("La división produciría un valor infinito. Revisa los componentes.")
    return a / b


# ---------- Motor de cálculo ----------

def solve_circuit(config):
    topology = config["topology"]
    kind = config["kind"]
    R, L, C = config["R"], config["L"], config["C"]
    Vp, f, phase_deg = config["Vp"], config["f"], config["phase"]
    w = 2 * math.pi * f
    phase_rad = math.radians(phase_deg)
    Vs = Vp * cmath.exp(1j * phase_rad)

    has_R = kind in ("R", "RL", "RC", "RLC")
    has_L = kind in ("L", "RL", "RLC")
    has_C = kind in ("C", "RC", "RLC")
    if has_R and R <= 0:
        raise ValueError("R debe ser mayor que cero para este circuito.")
    if has_L and L <= 0:
        raise ValueError("L debe ser mayor que cero para este circuito.")
    if has_C and C <= 0:
        raise ValueError("C debe ser mayor que cero para este circuito.")

    ZR = complex(R, 0) if has_R else None
    ZL = complex(0, w * L) if has_L else None
    ZC = complex(0, -1 / (w * C)) if has_C else None
    impedances = {}
    if has_R:
        impedances["R"] = ZR
    if has_L:
        impedances["L"] = ZL
    if has_C:
        impedances["C"] = ZC

    if topology == "Serie":
        Zt = sum(impedances.values(), 0j)
        It = safe_complex_div(Vs, Zt)
        voltages = {key: It * z for key, z in impedances.items()}
        currents = {key: It for key in impedances}
        Scomplex = 0.5 * Vs * It.conjugate()
        method_lines = [
            "1) Calculamos las reactancias:",
            f"   ω = 2πf = 2π({f:g}) = {w:.4f} rad/s",
        ]
        if has_L:
            method_lines.append(f"   XL = ωL = {w:.4f} × {L:g} = {w*L:.4f} Ω")
        if has_C:
            method_lines.append(f"   XC = 1/(ωC) = 1/({w:.4f} × {C:g}) = {1/(w*C):.4f} Ω")
        method_lines += [
            "",
            "2) Impedancia total en serie:",
            "   ZT = suma de las impedancias de los elementos",
            f"   ZT = {rectangular(Zt)} Ω",
            f"   |ZT| = {abs(Zt):.4f} Ω",
            f"   θZ = {angle(Zt):.3f}°",
            f"   ZT = {polar(Zt)} Ω",
            "",
            "3) Corriente total usando la ley de Ohm fasorial:",
            "   I = Vs / ZT",
            f"   I = ({polar(Vs)} V) / ({polar(Zt)} Ω)",
            f"   I = {polar(It)} A",
            f"   I rectangular = {rectangular(It)} A",
        ]
    else:
        admittances = {key: safe_complex_div(1+0j, z) for key, z in impedances.items()}
        Yt = sum(admittances.values(), 0j)
        It = Vs * Yt
        Zt = safe_complex_div(1+0j, Yt)
        voltages = {key: Vs for key in impedances}
        currents = {key: safe_complex_div(Vs, z) for key, z in impedances.items()}
        Scomplex = sum((0.5 * voltages[k] * currents[k].conjugate() for k in currents), 0j)
        method_lines = [
            "1) Calculamos las impedancias de cada rama:",
            f"   ω = 2πf = 2π({f:g}) = {w:.4f} rad/s",
        ]
        if has_L:
            method_lines.append(f"   XL = ωL = {w*L:.4f} Ω; ZL = j{w*L:.4f} Ω")
        if has_C:
            method_lines.append(f"   XC = 1/(ωC) = {1/(w*C):.4f} Ω; ZC = −j{1/(w*C):.4f} Ω")
        if has_R:
            method_lines.append(f"   ZR = {R:.4f} Ω")
        method_lines += [
            "",
            "2) Admitancia total en paralelo:",
            "   YT = suma de las admitancias de las ramas, Y = 1/Z",
            f"   YT = {rectangular(Yt)} S",
            f"   |YT| = {abs(Yt):.4f} S; ángulo = {angle(Yt):.3f}°",
            "",
            "3) Impedancia equivalente:",
            "   ZT = 1 / YT",
            f"   ZT = {rectangular(Zt)} Ω",
            f"   ZT = {polar(Zt)} Ω",
            "",
            "4) Corriente total:",
            "   IT = Vs × YT = suma de las corrientes de las ramas",
            f"   IT = {polar(It)} A",
            f"   IT rectangular = {rectangular(It)} A",
        ]

    Vrms = Vp / math.sqrt(2)
    Irms = abs(It) / math.sqrt(2)
    P = Scomplex.real
    Q = Scomplex.imag
    Sapp = abs(Scomplex)
    pf = 0.0 if Sapp < 1e-12 else max(-1.0, min(1.0, P / Sapp))
    phi = angle(Zt)

    method_lines += ["", "5) Voltajes y corrientes de los elementos:"]
    for key in impedances:
        v = voltages[key]
        i = currents[key]
        names = {"R": "Resistencia", "L": "Bobina", "C": "Capacitor"}
        method_lines += [
            f"   {names[key]}:",
            f"      V{key} = {polar(v)} V",
            f"      I{key} = {polar(i)} A",
        ]

    method_lines += [
        "",
        "6) Valores eficaces y potencias:",
        f"   Vs,rms = Vp/√2 = {Vrms:.4f} V",
        f"   IT,rms = |IT|/√2 = {Irms:.4f} A",
        f"   P = potencia activa = {P:.4f} W",
        f"   Q = potencia reactiva = {Q:.4f} var",
        f"   S = potencia aparente = {Sapp:.4f} VA",
        f"   FP = P/S = {pf:.4f}",
    ]

    if abs(Q) < 1e-7:
        interpretation = "Comportamiento predominantemente resistivo: voltaje y corriente están aproximadamente en fase."
    elif Q > 0:
        interpretation = "Comportamiento inductivo: la corriente total se retrasa respecto al voltaje."
    else:
        interpretation = "Comportamiento capacitivo: la corriente total se adelanta respecto al voltaje."
    if has_L and has_C:
        if abs(w * L - 1/(w*C)) / max(w*L, 1/(w*C)) < 0.05:
            interpretation += " El circuito está cerca de la resonancia (XL ≈ XC)."
    method_lines += ["", "7) Interpretación:", "   " + interpretation]

    t = np.linspace(0, 3/f, 1500)
    signals = {"t": t, "Vs": Vp*np.cos(w*t + phase_rad)}
    for key, v in voltages.items():
        signals["V"+key] = abs(v)*np.cos(w*t + cmath.phase(v))
    for key, i in currents.items():
        signals["I"+key] = abs(i)*np.cos(w*t + cmath.phase(i))
    signals["IT"] = abs(It)*np.cos(w*t + cmath.phase(It))

    return {
        **config, "w": w, "Vs_phasor": Vs, "impedances": impedances,
        "Zt": Zt, "It": It, "voltages": voltages, "currents": currents,
        "Vrms": Vrms, "Irms": Irms, "P": P, "Q": Q, "S": Sapp,
        "pf": pf, "phi": phi, "interpretation": interpretation,
        "steps": method_lines, "signals": signals, "has_R": has_R,
        "has_L": has_L, "has_C": has_C
    }


# ---------- Renderizado de Gráficos ----------

def plot_time(r):
    fig, ax_time = plt.subplots(2, 2, figsize=(9, 6), dpi=100)
    axv, axi, axz, axs = ax_time.flat
    sig = r["signals"]
    t_ms = sig["t"] * 1000.0
    palette = {
        "source": "#00a896", "R": "#3678c8", "L": "#d97706",
        "C": "#d94675", "grid": "#dbe3ed", "ink": "#203247",
        "purple": "#7956a8"
    }
    fig.patch.set_facecolor("#f4f7fb")
    for ax in (axv, axi, axz, axs):
        ax.clear()
        ax.set_facecolor("#ffffff")
        ax.grid(True, color=palette["grid"], alpha=0.95, linestyle="--", linewidth=0.7)
        ax.tick_params(colors=palette["ink"], labelsize=8)
        ax.xaxis.label.set_color(palette["ink"])
        ax.yaxis.label.set_color(palette["ink"])
        ax.title.set_color(palette["ink"])
        for sp in ax.spines.values():
            sp.set_color("#b9c6d5")
        ax.axhline(0, color="#7b8794", linewidth=0.7, alpha=0.65)

    axv.plot(t_ms, sig["Vs"], label="Fuente Vs", linewidth=2.5, color=palette["source"])
    for key, color in (("VR", palette["R"]), ("VL", palette["L"]), ("VC", palette["C"])):
        if key in sig:
            axv.plot(t_ms, sig[key], label=key, linewidth=1.7, color=color)
    axv.set_title("Voltajes instantáneos", fontsize=10, fontweight="bold", loc="left")
    axv.set_xlabel("Tiempo (ms)")
    axv.set_ylabel("Voltaje (V)")
    axv.legend(loc="upper right", fontsize=7, ncol=2, frameon=True,
               facecolor="#f8fafc", edgecolor="#dbe3ed")
    axv.margins(x=0.02)

    axi.plot(t_ms, sig["IT"], label="Corriente total", linewidth=2.5, color=palette["source"])
    if r["topology"] == "Paralelo":
        for key, color in (("IR", palette["R"]), ("IL", palette["L"]), ("IC", palette["C"])):
            if key in sig:
                axi.plot(t_ms, sig[key], label=key, linewidth=1.7, color=color)
    axi.set_title("Corrientes instantáneas", fontsize=10, fontweight="bold", loc="left")
    axi.set_xlabel("Tiempo (ms)")
    axi.set_ylabel("Corriente (A)")
    axi.legend(loc="upper right", fontsize=7, frameon=True,
               facecolor="#f8fafc", edgecolor="#dbe3ed")
    axi.margins(x=0.02)

    Z = r["Zt"]
    axz.plot([0, Z.real], [0, Z.imag], marker="o", linewidth=2.8,
             color=palette["purple"], markersize=5)
    axz.plot([Z.real, Z.real], [0, Z.imag], linestyle="--", color="#7b8794", linewidth=1.2)
    axz.plot([0, Z.real], [Z.imag, Z.imag], linestyle="--", color="#7b8794", linewidth=1.2)
    axz.scatter([Z.real], [Z.imag], color=palette["purple"], s=30, zorder=4)
    axz.annotate(f"ZT = {format_eng(abs(Z), 'Ω')}\nθ = {angle(Z):.2f}°",
                 (Z.real, Z.imag), xytext=(8, 8), textcoords="offset points",
                 color=palette["ink"], fontsize=8,
                 bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#dbe3ed"))
    limz = max(abs(Z.real), abs(Z.imag), abs(Z), 1e-3) * 1.4
    axz.set_xlim(min(-0.2*limz, Z.real-0.2*limz), max(1.15*limz, Z.real+0.2*limz))
    axz.set_ylim(min(-1.15*limz, Z.imag-0.2*limz), max(1.15*limz, Z.imag+0.2*limz))
    axz.set_aspect("equal", adjustable="datalim")
    axz.set_title("Triángulo de impedancia", fontsize=10, fontweight="bold", loc="left")
    axz.set_xlabel("Parte real R (Ω)")
    axz.set_ylabel("Parte imaginaria X (Ω)")

    P, Q, S = r["P"], r["Q"], r["S"]
    axs.plot([0, P], [0, 0], linewidth=2.8, color=palette["source"],
             label=f"P = {format_eng(P, 'W')}")
    axs.plot([P, P], [0, Q], linewidth=2.8, color=palette["R"],
             label=f"Q = {format_eng(Q, 'var')}")
    axs.plot([0, P], [0, Q], linewidth=2.8, color=palette["purple"],
             label=f"S = {format_eng(S, 'VA')}")
    axs.scatter([P], [Q], color=palette["purple"], s=30, zorder=4)
    axs.set_title("Triángulo de potencias", fontsize=10, fontweight="bold", loc="left")
    axs.set_xlabel("Potencia activa P (W)")
    axs.set_ylabel("Potencia reactiva Q (var)")
    lim = max(abs(P), abs(Q), S, 1e-3) * 1.35
    axs.set_xlim(min(-0.15*lim, P-0.15*lim), max(1.15*lim, P+0.15*lim))
    axs.set_ylim(min(-1.15*lim, Q-0.15*lim), max(1.15*lim, Q+0.15*lim))
    axs.legend(loc="best", fontsize=7, frameon=True,
               facecolor="#f8fafc", edgecolor="#dbe3ed")

    fig.suptitle(
        f"ANÁLISIS GRÁFICO · {r['kind']} en {r['topology'].lower()} · f = {format_eng(r['f'], 'Hz')}",
        color="#203247", fontsize=12, fontweight="bold"
    )
    fig.tight_layout(pad=2.0, rect=[0, 0, 1, 0.95])
    return fig


def plot_phasors(r):
    fig, ax = plt.subplots(figsize=(7, 5), dpi=95)
    fig.patch.set_facecolor("#f4f7fb")
    ax.set_facecolor("#ffffff")
    ax.axhline(0, color="#8190a3", linewidth=0.9)
    ax.axvline(0, color="#8190a3", linewidth=0.9)
    if r["topology"] == "Serie":
        vectors = [(r["voltages"][k], "V" + k) for k in r["voltages"]]
        vectors.append((r["Vs_phasor"], "Vs total"))
        unit = "V"
        subtitle = "En serie: los voltajes se suman fasorialmente"
    else:
        vectors = [(r["currents"][k], "I" + k) for k in r["currents"]]
        vectors.append((r["It"], "It total"))
        unit = "A"
        subtitle = "En paralelo: las corrientes se suman fasorialmente"
    maxv = max([abs(z) for z, _ in vectors] + [1e-3]) * 1.45
    colors = ["#009688", "#3678c8", "#d97706", "#d94675"]
    for idx, (z, label) in enumerate(vectors):
        color = colors[idx % len(colors)]
        ax.annotate("", xy=(z.real, z.imag), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", lw=2.4, color=color, mutation_scale=14))
        ax.scatter([z.real], [z.imag], color=color, s=22, zorder=3)
        tx = z.real + (0.035 * maxv if z.real >= 0 else -0.035 * maxv)
        ty = z.imag + (0.035 * maxv if z.imag >= 0 else -0.06 * maxv)
        ax.text(tx, ty, f"{label}\n{format_eng(abs(z), unit)} ∠ {angle(z):.1f}°",
                fontsize=8, color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#dbe3ed", alpha=0.92))
    ax.set_xlim(-maxv, maxv)
    ax.set_ylim(-maxv, maxv)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"Parte real ({unit})", color="#203247")
    ax.set_ylabel(f"Parte imaginaria ({unit})", color="#203247")
    ax.set_title("DIAGRAMA FASORIAL\n" + subtitle, color="#203247",
                 fontsize=11, fontweight="bold")
    ax.tick_params(colors="#203247")
    for sp in ax.spines.values():
        sp.set_color("#b9c6d5")
    ax.grid(True, color="#dbe3ed", alpha=0.9, linestyle="--")
    fig.tight_layout()
    return fig


def draw_schematic(r):
    fig, ax = plt.subplots(figsize=(7, 4), dpi=95)
    fig.patch.set_facecolor("#f4f7fb")
    ax.set_facecolor("#ffffff")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")
    wire, ink = "#26384d", "#172b45"
    colors = {"R": "#2878b5", "L": "#0b9b83", "C": "#bd5b91"}
    ax.set_title(f"ESQUEMA DEL CIRCUITO · {r['kind']} EN {r['topology'].upper()}",
                 fontsize=12, fontweight="bold", color=ink, pad=12)
    elements = [k for k in ("R", "L", "C") if k in r["impedances"]]
    labels = {
        "R": f"R = {format_eng(r['R'], 'Ω')}",
        "L": f"L = {format_eng(r['L'], 'H')}",
        "C": f"C = {format_eng(r['C'], 'F')}",
    }

    def line(x1, y1, x2, y2, lw=2.0):
        ax.plot([x1, x2], [y1, y2], color=wire, lw=lw,
                solid_capstyle="round", zorder=1)

    def source(cx, cy, radius=0.43):
        ax.add_patch(plt.Circle((cx, cy), radius, facecolor="#eef6ff",
                                edgecolor="#2878b5", lw=2.1, zorder=3))
        ax.text(cx, cy, "~", ha="center", va="center", fontsize=21,
                color="#2878b5", fontweight="bold")
        ax.text(cx-0.05, cy+radius+0.3, "FUENTE AC", ha="center",
                color=ink, fontsize=8, fontweight="bold")
        ax.text(cx-0.05, cy-radius-0.3, f"{format_eng(r['Vp'], 'V')} pico",
                ha="center", color="#52667c", fontsize=8)

    def component_h(key, x1, x2, y):
        mid = (x1+x2)/2
        color = colors[key]
        line(x1, y, mid-0.48, y)
        line(mid+0.48, y, x2, y)
        if key == "R":
            ax.add_patch(plt.Rectangle((mid-0.48, y-0.22), 0.96, 0.44,
                                       facecolor="#eaf3fb", edgecolor=color, lw=2.2, zorder=3))
        elif key == "L":
            xx = np.linspace(mid-0.48, mid+0.48, 180)
            yy = y + 0.17*np.sin(np.linspace(0, 8*np.pi, 180))
            ax.plot(xx, yy, color=color, lw=2.4, zorder=3)
        else:
            ax.plot([mid-0.12, mid-0.12], [y-0.40, y+0.40], color=color, lw=2.8, zorder=3)
            ax.plot([mid+0.12, mid+0.12], [y-0.40, y+0.40], color=color, lw=2.8, zorder=3)
        ax.text(mid, y+0.55, key, ha="center", color=color, fontsize=10, fontweight="bold")
        ax.text(mid, y-0.58, labels[key], ha="center", color=ink, fontsize=8)

    def component_v(key, x, y_top, y_bottom):
        mid = (y_top+y_bottom)/2
        color = colors[key]
        if key == "R":
            line(x, y_top, x, mid+0.43)
            ax.add_patch(plt.Rectangle((x-0.22, mid-0.43), 0.44, 0.86,
                                       facecolor="#eaf3fb", edgecolor=color, lw=2.2, zorder=3))
            line(x, mid-0.43, x, y_bottom)
        elif key == "L":
            line(x, y_top, x, mid+0.48)
            yy = np.linspace(mid+0.48, mid-0.48, 180)
            xx = x + 0.16*np.sin(np.linspace(0, 8*np.pi, 180))
            ax.plot(xx, yy, color=color, lw=2.4, zorder=3)
            line(x, mid-0.48, x, y_bottom)
        else:
            line(x, y_top, x, mid+0.12)
            ax.plot([x-0.40, x+0.40], [mid+0.12, mid+0.12], color=color, lw=2.8, zorder=3)
            ax.plot([x-0.40, x+0.40], [mid-0.12, mid-0.12], color=color, lw=2.8, zorder=3)
            line(x, mid-0.12, x, y_bottom)
        ax.text(x+0.50, mid+0.18, key, ha="left", color=color, fontsize=10, fontweight="bold")
        ax.text(x+0.50, mid-0.18, labels[key], ha="left", color=ink, fontsize=8)

    if r["topology"] == "Serie":
        y_top, y_bottom, left, right = 4.55, 1.45, 1.35, 10.7
        source(left, (y_top+y_bottom)/2)
        line(left, y_top, left, 4.55)
        line(left, 4.55, 2.15, 4.55)
        line(left, y_bottom, left, 1.45)
        line(left, y_bottom, right, y_bottom)
        line(right, y_bottom, right, y_top)
        if len(elements) == 1:
            spans = [(4.25, 7.75)]
        elif len(elements) == 2:
            spans = [(2.85, 4.95), (6.25, 8.35)]
        else:
            spans = [(2.55, 4.15), (5.05, 6.65), (7.55, 9.15)]
        line(2.15, y_top, spans[0][0], y_top)
        for idx, key in enumerate(elements):
            x1, x2 = spans[idx]
            component_h(key, x1, x2, y_top)
            if idx < len(elements)-1:
                line(x2, y_top, spans[idx+1][0], y_top)
        line(spans[-1][1], y_top, right, y_top)
        ax.text(6, 0.62, "Serie: misma corriente en todos los elementos; los voltajes se suman.",
                ha="center", color="#52667c", fontsize=9)
    else:
        y_top, y_bottom, left, right = 5.15, 1.55, 2.35, 10.15
        source(1.25, (y_top+y_bottom)/2)
        line(1.25, y_top, left, y_top)
        line(left, y_top, right, y_top)
        line(1.25, y_bottom, left, y_bottom)
        line(left, y_bottom, right, y_bottom)
        xs = np.linspace(left+1.1, right-1.1, len(elements)) if len(elements) > 1 else [6.2]
        for x, key in zip(xs, elements):
            line(x, y_top, x, y_top-0.50)
            component_v(key, x, y_top-0.50, y_bottom+0.50)
            line(x, y_bottom+0.50, x, y_bottom)
            ax.add_patch(plt.Circle((x, y_top), 0.055, color=wire, zorder=4))
            ax.add_patch(plt.Circle((x, y_bottom), 0.055, color=wire, zorder=4))
        ax.text(6.2, 0.62, "Paralelo: el voltaje es común en cada rama; las corrientes se suman.",
                ha="center", color="#52667c", fontsize=9)
    ax.text(11.6, 6.55, f"f = {format_eng(r['f'], 'Hz')}", ha="right",
            fontsize=8, color="#52667c")
    ax.text(6, 0.20, "Esquema ideal generado según la selección. No es una netlist SPICE.",
            ha="center", color="#64748b", fontsize=7.5)
    fig.tight_layout(pad=1.3)
    return fig


# ---------- Interfaz Streamlit ----------

st.set_page_config(page_title="VoltLab Pro", page_icon="⚡", layout="wide")

st.title("⚡ VOLTLAB PRO")
st.caption("SIMULACIÓN AC • FASORES • ANÁLISIS • GRÁFICAS")

# Valores por defecto en el estado de sesión
if "input_vals" not in st.session_state:
    st.session_state.input_vals = {
        "topology": "Serie",
        "kind": "RLC",
        "R": "4",
        "L": "0.2",
        "C": "0.0001",
        "Vp": "20",
        "f": "60",
        "phase": "0"
    }

# Panel lateral (Sidebar)
with st.sidebar:
    st.header("⚙️ CONFIGURAR CIRCUITO")
    
    # Presets
    st.subheader("Ejemplos rápidos")
    c_ex1, c_ex2 = st.columns(2)
    if c_ex1.button("Ejemplo RL"):
        st.session_state.input_vals = {
            "topology": "Serie", "kind": "RL", "R": "4", "L": "0.2",
            "C": "0.0001", "Vp": "20", "f": str(377/(2*math.pi)), "phase": "-60"
        }
        st.rerun()
    if c_ex2.button("Ejemplo RC"):
        st.session_state.input_vals = {
            "topology": "Serie", "kind": "RC", "R": "5", "L": "0.2",
            "C": "0.1", "Vp": "10", "f": str(4/(2*math.pi)), "phase": "0"
        }
        st.rerun()

    st.divider()

    topology_opt = ["Serie", "Paralelo"]
    topology = st.selectbox("Conexión", topology_opt, 
                            index=topology_opt.index(st.session_state.input_vals["topology"]))

    kind_opt = ["R", "RL", "RC", "RLC", "L", "C"]
    kind = st.selectbox("Elementos incluidos", kind_opt, 
                        index=kind_opt.index(st.session_state.input_vals["kind"]))

    R_str = st.text_input("Resistencia R (Ω)", st.session_state.input_vals["R"])
    L_str = st.text_input("Inductancia L (H)", st.session_state.input_vals["L"])
    C_str = st.text_input("Capacitancia C (F)", st.session_state.input_vals["C"])
    Vp_str = st.text_input("Voltaje pico Vp (V)", st.session_state.input_vals["Vp"])
    f_str = st.text_input("Frecuencia f (Hz)", st.session_state.input_vals["f"])
    phase_str = st.text_input("Fase fuente (° o rad)", st.session_state.input_vals["phase"])

    # Actualización del estado
    st.session_state.input_vals.update({
        "topology": topology, "kind": kind, "R": R_str, "L": L_str,
        "C": C_str, "Vp": Vp_str, "f": f_str, "phase": phase_str
    })

    st.caption(
        "Acepta prefijos y sufijos: 4.7kΩ, 15mH, 220µF, 20V, 60Hz, -30deg o 1.2rad.\n"
        "Prefijos: p, n, u/µ, m, k, M, G.\n"
        "Puedes omitir la unidad si el campo ya la indica.\n\n"
        "Fasores de pico; potencias RMS."
    )

# Proceso de simulación
try:
    cfg = {
        "topology": topology,
        "kind": kind,
        "R": parse_quantity(R_str, "R"),
        "L": parse_quantity(L_str, "L"),
        "C": parse_quantity(C_str, "C"),
        "Vp": parse_quantity(Vp_str, "V"),
        "f": parse_quantity(f_str, "f"),
        "phase": parse_quantity(phase_str, "phase"),
    }
    if not all(math.isfinite(v) for v in [cfg[k] for k in ("R", "L", "C", "Vp", "f", "phase")]):
        raise ValueError("Usa valores numéricos finitos.")
    if cfg["Vp"] < 0 or cfg["f"] <= 0 or cfg["R"] < 0 or cfg["L"] < 0 or cfg["C"] < 0:
        raise ValueError("V, R, L y C no pueden ser negativos; la frecuencia debe ser positiva.")

    res = solve_circuit(cfg)

    # Mostrar Tarjetas (Metrics)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("IMPEDANCIA TOTAL", f"{abs(res['Zt']):.4g} Ω")
    m2.metric("CORRIENTE PICO", f"{abs(res['It']):.4g} A")
    m3.metric("ÁNGULO Z", f"{res['phi']:.3f}°")
    m4.metric("FACTOR POTENCIA", f"{res['pf']:.4f}")

    # Tabs con las secciones
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📝 Desarrollo paso a paso",
        "📈 Gráficas",
        "📐 Fasores",
        "🔌 Esquema",
        "💡 Interpretación"
    ])

    with tab1:
        heading = [
            "VOLTLAB PRO — RESOLUCIÓN TIPO CUADERNO",
            "=" * 64,
            f"Configuración: circuito {res['kind']} en {res['topology'].lower()}",
            f"Componentes: R={format_eng(res['R'], 'Ω')}, L={format_eng(res['L'], 'H')}, C={format_eng(res['C'], 'F')}",
            f"Fuente: Vs(t) = {res['Vp']:g} cos({res['w']:.4g}t {angle_phrase(res['phase'])}) V",
            "",
        ]
        if res["topology"] == "Serie":
            for key, z in res["impedances"].items():
                heading.append(f"Impedancia de {key}: Z{key} = {rectangular(z)} Ω = {polar(z)} Ω")
        else:
            for key, z in res["impedances"].items():
                heading.append(f"Rama {key}: Z{key} = {rectangular(z)} Ω = {polar(z)} Ω")

        report_text = "\n".join(heading + res["steps"])
        st.code(report_text, language="text")

        st.download_button(
            "💾 Exportar desarrollo (TXT)",
            data=report_text,
            file_name="VoltLab_desarrollo.txt",
            mime="text/plain"
        )

    with tab2:
        st.pyplot(plot_time(res))

    with tab3:
        st.pyplot(plot_phasors(res))

    with tab4:
        st.pyplot(draw_schematic(res))

    with tab5:
        lines = [
            "INTERPRETACIÓN AUTOMÁTICA",
            "=" * 38,
            res["interpretation"],
            "",
            f"• Impedancia equivalente: {polar(res['Zt'])} Ω ({format_eng(abs(res['Zt']), 'Ω')}).",
            f"• Corriente total pico: {polar(res['It'])} A ({format_eng(abs(res['It']), 'A')}).",
            f"• Voltaje RMS de fuente: {format_eng(res['Vrms'], 'V')}.",
            f"• Corriente RMS total: {format_eng(res['Irms'], 'A')}.",
            f"• Potencia activa: {res['P']:.4f} W ({format_eng(res['P'], 'W')}).",
            f"• Potencia reactiva: {res['Q']:.4f} var.",
            f"• Potencia aparente: {res['S']:.4f} VA.",
            f"• Factor de potencia: {res['pf']:.4f}.",
            "",
            "Cómo leer el factor de potencia:",
            "• Cercano a 1: el circuito consume principalmente potencia activa.",
            "• Q positivo: comportamiento inductivo.",
            "• Q negativo: comportamiento capacitivo.",
            "",
            "Importante: simulación ideal en régimen sinusoidal permanente. No modela arranque, conmutación ni componentes semiconductores."
        ]
        st.text("\n".join(lines))

        # Exportación CSV
        csv_buffer = io.StringIO()
        csv_buffer.write("Parámetro;Valor;Unidad\n")
        rows = [
            ("Topología", res["topology"], ""), ("Tipo", res["kind"], ""),
            ("Frecuencia", res["f"], "Hz"), ("Omega", res["w"], "rad/s"),
            ("Impedancia total", rectangular(res["Zt"]), "Ω"),
            ("Módulo de impedancia", abs(res["Zt"]), "Ω"),
            ("Ángulo de Z", res["phi"], "grados"),
            ("Corriente pico", abs(res["It"]), "A"),
            ("Potencia activa", res["P"], "W"),
            ("Potencia reactiva", res["Q"], "var"),
            ("Potencia aparente", res["S"], "VA"),
            ("Factor de potencia", res["pf"], ""),
            ("Interpretación", res["interpretation"], "")
        ]
        for r_item in rows:
            csv_buffer.write(f"{r_item[0]};{r_item[1]};{r_item[2]}\n")

        st.download_button(
            "💾 Exportar datos (CSV)",
            data=csv_buffer.getvalue(),
            file_name="VoltLab_resultados.csv",
            mime="text/csv"
        )

except Exception as e:
    st.error(f"Error al simular: {e}")