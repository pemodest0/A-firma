#!/usr/bin/env python3
"""Sistema massa-mola didatico com animacao em tempo real.

Ao rodar, abre uma janela com:
- Esquerda: massa oscilando no sistema massa-mola.
- Meio: grafico x(t) evoluindo.
- Direita: grafico normalizado x(t)/x0 evoluindo.
"""

import math

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle


# Parametros principais (edite aqui para experimentar)
M, K = 1.0, 4.0
X0, V0 = 1.0, 0.0
DT, DUR = 0.02, 20.0
OMEGA_N = math.sqrt(K / M)  # frequencia natural (rad/s)

t, x, v = 0.0, X0, V0
ts, xs = [t], [x]
xn = [x / X0 if abs(X0) > 1e-12 else 0.0]
amp = max(abs(X0) * 1.2, 0.2)
norm_amp = max(1.2, amp / abs(X0)) if abs(X0) > 1e-12 else 2.0

fig, (ax_s, ax_x, ax_n) = plt.subplots(1, 3, figsize=(14, 4.5))
fig.suptitle("Sistema Massa-Mola")

# Painel 1: desenho do sistema
WALL_X, EQ_X = 0.0, 2.0
ax_s.set_xlim(-0.2, EQ_X + amp * 1.6 + 0.8)
ax_s.set_ylim(-1.0, 1.0)
ax_s.set_title("Movimento da massa")
ax_s.set_xlabel("posicao horizontal")
ax_s.set_yticks([])
ax_s.plot([WALL_X, WALL_X], [-0.8, 0.8], "k", lw=4)
ax_s.axvline(EQ_X, ls="--", lw=1, color="gray")
mola, = ax_s.plot([], [], lw=2, color="tab:blue")
massa = Rectangle((0, -0.2), 0.4, 0.4, color="tab:red")
ax_s.add_patch(massa)
label = ax_s.text(0.02, 0.95, "", transform=ax_s.transAxes, va="top")

# Painel 2: x(t)
ax_x.set_xlim(0, DUR)
ax_x.set_ylim(-amp, amp)
ax_x.set_title("Deslocamento x(t)")
ax_x.set_xlabel("tempo (s)")
ax_x.set_ylabel("x (m)")
ax_x.axhline(0, lw=1, color="gray")
linha_x, = ax_x.plot([], [], color="tab:orange")

# Painel 3: x(t)/x0
ax_n.set_xlim(0, DUR)
ax_n.set_ylim(-norm_amp, norm_amp)
ax_n.set_title("Normalizado x(t)/x0")
ax_n.set_xlabel("tempo (s)")
ax_n.set_ylabel("x/x0")
ax_n.axhline(0, lw=1, color="gray")
linha_xn, = ax_n.plot([], [], color="tab:green")


def update(_frame: int):
    global t, x, v
    a = -(K / M) * x
    v += a * DT
    x += v * DT
    t += DT

    x_mass = EQ_X + x
    mola.set_data([WALL_X, x_mass - 0.2], [0, 0])
    massa.set_x(x_mass - 0.2)

    ts.append(t)
    xs.append(x)
    xn.append(x / X0 if abs(X0) > 1e-12 else 0.0)
    linha_x.set_data(ts, xs)
    linha_xn.set_data(ts, xn)
    label.set_text(
        f"t={t:4.1f}s  x={x:6.3f} m  v={v:6.3f} m/s\\n"
        f"omega_n = sqrt(k/m) = {OMEGA_N:5.3f} rad/s"
    )
    return mola, massa, linha_x, linha_xn, label


frames = int(DUR / DT)
ani = FuncAnimation(
    fig,
    update,
    frames=frames,
    interval=max(1, int(DT * 1000)),
    blit=False,
    repeat=False,
    cache_frame_data=False,
)
fig._ani = ani  # Mantem referencia da animacao ate o fim do programa.
plt.tight_layout()
plt.show()
