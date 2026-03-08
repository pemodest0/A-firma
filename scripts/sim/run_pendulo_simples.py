#!/usr/bin/env python3
"""Pendulo simples didatico com animacao em tempo real.

Ao rodar, abre uma janela com:
- Esquerda: o pendulo oscilando.
- Direita: grafico do angulo theta(t) sendo desenhado.
"""

import math

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Parametros principais (edite aqui para experimentar)
L, G = 1.0, 9.81
THETA0, OMEGA0 = 0.8, 0.0
DAMPING = 0.03
DT, DUR = 0.02, 20.0

t, theta, omega = 0.0, THETA0, OMEGA0
ts, thetas = [t], [theta]
amp = max(abs(THETA0) * 1.2, 0.2)

fig, (ax_p, ax_g) = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle("Pendulo Simples")

ax_p.set_aspect("equal")
ax_p.set_xlim(-1.2 * L, 1.2 * L)
ax_p.set_ylim(-1.2 * L, 0.2 * L)
ax_p.set_title("Movimento do pendulo")
ax_p.plot(0, 0, "ko", ms=6)
haste, = ax_p.plot([], [], lw=2, color="tab:blue")
bob, = ax_p.plot([], [], "o", ms=12, color="tab:red")

ax_g.set_xlim(0, DUR)
ax_g.set_ylim(-amp, amp)
ax_g.set_title("Angulo ao longo do tempo")
ax_g.set_xlabel("tempo (s)")
ax_g.set_ylabel("theta (rad)")
linha_theta, = ax_g.plot([], [], color="tab:orange")
label = ax_g.text(0.02, 0.95, "", transform=ax_g.transAxes, va="top")


def update(_frame: int):
    global t, theta, omega
    alpha = -(G / L) * math.sin(theta) - DAMPING * omega
    omega += alpha * DT
    theta += omega * DT
    t += DT

    x, y = L * math.sin(theta), -L * math.cos(theta)
    haste.set_data([0, x], [0, y])
    bob.set_data([x], [y])

    ts.append(t)
    thetas.append(theta)
    linha_theta.set_data(ts, thetas)
    label.set_text(f"t={t:4.1f}s  theta={math.degrees(theta):6.2f} graus")
    return haste, bob, linha_theta, label


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
plt.tight_layout()
plt.show()
