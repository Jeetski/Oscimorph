import math


def generate(t, audio, settings):
    sigma = 10.0
    rho = 28.0 + 8.0 * audio["high_mids"]
    beta = 8.0 / 3.0
    dt = 0.005 + 0.003 * audio["subs"]

    x, y, z = 0.1, 0.0, 0.0
    points = []
    for _ in range(1200):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        x += dx * dt
        y += dy * dt
        z += dz * dt
        px = x / 30.0
        py = y / 30.0
        points.append((px, py))
    return [points]
