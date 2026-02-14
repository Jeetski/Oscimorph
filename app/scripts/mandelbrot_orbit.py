import math


def generate(t, audio, settings):
    # Orbit-trap style path around a moving complex parameter
    cx = -0.1 + 0.6 * math.cos(t * 0.2)
    cy = 0.65 + 0.2 * math.sin(t * 0.3)
    zoom = math.exp(t * 0.08)
    zx, zy = 0.0, 0.0
    points = []
    steps = int(500 + t * 40)
    scale = (0.8 + 0.2 * audio["all"]) * zoom
    for _ in range(steps):
        zx2 = zx * zx - zy * zy + cx
        zy2 = 2 * zx * zy + cy
        zx, zy = zx2, zy2
        points.append((zx * scale, zy * scale))
    return [points]
