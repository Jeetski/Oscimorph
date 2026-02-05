import math


def generate(t, audio, settings):
    c_re = -0.8 + 0.3 * math.cos(t * 0.2)
    c_im = 0.156 + 0.3 * math.sin(t * 0.2)
    zoom = math.exp(t * 0.08)
    points = []
    detail = min(3.0, 1.0 + t * 0.05)
    width = int(120 * detail)
    height = int(120 * detail)
    scale = 1.6 / zoom
    threshold = int(16 + audio["highs"] * 32 + t * 2.5)
    for iy in range(height):
        y = (iy / (height - 1)) * 2 - 1
        for ix in range(width):
            x = (ix / (width - 1)) * 2 - 1
            zx, zy = x * scale, y * scale
            it = 0
            while zx * zx + zy * zy < 4 and it < threshold:
                zx, zy = zx * zx - zy * zy + c_re, 2 * zx * zy + c_im
                it += 1
            if it == threshold:
                points.append((x, y))
    return [points]
