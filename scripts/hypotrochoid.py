import math


def generate(t, audio, settings):
    R = 0.75
    r = 0.2 + 0.15 * audio["subs"]
    d = 0.3 + 0.4 * audio["highs"]
    k = (R - r) / r
    points = []
    samples = 900
    phase = t * 0.4
    for i in range(samples + 1):
        p = (i / samples) * math.pi * 2 * 5
        x = (R - r) * math.cos(p + phase) + d * math.cos(k * (p + phase))
        y = (R - r) * math.sin(p + phase) - d * math.sin(k * (p + phase))
        points.append((x, y))
    return [points]
