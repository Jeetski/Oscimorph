import math


def generate(t, audio, settings):
    k = 4 + int(audio["lows"] * 4)
    amp = 0.85
    points = []
    samples = 700
    phase = t * 0.5
    for i in range(samples + 1):
        p = (i / samples) * math.pi * 2
        r = amp * math.cos(k * p + phase) * (0.7 + 0.3 * audio["highs"])
        x = math.cos(p) * r
        y = math.sin(p) * r
        points.append((x, y))
    return [points]
