import math


def generate(t, audio, settings):
    a = 3 + int(audio["lows"] * 4)
    b = 2 + int(audio["high_mids"] * 4)
    delta = t * 0.7
    points = []
    samples = 800
    amp = 0.8
    for i in range(samples + 1):
        p = (i / samples) * math.pi * 2
        x = math.sin(a * p + delta) * amp
        y = math.sin(b * p) * amp
        points.append((x, y))
    return [points]
