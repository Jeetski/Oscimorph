import math


def generate(t, audio, settings):
    order = 10 + int(audio["lows"] * 4) + int(t * 0.5)
    seq = "L"
    for _ in range(order):
        seq = seq + "L" + "".join("R" if c == "L" else "L" for c in reversed(seq))

    x, y = 0.0, 0.0
    angle = t * 0.5
    zoom = math.exp(t * 0.06)
    step = (0.03 + 0.01 * audio["highs"]) * zoom
    points = [(x, y)]
    for turn in seq:
        if turn == "L":
            angle += math.pi / 2
        else:
            angle -= math.pi / 2
        x += math.cos(angle) * step
        y += math.sin(angle) * step
        points.append((x, y))
    return [points]
