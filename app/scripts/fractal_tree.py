import math


def generate(t, audio, settings):
    depth = min(10, 6 + int(t * 0.4))
    spread = 0.45 + 0.25 * audio["high_mids"]
    shrink = 0.68 + 0.08 * audio["lows"]
    phase = t * 0.7
    zoom = math.exp(t * 0.05)
    focus_x = 0.15
    focus_y = 0.45

    lines = []

    def branch(x, y, length, angle, d):
        if d == 0:
            return
        x2 = x + math.cos(angle) * length
        y2 = y + math.sin(angle) * length
        lines.append([(x, y), (x2, y2)])
        branch(x2, y2, length * shrink, angle + spread + math.sin(phase) * 0.1, d - 1)
        branch(x2, y2, length * shrink, angle - spread + math.cos(phase) * 0.1, d - 1)

    branch(0.0, -0.9, 0.6, math.pi / 2, depth)

    zoomed = []
    for line in lines:
        p1, p2 = line
        x1 = (p1[0] - focus_x) * zoom + focus_x
        y1 = (p1[1] - focus_y) * zoom + focus_y
        x2 = (p2[0] - focus_x) * zoom + focus_x
        y2 = (p2[1] - focus_y) * zoom + focus_y
        zoomed.append([(x1, y1), (x2, y2)])
    return zoomed
