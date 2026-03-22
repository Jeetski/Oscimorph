import math


def generate(t, audio, settings):
    spread = 0.45 + 0.25 * audio["high_mids"]
    shrink = 0.68 + 0.08 * audio["lows"]
    phase = t * 0.7
    zoom = math.exp(t * 0.05)
    # Add depth only when the zoom has magnified enough to justify another branch generation.
    depth = 6.0 + (math.log(max(zoom, 1e-6)) / max(1e-6, -math.log(shrink)))
    min_screen_length = 0.01

    lines = []

    def branch(x, y, length, angle, d):
        if d <= 0.0 or (length * zoom) < min_screen_length:
            return
        grow = min(1.0, d)
        x2 = x + math.cos(angle) * length * grow
        y2 = y + math.sin(angle) * length * grow
        lines.append([(x, y), (x2, y2)])
        if d <= 1.0:
            return
        branch(x2, y2, length * shrink, angle + spread + math.sin(phase) * 0.1, d - 1)
        branch(x2, y2, length * shrink, angle - spread + math.cos(phase) * 0.1, d - 1)

    branch(0.0, -0.9, 0.6, math.pi / 2, depth)

    if lines:
        top_branch = max(lines, key=lambda line: line[1][1])
        parent_x, parent_y = top_branch[0]
        tip_x, tip_y = top_branch[1]
        focus_x = parent_x * 0.35 + tip_x * 0.65
        focus_y = parent_y * 0.35 + tip_y * 0.65
    else:
        focus_x, focus_y = 0.0, -0.3

    zoomed = []
    for line in lines:
        p1, p2 = line
        x1 = (p1[0] - focus_x) * zoom + focus_x
        y1 = (p1[1] - focus_y) * zoom + focus_y
        x2 = (p2[0] - focus_x) * zoom + focus_x
        y2 = (p2[1] - focus_y) * zoom + focus_y
        zoomed.append([(x1, y1), (x2, y2)])
    return zoomed
