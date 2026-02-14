import math
import random

_particles = None


def generate(t, audio, settings):
    global _particles
    if _particles is None:
        _particles = []
        for _ in range(120):
            angle = random.random() * math.pi * 2
            radius = random.random() * 0.9
            _particles.append([math.cos(angle) * radius, math.sin(angle) * radius, angle])

    speed = 0.5 + 1.5 * audio["all"]
    swirl = 0.6 + 0.6 * audio["highs"]
    points = []
    for p in _particles:
        p[2] += 0.01 * speed
        r = 0.2 + 0.7 * (0.5 + 0.5 * math.sin(p[2] * swirl))
        p[0] = math.cos(p[2]) * r
        p[1] = math.sin(p[2]) * r
        points.append((p[0], p[1]))
    return [points]
