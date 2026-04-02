
import random
import time
from dataclasses import dataclass, field
from typing import List, Tuple


#  Particle 

@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    life: float          # seconds remaining
    max_life: float
    color: Tuple[int, int, int] = (255, 200, 100)  # BGR – bright cyan-ish

    def update(self, dt: float):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.life -= dt
        self.radius = max(1, self.radius * 0.96)

    @property
    def alpha(self) -> float:
        return max(0.0, self.life / self.max_life)

    @property
    def alive(self) -> bool:
        return self.life > 0


#  Projectile 

@dataclass
class Projectile:
    x: float
    y: float
    vx: float
    vy: float
    radius: float = 28.0
    age: float = 0.0
    particles: List[Particle] = field(default_factory=list)
    _last_particle_time: float = 0.0

    def update(self, dt: float):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.age += dt

        # Spawn trail particles every ~20 ms
        now = time.time()
        if now - self._last_particle_time > 0.02:
            self._spawn_trail_particles()
            self._last_particle_time = now

        # Update existing particles
        for p in self.particles:
            p.update(dt)
        self.particles = [p for p in self.particles if p.alive]

    def _spawn_trail_particles(self):
        for _ in range(3):
            angle_offset = random.uniform(-3.14, 3.14)
            speed = random.uniform(30, 120)
            life = random.uniform(0.15, 0.4)
            color_choices = [
                (255, 220, 140),   # light cyan
                (255, 180, 80),    # blue-white
                (200, 255, 200),   # white-green
                (255, 255, 220),   # bright white
            ]
            self.particles.append(Particle(
                x=self.x + random.uniform(-5, 5),
                y=self.y + random.uniform(-5, 5),
                vx=speed * random.uniform(-1, 1),
                vy=speed * random.uniform(-1, 1),
                radius=random.uniform(2, 6),
                life=life,
                max_life=life,
                color=random.choice(color_choices),
            ))

    def is_off_screen(self, width: int, height: int, margin: int = 80) -> bool:
        return (
            self.x < -margin or self.x > width + margin
            or self.y < -margin or self.y > height + margin
        )


#  Projectile Manager 

class ProjectileManager:
    """Maintains a list of active projectiles and updates them each frame."""

    def __init__(self):
        self.projectiles: List[Projectile] = []

    def spawn(
        self,
        x: float,
        y: float,
        direction: Tuple[float, float],
        speed: float = 900.0,
        radius: float = 28.0,
    ):
        """Create a new projectile at (x, y) moving in the given direction."""
        dx, dy = direction
        # Normalise
        mag = (dx ** 2 + dy ** 2) ** 0.5
        if mag < 1e-6:
            dx, dy = -1.0, 0.0  # default: shoot left (screen coords)
            mag = 1.0
        dx /= mag
        dy /= mag

        proj = Projectile(
            x=x,
            y=y,
            vx=dx * speed,
            vy=dy * speed,
            radius=radius,
        )
        # Spawn burst particles at launch
        for _ in range(20):
            angle = random.uniform(0, 6.28)
            spd = random.uniform(60, 250)
            life = random.uniform(0.2, 0.6)
            proj.particles.append(Particle(
                x=x, y=y,
                vx=spd * random.uniform(-1, 1),
                vy=spd * random.uniform(-1, 1),
                radius=random.uniform(3, 8),
                life=life,
                max_life=life,
                color=random.choice([
                    (255, 255, 200),
                    (255, 200, 100),
                    (200, 255, 255),
                ]),
            ))
        self.projectiles.append(proj)

    def update(self, dt: float, screen_w: int, screen_h: int):
        for proj in self.projectiles:
            proj.update(dt)
        self.projectiles = [
            p for p in self.projectiles if not p.is_off_screen(screen_w, screen_h)
        ]

    @property
    def count(self) -> int:
        return len(self.projectiles)
