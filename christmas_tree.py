

import math
import sys
import time
import numpy as np

# ---------------------------------------------------------------- settings
TITLE = "ASCII Christmas Tree"
COLS, ROWS = 120, 62                 # character grid
FPS = 30
MONO = "consolas,menlo,dejavusansmono,couriernew,monospace"
HINT = "Space pause | R restart | S screenshot | Up/Down speed | Esc quit"

TAU = 2 * math.pi
FLOOR, LOOKY, DIST, PITCH = -1.0, 0.05, 5.5, 0.18     # camera looks slightly down at the tree
CA, SA = math.cos(PITCH), math.sin(PITCH)
ROT_SPEED = math.radians(14)                          # tree spin, radians per second
_L = np.array([-0.55, 0.65, -0.55])
LX, LY, LZ = _L / np.linalg.norm(_L)                  # light from upper left, in front

# timeline (seconds)
T_GROW = 6.0                                          # white ripple builds the tree
T_SPIRAL, SPIRAL_DUR = 6.2, 2.6                       # golden spiral winds up
T_STAR = T_SPIRAL + SPIRAL_DUR                        # star flashes on when the spiral reaches the top
T_BALLS, BALL_GAP = T_STAR + 0.6, 0.07                # red baubles pop in, bottom first
T_GIFT, GIFT_GAP, GIFT_DROP = 12.0, 0.3, 0.7          # gifts drop in and bounce
T_MSG, MSG_RATE = 14.6, 8                             # message types out (letters per second)
RIPPLE_EVERY, RIPPLE_RISE = 4.0, 1.8
T_RIPPLE = T_MSG + 2                                  # afterwards: rings of light keep climbing

MESSAGE = "MERRY CHRISTMAS"
RAMP = np.frombuffer(b".,:;-=+*#%@", dtype=np.uint8)  # dark -> bright
TINTS = np.array([[235, 45, 50], [250, 205, 70], [60, 200, 90]], float)

# tree tiers: (y bottom, y top, radius bottom, radius top)
TIERS = [(-0.82, -0.12, 1.00, 0.16), (-0.42, 0.22, 0.80, 0.12), (-0.08, 0.52, 0.62, 0.09),
         (0.24, 0.80, 0.44, 0.05), (0.52, 1.02, 0.27, 0.0)]
FR = 1.25                                             # radius of the snowy floor

GIFTS = [  # angle, distance, width, height, depth, box colour, ribbon colour
    (0.35, 1.02, .34, .24, .30, (205, 35, 45), (245, 205, 80)),
    (1.45, 0.98, .26, .32, .26, (40, 90, 215), (235, 235, 245)),
    (2.50, 1.05, .38, .20, .28, (225, 185, 55), (200, 30, 40)),
    (3.50, 0.95, .28, .28, .28, (150, 50, 185), (245, 205, 80)),
    (4.50, 1.04, .32, .22, .34, (210, 40, 60), (70, 180, 95)),
    (5.45, 1.00, .24, .36, .24, (30, 145, 80), (245, 60, 60)),
]
LIGHT_COLS = np.array([[255, 215, 90], [255, 240, 175], [255, 185, 55], [255, 255, 235], [255, 200, 120]], float)


# ---------------------------------------------------------------- helpers
def clamp(v, a=0.0, b=1.0):
    return a if v < a else b if v > b else v


def ease_back(x):
    c1 = 1.70158
    return 1 + (c1 + 1) * (x - 1) ** 3 + c1 * (x - 1) ** 2


def ease_bounce(x):
    n, d = 7.5625, 2.75
    if x < 1 / d:
        return n * x * x
    if x < 2 / d:
        x -= 1.5 / d
        return n * x * x + 0.75
    if x < 2.5 / d:
        x -= 2.25 / d
        return n * x * x + 0.9375
    x -= 2.625 / d
    return n * x * x + 0.984375


def envelope(y):
    """Outer radius of the tree at height y."""
    y = np.asarray(y, float)
    r = np.zeros_like(y)
    for y0, y1, rb, rt in TIERS:
        v = rb + (rt - rb) * (y - y0) / (y1 - y0)
        r = np.where((y >= y0) & (y <= y1), np.maximum(r, v), r)
    return r


def spiral_at(u):
    """Golden spiral: the tiers' lower edges all lie on one cone, r = 0.545 * (1.02 - y),
    so the garland rides that cone just outside the branch tips with an even pitch."""
    th = u * TAU * 5.5 + 2.2
    y = -0.74 + u * 1.60
    r = 0.545 * (1.02 - y) + 0.055
    return r * np.cos(th), y, r * np.sin(th), np.cos(th), np.sin(th)


# ---------------------------------------------------------------- scene
class Scene:
    def __init__(self, cw, ch):
        self.cw, self.ch = cw, ch                     # pixel size of one character cell
        W, H = COLS * cw, ROWS * ch
        self.F = min(0.62 * H * DIST / 2.3, 0.9 * W * DIST / 2.7)   # focal length in pixels
        self.FX, self.FY = self.F / cw, self.F / ch                 # ... in cells
        self.CX, self.CY = COLS / 2, ROWS * 0.36
        rng = np.random.default_rng(20261225)
        self._build_points(rng)
        self._build_decorations(rng)
        self.restart()

    # --- geometry -------------------------------------------------------
    def _build_points(self, rng):
        parts = []

        def add(x, y, z, nx, ny, nz, r, g, b, kind, tex, gift=255):
            n = np.size(x)
            f = lambda v: np.broadcast_to(np.asarray(v, np.float32), (n,)).astype(np.float32)
            nx, ny, nz = f(nx), f(ny), f(nz)
            m = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
            parts.append(dict(x=f(x), y=f(y), z=f(z), nx=nx / m, ny=ny / m, nz=nz / m,
                              r=f(r), g=f(g), b=f(b), tex=f(tex),
                              kind=np.full(n, kind, np.uint8), gift=np.full(n, gift, np.uint8)))

        # kind 0: needles, one cone per tier with wavy, drooping edges
        for y0, y1, rb, rt in TIERS:
            nh, nt = 70, round(rb * 380)
            u = np.repeat(np.linspace(0, 1, nh + 1), nt)
            j = np.tile(np.arange(nt), nh + 1)
            n = u.size
            th = (j + rng.random(n) * 0.8) / nt * TAU
            wav = 1 + 0.09 * np.sin(th * 9 + u * 7) * (1 - u * 0.6) + 0.05 * (rng.random(n) - 0.5)
            r = (rb + (rt - rb) * u) * wav
            y = y0 + (y1 - y0) * u - (1 - u) ** 2 * 0.07 * np.abs(np.sin(th * 4.5))
            c, s = np.cos(th), np.sin(th)
            dark = np.where(rng.random(n) < 0.12, 0.6, 1.0)
            add(r * c, y, r * s, c, (rb - rt) / (y1 - y0), s,
                (12 + rng.random(n) * 30) * dark, (90 + rng.random(n) * 75) * dark,
                (28 + rng.random(n) * 30) * dark, 0, 0.7 + rng.random(n) * 0.5)

        # kind 1: trunk
        yy, tt = np.meshgrid(FLOOR + np.linspace(0, 0.2, 17), np.arange(60) / 60 * TAU)
        yy, tt = yy.ravel(), tt.ravel()
        add(0.13 * np.cos(tt), yy, 0.13 * np.sin(tt), np.cos(tt), 0, np.sin(tt),
            110, 68, 32, 1, 0.8 + rng.random(tt.size) * 0.3)

        # kind 3: snowy floor disc, fading at the rim
        for rr in np.arange(0.1, FR + 1e-9, 0.015):
            nt = round(rr * 360)
            th = (np.arange(nt) + rng.random(nt)) / nt * TAU
            fade = 1 - (rr / FR) ** 3
            add(rr * np.cos(th), np.full(nt, FLOOR), rr * np.sin(th), 0, 1, 0,
                175, 185, 220, 3, fade * (0.75 + rng.random(nt) * 0.4))

        # kind 2: gift boxes with a ribbon cross
        st = 0.016
        for gi, (a, d, w, h, dp, col, rib) in enumerate(GIFTS):
            cx, cz, ry = d * math.cos(a), d * math.sin(a), a + 0.6
            c, s = math.cos(ry), math.sin(ry)
            hw, hd = w / 2, dp / 2
            us, vs, ys = np.arange(-hw, hw + 1e-9, st), np.arange(-hd, hd + 1e-9, st), np.arange(0, h + 1e-9, st)

            def face(lx, ly, lz, nx, ny, nz):
                lx, ly, lz = np.broadcast_arrays(lx, ly, lz)
                lx, ly, lz = lx.ravel(), ly.ravel(), lz.ravel()
                is_rib = (np.abs(lx) < w * 0.09) | (np.abs(lz) < dp * 0.09)
                cr = np.where(is_rib[:, None], np.array(rib, float), np.array(col, float))
                add(cx + lx * c - lz * s, FLOOR + ly, cz + lx * s + lz * c,
                    nx * c - nz * s, ny, nx * s + nz * c,
                    cr[:, 0], cr[:, 1], cr[:, 2], 2, 0.9 + rng.random(lx.size) * 0.2, gi)

            U, V = np.meshgrid(us, vs)
            face(U, h, V, 0, 1, 0)                                   # top
            U, Y = np.meshgrid(us, ys)
            face(U, Y, hd, 0, 0, 1)
            face(U, Y, -hd, 0, 0, -1)
            V, Y = np.meshgrid(vs, ys)
            face(hw, Y, V, 1, 0, 0)
            face(-hw, Y, V, -1, 0, 0)

        P = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        self.P = P
        self.P_th = np.arctan2(P["z"], P["x"]).astype(np.float32)
        self.P_rr = np.hypot(P["x"], P["z"]).astype(np.float32)
        self.P_rgb = np.stack([P["r"], P["g"], P["b"]], 1)
        self.gift_idx = [np.flatnonzero(P["gift"] == g) for g in range(len(GIFTS))]

    def _build_decorations(self, rng):
        # baubles: 22 big red, 12 small gold, popping in from the bottom up
        orns = []
        for i in range(34):
            red = i < 22
            y = -0.72 + rng.random() ** 0.8 * 1.5
            th = i * 2.399 + rng.random() * 0.6
            rad = 0.075 + rng.random() * 0.02 if red else 0.045 + rng.random() * 0.015
            r = float(envelope(y)) + rad * 0.45
            orns.append([r * math.cos(th), y, r * math.sin(th), rad,
                         *((215, 25, 35) if red else (245, 195, 60))])
        orns.sort(key=lambda o: o[1])
        self.orns = [(o, i * BALL_GAP + rng.random() * 0.15) for i, o in enumerate(orns)]

        # golden spiral: two strands of tinsel beads
        GN = 2600
        u = np.repeat(np.arange(GN) / GN, 2)
        x, y, z, nx, nz = spiral_at(u)
        self.g_u, self.g_x, self.g_z, self.g_nx, self.g_nz = u, x, z, nx, nz
        self.g_y = y + np.tile([-0.012, 0.012], GN)

        # 170 gold lights strung along the spiral
        self.NL = 170
        u = (np.arange(self.NL) + 0.5) / self.NL
        x, y, z, nx, nz = spiral_at(u)
        self.l_x, self.l_y, self.l_z, self.l_nx, self.l_nz = x * 1.02, y + 0.02, z * 1.02, nx, nz
        self.l_col = LIGHT_COLS[np.arange(self.NL) % len(LIGHT_COLS)]
        self.l_ph = rng.random(self.NL) * TAU
        self.l_sp = 1.5 + rng.random(self.NL) * 2.5

        # background stars (fixed) and snowflakes (moving)
        srng = np.random.default_rng(7)
        nb = COLS * ROWS // 90
        self.bg_x = srng.integers(0, COLS, nb)
        self.bg_y = srng.integers(0, int(ROWS * 0.8), nb)
        self.bg_ph = srng.random(nb) * TAU
        nf = COLS * ROWS // 45
        self.f_L = np.arange(nf) % 3
        self.f_v = np.array([1.5, 3.0, 5.5])[self.f_L] + srng.random(nf) * 2
        self.f_ph = srng.random(nf) * TAU
        self._snow_rng = srng

    def restart(self):
        r = self._snow_rng
        self.f_x = r.random(self.f_L.size) * COLS
        self.f_y = r.random(self.f_L.size) * ROWS

    # --- camera -----------------------------------------------------------
    def project(self, x, y, z, cr, sr):
        X = x * cr - z * sr
        Z = x * sr + z * cr
        Y = y - LOOKY
        yc = Y * CA + Z * SA
        zc = Z * CA - Y * SA + DIST
        return X, yc, zc, self.CX + X * self.FX / zc, self.CY - yc * self.FY / zc

    # --- one frame -----------------------------------------------------------
    def render(self, t, dt):
        """Returns (chars, rgb, info): chars is a COLS*ROWS array of ASCII codes (0 = empty)."""
        P, N = self.P, COLS * ROWS
        rot = t * ROT_SPEED
        cr, sr = math.cos(rot), math.sin(rot)
        growing = t < T_GROW
        gs = clamp(t / T_GROW)
        grow_y = FLOOR - 0.1 + 2.35 * gs * gs * (3 - 2 * gs) if growing else 9.0
        flash = clamp(1 - (t - T_STAR) / 1.2) if t > T_STAR else 0.0

        rip_on, rip_y, rip_amp, rip_ph = False, 0.0, 0.0, 0.0
        if t > T_RIPPLE:
            rip_ph = ((t - T_RIPPLE) % RIPPLE_EVERY) / RIPPLE_RISE
            if rip_ph < 1:
                rip_on = True
                rip_y = FLOOR + rip_ph * 2.15
                rip_amp = 0.9 * math.sin(math.pi * min(1.0, rip_ph * 1.4 + 0.15))
        floor_ring = ((t / 1.5) % 1) * FR * 1.15 if growing else rip_ph * FR * 1.15 if rip_on else -1.0

        cand = []      # (cell, depth, char, rgb, light_glow) candidates for the z-buffer

        # ---- point cloud: needles, trunk, gifts, floor
        x, y, z = P["x"].copy(), P["y"].copy(), P["z"].copy()
        kind = P["kind"]
        tree = kind < 2
        keep = np.ones(x.size, bool)
        band = np.zeros(x.size, np.float32)
        if growing:                                   # rippling white growth front
            front = grow_y + 0.05 * np.sin(3 * self.P_th - 6 * t) + 0.025 * np.sin(7 * self.P_th + 4 * t)
            keep &= ~(tree & (y > front))
            g = (front - y) / 0.11
            band = np.where(tree & (g < 1), 1 - g, 0).astype(np.float32)
        if rip_on:                                    # later ripples climbing the tree
            g = np.abs(y - rip_y - 0.03 * np.sin(3 * self.P_th - 5 * t)) / 0.09
            band = np.where(tree & (g < 1), np.maximum(band, (1 - g) * rip_amp), band)
        swell = 1 + 0.07 * band
        x *= swell
        z *= swell
        for gi, idx in enumerate(self.gift_idx):      # gifts drop in and bounce
            s = clamp((t - T_GIFT - gi * GIFT_GAP) / GIFT_DROP)
            if s <= 0:
                keep[idx] = False
            else:
                y[idx] += (1 - ease_bounce(s)) * 1.6

        ids = np.flatnonzero(keep)
        X, yc, zc, sx, sy = self.project(x[ids], y[ids], z[ids], cr, sr)
        nx0, ny0, nz0 = P["nx"][ids], P["ny"][ids], P["nz"][ids]
        NX = nx0 * cr - nz0 * sr
        NZ0 = nx0 * sr + nz0 * cr
        NY = ny0 * CA + NZ0 * SA
        NZ = NZ0 * CA - ny0 * SA
        ok = (NX * X + NY * yc + NZ * zc <= 0) & (sx >= 0) & (sx < COLS) & (sy >= 0) & (sy < ROWS)
        ids, zc, sx, sy, NX, NY, NZ = ids[ok], zc[ok], sx[ok], sy[ok], NX[ok], NY[ok], NZ[ok]
        k = kind[ids]
        b = (0.16 + 0.9 * np.clip(NX * LX + NY * LY + NZ * LZ, 0, None)) * P["tex"][ids]
        wr = np.zeros(ids.size, np.float32)
        is_floor = k == 3
        b = np.where(is_floor, b * 0.55, b + flash * 0.25)
        if floor_ring > 0:
            g = np.abs(self.P_rr[ids] - floor_ring) / 0.07
            ring = is_floor & (g < 1)
            b = np.where(ring, b + 0.45 * (1 - g), b)
            wr = np.where(ring, 0.4 * (1 - g), wr)
        bd = band[ids]
        b = b + 0.65 * bd
        wr = np.where(bd > 0, 0.75 * bd, wr)
        chars = RAMP[np.minimum((b * 10).astype(int), 10)]
        col = self.P_rgb[ids] * (0.3 + 0.85 * b)[:, None]
        col += (np.array([255, 255, 220]) - col) * wr[:, None]
        cand.append((sy.astype(int) * COLS + sx.astype(int), zc, chars, col, None))

        # ---- golden spiral garland, drawn upward with a glint racing along it
        reach = clamp((t - T_SPIRAL) / SPIRAL_DUR) if t > T_SPIRAL else 0.0
        if t > T_SPIRAL:
            m = self.g_u <= reach
            u = self.g_u[m]
            X, yc, zc, sx, sy = self.project(self.g_x[m], self.g_y[m], self.g_z[m], cr, sr)
            nx0, nz0 = self.g_nx[m], self.g_nz[m]
            NX = nx0 * cr - nz0 * sr
            NZ0 = nx0 * sr + nz0 * cr
            NY, NZ = NZ0 * SA, NZ0 * CA
            ok = (NX * X + NY * yc + NZ * zc <= 0.1) & (sx >= 0) & (sx < COLS) & (sy >= 0) & (sy < ROWS)
            u, zc, sx, sy, NX, NY, NZ = u[ok], zc[ok], sx[ok], sy[ok], NX[ok], NY[ok], NZ[ok]
            d = np.clip(NX * LX + NY * LY + NZ * LZ, 0, None)
            sh = (0.5 + 0.5 * np.sin(u * 90 - t * 7)) ** 6
            tip = np.clip(1 - (reach - u) * 25, 0, 1) if reach < 1 else np.zeros_like(u)
            e = np.minimum(1, 0.35 + 0.6 * d + 0.7 * sh + tip)
            ch = np.select([(sh > 0.6) | (tip > 0.5), e > 0.7, e > 0.45], [ord("*"), ord("="), ord("~")], ord("-"))
            w = (np.maximum(sh, tip) * 0.6)[:, None]
            col = np.stack([255 * e, 195 * e, 60 * e], 1)
            col += (np.array([255, 250, 210]) - col) * w
            cand.append((sy.astype(int) * COLS + sx.astype(int), zc - 0.05, ch.astype(np.uint8), col, None))

        # ---- gold lights along the spiral
        on = min(self.NL, int(reach * self.NL)) if t > T_SPIRAL else 0
        if on:
            kk = np.arange(on)
            X, yc, zc, sx, sy = self.project(self.l_x[:on], self.l_y[:on], self.l_z[:on], cr, sr)
            nx0, nz0 = self.l_nx[:on], self.l_nz[:on]
            NX = nx0 * cr - nz0 * sr
            NZ0 = nx0 * sr + nz0 * cr
            ok = (NX * X + NZ0 * SA * yc + NZ0 * CA * zc <= 0.15) & (sx >= 0) & (sx < COLS) & (sy >= 0) & (sy < ROWS)
            kk, zc, sx, sy = kk[ok], zc[ok], sx[ok], sy[ok]
            tw = (0.5 + 0.5 * np.sin(t * self.l_sp[kk] + self.l_ph[kk])) ** 2
            fresh = np.clip(1 - (reach * self.NL - kk) / 8, 0, 1)      # pop as each switches on
            e = np.maximum(tw, fresh)
            ch = np.select([e > 0.6, e > 0.25], [ord("*"), ord("+")], ord(".")).astype(np.uint8)
            col = self.l_col[kk] * (0.35 + 0.65 * e)[:, None]
            cand.append((sy.astype(int) * COLS + sx.astype(int), zc - 0.08, ch, col, (kk, e)))

        # ---- baubles: shaded spheres that pop in
        for (ox, oy, oz, orad, r0, g0, b0), delay in self.orns:
            ps = clamp((t - T_BALLS - delay) / 0.35)
            if ps <= 0:
                continue
            rad = orad * max(0.05, ease_back(ps))
            _, _, zc, sxc, syc = self.project(np.array([ox]), np.array([oy]), np.array([oz]), cr, sr)
            zc, sxc, syc = float(zc[0]), float(sxc[0]), float(syc[0])
            base = np.array([r0, g0, b0], float)
            pr = rad * self.F / zc                                   # radius in pixels
            if pr < self.cw * 0.8:
                ix, iy = int(math.floor(sxc)), int(math.floor(syc))
                if 0 <= ix < COLS and 0 <= iy < ROWS:
                    cand.append((np.array([iy * COLS + ix]), np.array([zc - rad]),
                                 np.array([ord("o")], np.uint8), base[None, :], None))
                continue
            rx, ry = pr / self.cw, pr / self.ch
            IX, IY = np.meshgrid(np.arange(math.floor(sxc - rx), math.ceil(sxc + rx) + 1),
                                 np.arange(math.floor(syc - ry), math.ceil(syc + ry) + 1))
            IX, IY = IX.ravel(), IY.ravel()
            dx = (IX + 0.5 - sxc) * self.cw / pr
            dy = (IY + 0.5 - syc) * self.ch / pr
            d2 = dx * dx + dy * dy
            m = (d2 <= 1) & (IX >= 0) & (IX < COLS) & (IY >= 0) & (IY < ROWS)
            IX, IY, dx, dy, d2 = IX[m], IY[m], dx[m], dy[m], d2[m]
            nzs = -np.sqrt(1 - d2)
            diff = np.clip(dx * LX - dy * LY + nzs * LZ, 0, None)
            bb = 0.18 + 0.82 * diff
            spec = diff ** 24 > 0.55
            ch = np.select([spec, bb > 0.72, bb > 0.42], [ord("*"), ord("@"), ord("O")], ord("o")).astype(np.uint8)
            col = base * (0.3 + 0.9 * bb)[:, None]
            col[spec] = 255
            cand.append((IY * COLS + IX, zc + nzs * rad, ch, col, None))

        # ---- z-buffer: keep the nearest candidate in every cell
        cells = np.concatenate([c[0] for c in cand])
        depth = np.concatenate([c[1] for c in cand])
        chs = np.concatenate([c[2] for c in cand])
        cols_ = np.concatenate([c[3] for c in cand])
        order = np.lexsort((depth, cells))
        cs = cells[order]
        first = np.ones(cs.size, bool)
        first[1:] = cs[1:] != cs[:-1]
        win = order[first]
        chars_buf = np.zeros(N, np.uint8)
        rgb = np.zeros((N, 3), np.float32)
        chars_buf[cells[win]] = chs[win]
        rgb[cells[win]] = cols_[win]

        # ---- light glow spills onto neighbouring cells
        light_parts = [(i, c) for i, c in enumerate(cand) if c[4] is not None]
        if light_parts:
            i_l, (lc, ld, _, lcol, (kk, e)) = light_parts[0]
            start = sum(c[0].size for c in cand[:i_l])
            won = np.zeros(cells.size, bool)
            won[win] = True
            lw = won[start:start + lc.size]
            lc, lcol, e = lc[lw], self.l_col[kk[lw]], e[lw]
            lx, ly = lc % COLS, lc // COLS
            for ox, oy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                jx, jy = lx + ox, ly + oy
                m = (jx >= 0) & (jx < COLS) & (jy >= 0) & (jy < ROWS)
                j = jy[m] * COLS + jx[m]
                filled = chars_buf[j] != 0
                np.add.at(rgb, j[filled], lcol[m][filled] * (0.45 * e[m][filled])[:, None])

        # ---- star on top
        if t > T_STAR:
            self._draw_star(t, rot, flash, chars_buf, rgb)

        # ---- background stars and snow
        empty = chars_buf == 0
        bi = self.bg_y * COLS + self.bg_x
        m = empty[bi]
        tw = 0.5 + 0.5 * np.sin(t * 1.3 + self.bg_ph[m])
        chars_buf[bi[m]] = np.where(tw > 0.85, ord("+"), ord("."))
        rgb[bi[m]] = np.stack([40 + 110 * tw, 50 + 110 * tw, 90 + 120 * tw], 1)

        self.f_y += self.f_v * dt
        self.f_x += np.sin(t * 0.8 + self.f_ph) * 0.6 * dt * (self.f_L + 1)
        wrap = self.f_y >= ROWS
        self.f_y[wrap] %= ROWS
        self.f_x[wrap] = self._snow_rng.random(wrap.sum()) * COLS
        self.f_x %= COLS
        fi = self.f_y.astype(int) * COLS + self.f_x.astype(int)
        for L, chv, cv in ((0, ".", (90, 100, 130)), (1, None, (160, 170, 200)), (2, "*", (230, 235, 255))):
            m = self.f_L == L
            j = fi[m]
            if L < 2:
                free = chars_buf[j] == 0
                j = j[free]
                ph = self.f_ph[m][free]
            else:
                ph = self.f_ph[m]
            chars_buf[j] = ord(chv) if chv else np.where(ph > 3, ord("+"), ord("."))
            rgb[j] = cv

        # ---- message, just below the front edge of the snow
        letters = 0
        if t > T_MSG:
            letters = min(len(MESSAGE), int((t - T_MSG) * MSG_RATE) + 1)
            fy0 = FLOOR - LOOKY
            fyc, fzc = fy0 * CA - FR * SA, -FR * CA - fy0 * SA + DIST
            row = int(clamp(round(self.CY - fyc * self.FY / fzc) + 1, 1, ROWS - 3))
            x0 = (COLS - (len(MESSAGE) * 2 - 1)) // 2
            xa, xb = max(x0 - 3, 0), min(x0 + letters * 2 + 2, COLS)
            for r in range(row - 1, row + 2):                       # clear a band
                chars_buf[r * COLS + xa:r * COLS + xb] = 0
            for i, c in enumerate(MESSAGE[:letters]):
                if c == " ":
                    continue
                tint = TINTS[(i // 2 + int(t * 1.5)) % 3]
                bri = 0.85 + 0.25 * math.sin(t * 5 + i)
                pop = clamp(1 - ((t - T_MSG) - i / MSG_RATE) / 0.25)
                j = row * COLS + x0 + i * 2
                chars_buf[j] = ord(c)
                rgb[j] = tint * bri + (255 - tint * bri) * pop

        info = dict(rot=math.degrees(rot) % 360, lights=on, letters=letters)
        return chars_buf, np.clip(rgb, 0, 255).astype(np.uint8), info

    def _draw_star(self, t, rot, flash, chars_buf, rgb):
        s = clamp((t - T_STAR) / 0.7)
        sc = max(0.0, ease_back(s))
        y = 1.12 - LOOKY
        yc, zc = y * CA, DIST - y * SA
        sxc, syc = self.CX, self.CY - yc * self.FY / zc
        Ro = 0.15 * self.F / zc * sc                                  # outer radius, pixels
        if Ro <= 1:
            return
        cw, ch = self.cw, self.ch
        wf = 0.3 + 0.7 * abs(math.cos(rot))                           # spins with the tree
        pulse = 0.85 + 0.15 * math.sin(t * 4)
        halo = Ro * (1.7 + flash * 1.6)
        ext = max(halo, Ro) + cw
        IX, IY = np.meshgrid(np.arange(math.floor(sxc - ext / cw), math.ceil(sxc + ext / cw) + 1),
                             np.arange(math.floor(syc - ext / ch), math.ceil(syc + ext / ch) + 1))
        IX, IY = IX.ravel(), IY.ravel()
        m = (IX >= 0) & (IX < COLS) & (IY >= 0) & (IY < ROWS)
        IX, IY = IX[m], IY[m]
        dxp, dyp = (IX + 0.5 - sxc) * cw, (IY + 0.5 - syc) * ch
        dx = dxp / wf
        rr = np.hypot(dx, dyp)
        seg = TAU / 5
        sec = np.mod(np.arctan2(dx, -dyp), seg) - seg / 2
        lim = Ro * (0.45 + 0.55 * (1 - np.abs(sec) / (seg / 2)) ** 1.8)
        j = IY * COLS + IX
        inside = rr < lim
        inner = 1 - rr[inside] / lim[inside]
        w = flash * 0.85
        chars_buf[j[inside]] = np.select([inner > 0.55, inner > 0.3], [ord("@"), ord("#")], ord("*"))
        rgb[j[inside]] = [255 * pulse + 40, (205 + 50 * w) * pulse, (60 + 195 * w) * pulse]
        rh = np.hypot(dxp, dyp)
        hm = (~inside) & (rh < halo) & (chars_buf[j] == 0) & (((IX * 7 + IY * 13 + int(t * 6)) % 4) == 0)
        f = 1 - rh[hm] / halo
        chars_buf[j[hm]] = np.where(f > 0.5, ord(":"), ord("."))
        rgb[j[hm]] = np.stack([255 * f * (0.5 + flash), 200 * f * (0.5 + flash), 80 * f * (0.4 + flash)], 1)
        if flash > 0:                                                  # flash rays
            n = round(Ro * 3 * flash / cw)
            cx, cy = round(sxc), round(syc)
            for k in range(1, n + 1):
                f = (1 - k / (n + 1)) * flash * 255
                for px, py, c in ((cx + k + round(Ro / cw), cy, "-"), (cx - k - round(Ro / cw), cy, "-"),
                                  (cx, math.floor(syc) - k - round(Ro / ch), "|")):
                    if 0 <= px < COLS and 0 <= py < ROWS:
                        chars_buf[py * COLS + px] = ord(c)
                        rgb[py * COLS + px] = (f, f * 0.9, f * 0.5)


# ---------------------------------------------------------------- window
def run_window():
    try:
        import pygame
    except ImportError:
        sys.exit("This script needs pygame and numpy:  pip install pygame numpy")

    pygame.init()
    disp = pygame.display.Info()
    px = max(8, int(disp.current_h * 0.88 / ROWS / 1.15))
    while True:                                                       # biggest font that fits the screen
        font = pygame.font.SysFont(MONO, px)
        cw, ch = font.size("M")
        if (cw * COLS <= disp.current_w * 0.95 and ch * ROWS <= disp.current_h * 0.88) or px <= 7:
            break
        px -= 1
    screen = pygame.display.set_mode((cw * COLS, ch * ROWS))
    pygame.display.set_caption(TITLE)
    hud = pygame.font.SysFont(MONO, max(11, px))
    scene = Scene(cw, ch)
    glyphs = {}                                                       # (char, colour) -> rendered surface
    clock = pygame.time.Clock()
    t, paused, speed, fps = 0.0, False, 1.0, float(FPS)
    last = time.perf_counter()

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                pygame.quit()
                return
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_SPACE:
                    paused = not paused
                elif ev.key == pygame.K_r:
                    t, paused = 0.0, False
                    scene.restart()
                elif ev.key == pygame.K_s:
                    name = time.strftime("christmas_tree_%Y%m%d_%H%M%S.png")
                    pygame.image.save(screen, name)
                    print("saved", name)
                elif ev.key == pygame.K_UP:
                    speed = min(4.0, speed * 2)
                elif ev.key == pygame.K_DOWN:
                    speed = max(0.25, speed / 2)

        now = time.perf_counter()
        real = min(0.1, now - last)
        last = now
        dt = 0.0 if paused else real * speed
        t += dt
        if real > 0:
            fps += (1 / real - fps) * 0.05

        chars, rgb, info = scene.render(t, dt)

        screen.fill((0, 0, 0))
        nz = np.flatnonzero(chars)
        q = (rgb[nz] >> 4).astype(np.int32)                           # 16 levels per channel -> cacheable
        keys = (chars[nz].astype(np.int32) << 12) | (q[:, 0] << 8) | (q[:, 1] << 4) | q[:, 2]
        xs, ys = (nz % COLS) * cw, (nz // COLS) * ch
        batch = []
        for key, x, y in zip(keys.tolist(), xs.tolist(), ys.tolist()):
            surf = glyphs.get(key)
            if surf is None:
                colour = (((key >> 8) & 15) * 17, ((key >> 4) & 15) * 17, (key & 15) * 17)
                surf = glyphs[key] = font.render(chr(key >> 12), True, colour)
            batch.append((surf, (x, y)))
        screen.blits(batch, doreturn=False)

        # HUD
        W = cw * COLS
        lh = hud.get_linesize()
        for i, line in enumerate((f"rotation {info['rot']:6.1f}°", f"lights   {info['lights']}",
                                  f"letters  {info['letters']}/{len(MESSAGE)}")):
            screen.blit(hud.render(line, True, (125, 125, 125)), (10, 10 + i * lh))
        stamp = hud.render(("|| " if paused else "") + f"t {t:5.2f}s", True, (136, 136, 136) if paused else (242, 193, 78))
        screen.blit(stamp, (W - 10 - stamp.get_width(), 10))
        rate = hud.render(f"{fps:4.0f} fps" + (f"  x{speed:g}" if speed != 1 else ""), True, (92, 92, 92))
        screen.blit(rate, (W - 10 - rate.get_width(), 10 + lh))
        hint = hud.render(HINT, True, (70, 70, 70))
        screen.blit(hint, (10, ch * ROWS - lh - 6))

        pygame.display.flip()
        clock.tick(FPS)


def print_frame(t):
    """Plain-text dump of one frame (no pygame needed)."""
    scene = Scene(7, 14)
    chars, _, info = scene.render(t, 1 / FPS)
    grid = chars.reshape(ROWS, COLS)
    print("\n".join("".join(chr(c) if c else " " for c in row).rstrip() for row in grid))
    print(info)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--print":
        print_frame(float(sys.argv[2]))
    else:
        run_window()
