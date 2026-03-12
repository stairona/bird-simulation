# bird_sim.py
# Bird–Wind Turbine Risk Simulator (DEMO / NOT real Isabella County turbine data)
# Shows how season + migration corridors can change collision risk.

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# -----------------------------
# World (normalized map coordinates)
# -----------------------------
W, H = 100.0, 100.0

# Two example migration corridors (centerline + width).
corridors = [
    {"name": "NW→SE corridor", "p0": np.array([10.0, 90.0]), "p1": np.array([90.0, 10.0]), "sigma": 10.0},
    {"name": "W→E corridor",   "p0": np.array([ 0.0, 55.0]), "p1": np.array([100.0, 55.0]), "sigma":  7.0},
]

def dist_point_to_segment(P, A, B):
    AB = B - A
    denom = np.dot(AB, AB) + 1e-12
    t = np.dot(P - A, AB) / denom
    t = np.clip(t, 0.0, 1.0)
    proj = A + t * AB
    return np.linalg.norm(P - proj)

# -----------------------------
# Synthetic turbines (demo layout)
# -----------------------------
N_TURBINES = 35
turbines = []
for i in range(N_TURBINES):
    if i < int(0.65 * N_TURBINES):
        # Cluster near corridor 1 (to demonstrate overlap)
        t = rng.uniform(0.2, 0.8)
        base = corridors[0]["p0"] * (1 - t) + corridors[0]["p1"] * t
        jitter = rng.normal(0, 7, size=2)
        pos = np.clip(base + jitter, [0, 0], [W, H])
    else:
        pos = rng.uniform([0, 0], [W, H])
    turbines.append(pos)
turbines = np.array(turbines)

# -----------------------------
# Season parameters (demo)
# -----------------------------
SEASONS = [
    {"name": "Winter", "months": [12, 1, 2],  "migration_intensity": 0.20, "resident_fraction": 0.85, "night_fraction": 0.45, "weather_risk": 1.10},
    {"name": "Spring", "months": [3, 4, 5],   "migration_intensity": 1.00, "resident_fraction": 0.45, "night_fraction": 0.60, "weather_risk": 1.20},
    {"name": "Summer", "months": [6, 7, 8],   "migration_intensity": 0.25, "resident_fraction": 0.75, "night_fraction": 0.40, "weather_risk": 1.00},
    {"name": "Fall",   "months": [9, 10, 11], "migration_intensity": 0.90, "resident_fraction": 0.50, "night_fraction": 0.65, "weather_risk": 1.25},
]
month_to_season = {}
for s in SEASONS:
    for m in s["months"]:
        month_to_season[m] = s

# -----------------------------
# Simulation controls (tune these)
# -----------------------------
DAYS = 365
START_MONTH = 1

BIRDS_PER_DAY_BASE = 600
MIGRANT_SPEED = 2.4
RESIDENT_SPEED = 1.2
STEPS_PER_DAY = 18

# Collision knobs (demo)
ROTOR_RADIUS = 2.2
BASE_STRIKE_PROB = 0.0025
AVOIDANCE = 0.80               # higher = fewer collisions
NIGHT_RISK_MULT = 1.5
ALTITUDE_MATCH_PROB = 0.55

# -----------------------------
# Bird spawning
# -----------------------------
def sample_entry_exit_for_corridor(c):
    p0, p1 = c["p0"], c["p1"]
    direction = p1 - p0
    direction = direction / (np.linalg.norm(direction) + 1e-12)

    perp = np.array([-direction[1], direction[0]])
    lateral = rng.normal(0, c["sigma"])

    start = p0 + perp * lateral + rng.normal(0, 1.5, size=2)
    end   = p1 + perp * rng.normal(0, c["sigma"]) + rng.normal(0, 1.5, size=2)

    start = np.clip(start, [0, 0], [W, H])
    end   = np.clip(end,   [0, 0], [W, H])
    return start, end

def spawn_migrants(n):
    birds = []
    for _ in range(n):
        c = corridors[rng.integers(0, len(corridors))]
        start, end = sample_entry_exit_for_corridor(c)
        vec = end - start
        vec = vec / (np.linalg.norm(vec) + 1e-12)
        speed = MIGRANT_SPEED * rng.uniform(0.75, 1.25)
        birds.append({"pos": start, "vel": vec * speed, "type": "migrant", "alive": True})
    return birds

def spawn_residents(n):
    birds = []
    for _ in range(n):
        pos = rng.uniform([0, 0], [W, H])
        ang = rng.uniform(0, 2 * np.pi)
        speed = RESIDENT_SPEED * rng.uniform(0.6, 1.4)
        vel = np.array([np.cos(ang), np.sin(ang)]) * speed
        birds.append({"pos": pos, "vel": vel, "type": "resident", "alive": True})
    return birds

# -----------------------------
# Collision probability
# -----------------------------
def per_step_collision_prob(season, is_night, inside_risk):
    if not inside_risk:
        return 0.0
    p = BASE_STRIKE_PROB
    p *= season["weather_risk"]
    if is_night:
        p *= NIGHT_RISK_MULT
    p *= ALTITUDE_MATCH_PROB
    p *= (1.0 - AVOIDANCE)
    return float(np.clip(p, 0.0, 0.25))

# -----------------------------
# Simulation
# -----------------------------
def simulate():
    month_lengths = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
    month = START_MONTH
    day_in_month = 1

    daily_deaths = np.zeros(DAYS, dtype=int)
    daily_birds = np.zeros(DAYS, dtype=int)

    # Heatmap for turbine strikes
    grid_n = 60
    heat = np.zeros((grid_n, grid_n), dtype=float)

    for day in range(DAYS):
        season = month_to_season[month]

        birds_today = int(BIRDS_PER_DAY_BASE * (0.35 + 0.65 * season["migration_intensity"]) * rng.uniform(0.85, 1.15))
        daily_birds[day] = birds_today

        n_res = int(birds_today * season["resident_fraction"])
        n_mig = birds_today - n_res

        birds = spawn_residents(n_res) + spawn_migrants(n_mig)

        deaths = 0

        for _step in range(STEPS_PER_DAY):
            is_night = (rng.random() < season["night_fraction"])

            for b in birds:
                if not b["alive"]:
                    continue

                # Move
                b["pos"] = b["pos"] + b["vel"]

                x, y = b["pos"]

                # Border handling
                if b["type"] == "resident":
                    if x < 0 or x > W:
                        b["vel"][0] *= -1
                        b["pos"][0] = np.clip(b["pos"][0], 0, W)
                    if y < 0 or y > H:
                        b["vel"][1] *= -1
                        b["pos"][1] = np.clip(b["pos"][1], 0, H)
                else:
                    # Migrants leaving the map are removed from further simulation
                    if (x < -5) or (x > W + 5) or (y < -5) or (y > H + 5):
                        b["alive"] = False
                        continue

                # Turbine proximity (closest turbine)
                d = np.linalg.norm(turbines - b["pos"], axis=1)
                min_i = int(np.argmin(d))
                inside = (d[min_i] <= ROTOR_RADIUS)

                # Collision draw
                p_col = per_step_collision_prob(season, is_night, inside)
                if rng.random() < p_col:
                    deaths += 1
                    b["alive"] = False

                    gx = int(np.clip((turbines[min_i, 0] / W) * (grid_n - 1), 0, grid_n - 1))
                    gy = int(np.clip((turbines[min_i, 1] / H) * (grid_n - 1), 0, grid_n - 1))
                    heat[grid_n - 1 - gy, gx] += 1.0

        daily_deaths[day] = deaths

        # advance calendar
        day_in_month += 1
        if day_in_month > month_lengths[month]:
            day_in_month = 1
            month += 1
            if month == 13:
                month = 1

    return daily_birds, daily_deaths, heat

# -----------------------------
# Run + Plot
# -----------------------------
birds, deaths, heat = simulate()

days = np.arange(1, DAYS + 1)
death_rate = deaths / np.maximum(birds, 1)

plt.figure(figsize=(11, 6))
plt.plot(days, deaths, linewidth=1.2)
plt.title("Simulated Daily Bird Fatalities (Demonstration)")
plt.xlabel("Day of Year")
plt.ylabel("Deaths (count)")
plt.tight_layout()

plt.figure(figsize=(11, 6))
plt.plot(days, death_rate, linewidth=1.2)
plt.title("Simulated Daily Fatality Rate (Deaths / Birds per Day)")
plt.xlabel("Day of Year")
plt.ylabel("Rate")
plt.tight_layout()

plt.figure(figsize=(7, 6))
plt.imshow(heat, aspect="auto")
plt.title("Turbine Strike Hotspots (Synthetic Layout)")
plt.xlabel("X (west→east)")
plt.ylabel("Y (south→north)")
plt.tight_layout()

plt.figure(figsize=(7, 7))
plt.scatter(turbines[:, 0], turbines[:, 1], s=18)
for c in corridors:
    plt.plot([c["p0"][0], c["p1"][0]], [c["p0"][1], c["p1"][1]], linewidth=2)
plt.xlim(0, W)
plt.ylim(0, H)
plt.title("Synthetic Turbines + Migration Corridors (Demo Geometry)")
plt.xlabel("X")
plt.ylabel("Y")
plt.tight_layout()

plt.show()

# -----------------------------
# Monthly summary
# -----------------------------
month_lengths_list = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

month_deaths = []
month_birds = []
idx = 0
for ml in month_lengths_list:
    month_deaths.append(int(deaths[idx:idx+ml].sum()))
    month_birds.append(int(birds[idx:idx+ml].sum()))
    idx += ml

print("\nMonthly totals (demo):")
for m in range(12):
    rate = month_deaths[m] / max(month_birds[m], 1)
    print(f"{month_names[m]}: deaths={month_deaths[m]:5d}  birds={month_birds[m]:7d}  rate={rate:.5f}")

print("\nTuning tips:")
print("- Increase Spring/Fall 'migration_intensity' to show bigger seasonal peaks.")
print("- Move corridors or turbine placement to show how overlap drives hotspots.")
print("- Increase AVOIDANCE to show mitigation reducing collisions.")
print("- Increase NIGHT_RISK_MULT and/or night_fraction to show night-migration effects.")
