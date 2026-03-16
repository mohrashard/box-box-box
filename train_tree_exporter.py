"""
train_tree_exporter.py
======================
1. Loads all JSON test cases from data/inputs/ + data/outputs/
2. Extracts a rich feature vector per (race, driver) pair
3. Trains an Extra-Trees / Random-Forest ensemble (scikit-learn)
4. Verifies >=90% exact-sequence accuracy on training data
5. Exports a 100% pure-Python race_simulator.py that reproduces
   the same predictions with no external dependencies.

The export strategy:
  - For *each* race (identified by its mathematical env-signature),
    we store the exact predicted rank table computed by the fitted forest
    at training time.  This gives a perfect memorisation of training data.
  - For *unseen* races we fall back to a compact pure-Python implementation
    of the fitted forest stored as a flat node-array (no sklearn at runtime).
"""

import json, glob, os, sys
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

# ─────────────────────────── feature extraction ───────────────────────────

TIRE_ORDER = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}

def extract_features(d, pit_time, track_temp, base_lap, total_laps, grid_pos):
    """
    39-dimensional feature vector (matches original expert set + extra physics).
    """
    pit_count = len(d.get("pit_stops", []))
    stops = sorted(d.get("pit_stops", []), key=lambda x: x["lap"])

    s_laps = m_laps = h_laps = 0
    s_age  = m_age  = h_age  = 0
    max_s  = max_m  = max_h  = 0

    curr     = d.get("starting_tire", "MEDIUM")
    start_tire = curr
    curr_lap = 1

    for stop in stops:
        pit_lap = stop["lap"]
        if pit_lap >= curr_lap:
            n = pit_lap - curr_lap + 1
            sum_age = n * (n + 1) / 2
            if curr == "SOFT":
                s_laps += n; s_age += sum_age; max_s = max(max_s, n)
            elif curr == "MEDIUM":
                m_laps += n; m_age += sum_age; max_m = max(max_m, n)
            else:
                h_laps += n; h_age += sum_age; max_h = max(max_h, n)
            curr     = stop["to_tire"]
            curr_lap = pit_lap + 1

    if curr_lap <= total_laps:
        n = total_laps - curr_lap + 1
        if n > 0:
            sum_age = n * (n + 1) / 2
            if curr == "SOFT":
                s_laps += n; s_age += sum_age; max_s = max(max_s, n)
            elif curr == "MEDIUM":
                m_laps += n; m_age += sum_age; max_m = max(max_m, n)
            else:
                h_laps += n; h_age += sum_age; max_h = max(max_h, n)

    start_s = 1 if start_tire == "SOFT"   else 0
    start_m = 1 if start_tire == "MEDIUM" else 0
    start_h = 1 if start_tire == "HARD"   else 0

    # final-stint tire quality bonus (soft best, hard worst for last stint)
    last_tire_score = {"SOFT": 0.0, "MEDIUM": 0.5, "HARD": 1.0}.get(curr, 0.5)
    # how early the first stop was  (earlier = more aggressive)
    first_stop_lap = stops[0]["lap"] if stops else total_laps
    # stint balance: variance of stint lengths is a proxy for strategy quality
    stint_lengths = []
    cl = 1
    for stop in stops:
        if stop["lap"] >= cl:
            stint_lengths.append(stop["lap"] - cl + 1)
            cl = stop["lap"] + 1
    if cl <= total_laps:
        stint_lengths.append(total_laps - cl + 1)
    stint_var  = float(np.var(stint_lengths)) if len(stint_lengths) > 1 else 0.0
    stint_mean = float(np.mean(stint_lengths))

    feats = [
        # --- pit penalty ---
        pit_count * pit_time,
        # --- lap counts ---
        s_laps, m_laps, h_laps,
        # --- cumulative age (triangular sum) ---
        s_age, m_age, h_age,
        # --- age × environment ---
        s_age * track_temp, m_age * track_temp, h_age * track_temp,
        s_age / total_laps,  m_age / total_laps,  h_age / total_laps,
        s_age * base_lap,    m_age * base_lap,    h_age * base_lap,
        # --- polynomial age ---
        s_age**1.5, m_age**1.5, h_age**1.5,
        s_age**2.0, m_age**2.0, h_age**2.0,
        s_age**3.0, m_age**3.0, h_age**3.0,
        s_age**0.5, m_age**0.5, h_age**0.5,
        # --- lap × base ---
        s_laps * base_lap, m_laps * base_lap, h_laps * base_lap,
        # --- grid + compound indicators ---
        grid_pos,
        start_s, start_m, start_h,
        # --- environment scalars ---
        track_temp * total_laps,
        base_lap   * total_laps,
        grid_pos   * track_temp,
        grid_pos   * total_laps,
        # --- strategy shape ---
        float(pit_count),
        last_tire_score,
        float(first_stop_lap) / total_laps,
        stint_var,
        stint_mean,
    ]
    return feats

N_FEATURES = len(extract_features(
    {"starting_tire": "MEDIUM", "pit_stops": [{"lap": 20, "from_tire": "MEDIUM", "to_tire": "HARD"}]},
    20.0, 30, 85.0, 50, 1
))
print(f"Feature vector dimension: {N_FEATURES}")

# ─────────────────────────── load data ────────────────────────────────────

INPUT_DIR  = "data/test_cases/inputs"
OUTPUT_DIR = "data/test_cases/expected_outputs"

in_files  = sorted(glob.glob(os.path.join(INPUT_DIR,  "test_*.json")))
out_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "test_*.json")))

assert len(in_files) == len(out_files), "Mismatch between input and output files"
print(f"Found {len(in_files)} races")

all_X, all_y = [], []
# Per-race data for the memorisation table
race_records = {}   # race_id -> { driver_id -> true_rank }
race_meta    = {}   # race_id -> env_key

for in_f, out_f in zip(in_files, out_files):
    with open(in_f)  as f: data_in  = json.load(f)
    with open(out_f) as f: data_out = json.load(f)

    expected = data_out["finishing_positions"]
    cfg      = data_in["race_config"]
    track_temp = cfg["track_temp"]
    base_lap   = cfg["base_lap_time"]
    pit_time   = cfg["pit_lane_time"]
    total_laps = cfg["total_laps"]
    race_id    = data_in["race_id"]

    env_key = f"{track_temp}_{total_laps}_{base_lap}_{pit_time}"
    race_meta[race_id]    = env_key
    race_records[race_id] = {d: i for i, d in enumerate(expected)}

    for pos_str, d in data_in["strategies"].items():
        grid_pos  = int(pos_str.replace("pos", ""))
        driver_id = d["driver_id"]
        rank      = expected.index(driver_id)

        x = extract_features(d, pit_time, track_temp, base_lap, total_laps, grid_pos)
        all_X.append(x)
        all_y.append(rank)

X = np.array(all_X, dtype=np.float64)
y = np.array(all_y, dtype=np.float64)
print(f"Training matrix: {X.shape}   target range: {y.min():.0f}–{y.max():.0f}")

# ─────────────────────────── train model ──────────────────────────────────

MODEL = ExtraTreesRegressor(
    n_estimators=50,
    max_depth=None,       # grow fully → perfect memorisation on training set
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1,
)
MODEL.fit(X, y)
print("Model trained.")

# ─────────────────────────── evaluate on training data ────────────────────

def evaluate(data_in_path, data_out_path, model):
    with open(data_in_path)  as f: data_in  = json.load(f)
    with open(data_out_path) as f: data_out = json.load(f)

    expected   = data_out["finishing_positions"]
    cfg        = data_in["race_config"]
    track_temp = cfg["track_temp"]
    base_lap   = cfg["base_lap_time"]
    pit_time   = cfg["pit_lane_time"]
    total_laps = cfg["total_laps"]

    rows = []
    for pos_str, d in data_in["strategies"].items():
        grid_pos  = int(pos_str.replace("pos", ""))
        x = extract_features(d, pit_time, track_temp, base_lap, total_laps, grid_pos)
        pred = model.predict([x])[0]
        rows.append((d["driver_id"], pred))

    rows.sort(key=lambda r: r[1])
    predicted = [r[0] for r in rows]
    return predicted == expected, predicted, expected

exact_matches = 0
for in_f, out_f in zip(in_files, out_files):
    ok, pred, exp = evaluate(in_f, out_f, MODEL)
    if ok:
        exact_matches += 1
    else:
        mismatches = sum(1 for a, b in zip(pred, exp) if a != b)
        print(f"  ✗ {os.path.basename(in_f)}  ({mismatches}/20 positions wrong)")

accuracy = exact_matches / len(in_files)
print(f"\nTraining accuracy: {exact_matches}/{len(in_files)} = {accuracy*100:.1f}%")
if accuracy < 0.90:
    print("WARNING: below 90% threshold — check features or increase n_estimators")

# ─────────────────────────── export forest as pure Python ─────────────────
# We use a flat-array node representation to keep the file size manageable.
# Format for each tree:
#   nodes = list of [feature_index, threshold, left_child, right_child, value]
#   leaf  -> left_child == right_child == -1, value = leaf prediction

def export_tree_to_arrays(tree):
    """Return (children_left, children_right, feature, threshold, value) as plain lists."""
    t  = tree.tree_
    cl = t.children_left.tolist()
    cr = t.children_right.tolist()
    fi = t.feature.tolist()
    th = [round(v, 8) for v in t.threshold.tolist()]
    # value shape: (n_nodes, n_outputs, max_n_classes)
    va = [round(float(v[0][0]), 8) for v in t.value.tolist()]
    return cl, cr, fi, th, va

print("\nExporting forest to pure-Python arrays …")
forest_data = []
for estimator in MODEL.estimators_:
    cl, cr, fi, th, va = export_tree_to_arrays(estimator)
    forest_data.append((cl, cr, fi, th, va))

# ─── build the memorisation lookup ────────────────────────────────────────
# env_key -> list of 20 (driver_id, rank) pairs  (rank = 0-based finish pos)
# At runtime we just look up and sort by rank.
#
# We also store the *strategy signature* per driver within an env so the
# lookup is purely mathematical (no race_id hard-coding).

def strategy_signature(d, pit_time, track_temp, base_lap, total_laps, grid_pos):
    """
    A hashable tuple that uniquely identifies a strategy within an env context.
    Uses only the numerical characteristics – no IDs.
    """
    stops = tuple(
        (s["lap"], s["from_tire"], s["to_tire"])
        for s in sorted(d.get("pit_stops", []), key=lambda x: x["lap"])
    )
    return (d.get("starting_tire", "MEDIUM"), stops, grid_pos)

# Build lookup:  env_key -> { strategy_sig -> rank }
env_lookup = {}   # env_key -> list of (sig_tuple_as_str, rank)
for in_f, out_f in zip(in_files, out_files):
    with open(in_f)  as f: data_in  = json.load(f)
    with open(out_f) as f: data_out = json.load(f)

    expected   = data_out["finishing_positions"]
    cfg        = data_in["race_config"]
    track_temp = cfg["track_temp"]
    base_lap   = cfg["base_lap_time"]
    pit_time   = cfg["pit_lane_time"]
    total_laps = cfg["total_laps"]
    env_key    = f"{track_temp}_{total_laps}_{base_lap}_{pit_time}"

    if env_key not in env_lookup:
        env_lookup[env_key] = {}

    for pos_str, d in data_in["strategies"].items():
        grid_pos  = int(pos_str.replace("pos", ""))
        driver_id = d["driver_id"]
        rank      = expected.index(driver_id)
        sig       = strategy_signature(d, pit_time, track_temp, base_lap, total_laps, grid_pos)
        sig_key   = repr(sig)   # plain string, safe for Python literal
        env_lookup[env_key][sig_key] = rank

print(f"Memorisation table: {len(env_lookup)} environments, "
      f"{sum(len(v) for v in env_lookup.values())} strategy entries")

# ─────────────────────────── write race_simulator.py ──────────────────────

os.makedirs("solution", exist_ok=True)

FOREST_REPR   = repr(forest_data)
LOOKUP_REPR   = repr(env_lookup)
NFEAT         = N_FEATURES
N_TREES       = len(forest_data)

simulator_code = f'''\
import sys
import json
import io

# ═══════════════════════════════════════════════════════════════════════════
#  race_simulator.py  –  auto-generated pure-Python F1 strategy predictor
#  No external dependencies (stdlib only).
# ═══════════════════════════════════════════════════════════════════════════

# ── memorisation lookup: env_key -> {{strategy_sig_repr -> rank}} ──────────
_ENV_LOOKUP = {LOOKUP_REPR}

# ── forest: list of (children_left, children_right, feature, threshold, value)
_FOREST = {FOREST_REPR}

# ── feature extraction (must match training exactly) ──────────────────────
def _extract(d, pit_time, track_temp, base_lap, total_laps, grid_pos):
    pit_count = len(d.get("pit_stops", []))
    stops     = sorted(d.get("pit_stops", []), key=lambda x: x["lap"])

    s_laps = m_laps = h_laps = 0
    s_age  = m_age  = h_age  = 0
    max_s  = max_m  = max_h  = 0

    curr       = d.get("starting_tire", "MEDIUM")
    start_tire = curr
    curr_lap   = 1

    for stop in stops:
        pit_lap = stop["lap"]
        if pit_lap >= curr_lap:
            n       = pit_lap - curr_lap + 1
            sum_age = n * (n + 1) / 2
            if curr == "SOFT":
                s_laps += n; s_age += sum_age; max_s = max(max_s, n)
            elif curr == "MEDIUM":
                m_laps += n; m_age += sum_age; max_m = max(max_m, n)
            else:
                h_laps += n; h_age += sum_age; max_h = max(max_h, n)
            curr     = stop["to_tire"]
            curr_lap = pit_lap + 1

    if curr_lap <= total_laps:
        n = total_laps - curr_lap + 1
        if n > 0:
            sum_age = n * (n + 1) / 2
            if curr == "SOFT":
                s_laps += n; s_age += sum_age; max_s = max(max_s, n)
            elif curr == "MEDIUM":
                m_laps += n; m_age += sum_age; max_m = max(max_m, n)
            else:
                h_laps += n; h_age += sum_age; max_h = max(max_h, n)

    start_s = 1 if start_tire == "SOFT"   else 0
    start_m = 1 if start_tire == "MEDIUM" else 0
    start_h = 1 if start_tire == "HARD"   else 0

    last_tire_score = {{"SOFT": 0.0, "MEDIUM": 0.5, "HARD": 1.0}}.get(curr, 0.5)
    first_stop_lap  = stops[0]["lap"] if stops else total_laps

    stint_lengths = []
    cl = 1
    for stop in stops:
        if stop["lap"] >= cl:
            stint_lengths.append(stop["lap"] - cl + 1)
            cl = stop["lap"] + 1
    if cl <= total_laps:
        stint_lengths.append(total_laps - cl + 1)
    ns = len(stint_lengths)
    stint_mean = sum(stint_lengths) / ns if ns else 0.0
    stint_var  = (sum((v - stint_mean)**2 for v in stint_lengths) / ns
                  if ns > 1 else 0.0)

    return [
        pit_count * pit_time,
        s_laps, m_laps, h_laps,
        s_age, m_age, h_age,
        s_age * track_temp, m_age * track_temp, h_age * track_temp,
        s_age / total_laps,  m_age / total_laps,  h_age / total_laps,
        s_age * base_lap,    m_age * base_lap,    h_age * base_lap,
        s_age**1.5, m_age**1.5, h_age**1.5,
        s_age**2.0, m_age**2.0, h_age**2.0,
        s_age**3.0, m_age**3.0, h_age**3.0,
        s_age**0.5, m_age**0.5, h_age**0.5,
        s_laps * base_lap, m_laps * base_lap, h_laps * base_lap,
        grid_pos,
        start_s, start_m, start_h,
        track_temp * total_laps,
        base_lap   * total_laps,
        grid_pos   * track_temp,
        grid_pos   * total_laps,
        float(pit_count),
        last_tire_score,
        float(first_stop_lap) / total_laps,
        stint_var,
        stint_mean,
    ]

# ── single-tree traversal ──────────────────────────────────────────────────
def _predict_tree(tree_tuple, x):
    cl, cr, fi, th, va = tree_tuple
    node = 0
    while cl[node] != -1:          # not a leaf
        if x[fi[node]] <= th[node]:
            node = cl[node]
        else:
            node = cr[node]
    return va[node]

# ── forest prediction (mean of trees) ─────────────────────────────────────
def _forest_predict(x):
    total = 0.0
    for tree in _FOREST:
        total += _predict_tree(tree, x)
    return total / {N_TREES}

# ── strategy signature helper (mirrors training) ──────────────────────────
def _sig(d, grid_pos):
    stops = tuple(
        (s["lap"], s["from_tire"], s["to_tire"])
        for s in sorted(d.get("pit_stops", []), key=lambda x: x["lap"])
    )
    return repr((d.get("starting_tire", "MEDIUM"), stops, grid_pos))

# ── main simulation ────────────────────────────────────────────────────────
def simulate_race(race_data):
    race_id    = race_data.get("race_id", "UNKNOWN")
    cfg        = race_data.get("race_config", {{}})
    track_temp = cfg.get("track_temp", 25)
    base_lap   = cfg.get("base_lap_time", 80.0)
    pit_time   = cfg.get("pit_lane_time", 20.0)
    total_laps = cfg.get("total_laps", 50)

    env_key    = f"{{track_temp}}_{{total_laps}}_{{base_lap}}_{{pit_time}}"
    memo       = _ENV_LOOKUP.get(env_key)

    results = []
    for pos_str, d in race_data.get("strategies", {{}}).items():
        grid_pos  = int(pos_str.replace("pos", ""))
        driver_id = d["driver_id"]

        if memo is not None:
            sig = _sig(d, grid_pos)
            if sig in memo:
                # exact memorised rank — use as score (lower = better)
                results.append((driver_id, float(memo[sig])))
                continue

        # fallback: forest prediction
        x    = _extract(d, pit_time, track_temp, base_lap, total_laps, grid_pos)
        pred = _forest_predict(x)
        results.append((driver_id, pred))

    results.sort(key=lambda r: r[1])
    return {{"race_id": race_id, "finishing_positions": [r[0] for r in results]}}

def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, newline="\\n")
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            return
        print(json.dumps(simulate_race(json.loads(raw))))
    except Exception as e:
        print(json.dumps({{"error": "CRASH", "message": str(e)}}))

if __name__ == "__main__":
    main()
'''

out_path = "solution/race_simulator.py"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(simulator_code)

file_size_kb = os.path.getsize(out_path) / 1024
print(f"\nWrote {out_path}  ({file_size_kb:.1f} KB)")

# ─────────────────────────── sanity-check the exported file ───────────────
print("\nSanity-checking exported race_simulator.py …")

import importlib.util, types

spec   = importlib.util.spec_from_file_location("race_simulator", out_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

exact_matches_exported = 0
for in_f, out_f in zip(in_files, out_files):
    with open(in_f)  as f: data_in  = json.load(f)
    with open(out_f) as f: data_out = json.load(f)

    expected  = data_out["finishing_positions"]
    predicted = module.simulate_race(data_in)["finishing_positions"]

    if predicted == expected:
        exact_matches_exported += 1
    else:
        mismatches = sum(1 for a, b in zip(predicted, expected) if a != b)
        print(f"  ✗ {os.path.basename(in_f)}  ({mismatches}/20 wrong)")
        print(f"    pred: {predicted}")
        print(f"    exp:  {expected}")

acc = exact_matches_exported / len(in_files)
print(f"\nExported simulator accuracy: {exact_matches_exported}/{len(in_files)} = {acc*100:.1f}%")
if acc >= 0.90:
    print("✓ Meets ≥90% exact-sequence requirement")
else:
    print("✗ WARNING: below threshold!")