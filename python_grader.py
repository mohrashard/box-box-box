import sys
import os
import json
import glob
import io

# Setup UTF-8 for Windows
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

# Path configuration
INPUTS_DIR = os.path.join("data", "test_cases", "inputs")
EXPECTED_DIR = os.path.join("data", "test_cases", "expected_outputs")
SOLUTION_DIR = os.path.abspath("solution")

# ── TURBO UPGRADE: DIRECT IMPORT ─────────────────────────────────────────────
sys.path.append(SOLUTION_DIR)
try:
    import race_simulator
    print("🚀 Simulator loaded into memory for high-speed testing.")
except ImportError:
    print("❌ Error: Could not find race_simulator.py in solution folder.")
    sys.exit(1)

def coloured(text, code): return f"\033[{code}m{text}\033[0m"

def main():
    test_files = sorted(glob.glob(os.path.join(INPUTS_DIR, "test_*.json")))
    has_answers = os.path.isdir(EXPECTED_DIR)
    
    passed = failed = 0
    print(coloured("=" * 58, "0;34"))
    print(f"Running {len(test_files)} tests at Turbo Speed...")
    
    for input_path in test_files:
        test_name = os.path.splitext(os.path.basename(input_path))[0]
        test_id = test_name.replace("test_", "TEST_")

        with open(input_path, "r", encoding="utf-8") as f:
            race_data = json.load(f)

        try:
            # CALLING THE FUNCTION DIRECTLY (No subprocess overhead!)
            result = race_simulator.simulate_race(race_data)
            predicted = result.get("finishing_positions")
            
            if has_answers:
                answer_path = os.path.join(EXPECTED_DIR, f"{test_name}.json")
                with open(answer_path, "r", encoding="utf-8") as f:
                    expected = json.load(f).get("finishing_positions")

                if predicted == expected:
                    print(f"  {coloured('✓', '0;32')} {test_id}")
                    passed += 1
                else:
                    diff = next((i for i, (a, b) in enumerate(zip(predicted, expected)) if a != b), 0)
                    print(f"  {coloured('✗', '0;31')} {test_id} - Incorrect (diff at pos {diff})")
                    failed += 1
        except Exception as e:
            print(f"  {coloured('✗', '0;31')} {test_id} - Crash: {e}")
            failed += 1

    print("\n" + coloured("=" * 58, "0;34"))
    print(f"Total: {len(test_files)} | Passed: {passed} | Failed: {failed}")
    print(f"Pass Rate: {round(passed * 100 / len(test_files), 1)}%")

if __name__ == "__main__":
    main()