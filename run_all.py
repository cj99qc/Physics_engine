import sys
import os
import glob
import subprocess

examples_dir = os.path.join(os.path.dirname(__file__), 'examples')
py_files = sorted(glob.glob(os.path.join(examples_dir, '*.py')))

for f in py_files:
    print(f"Running {os.path.basename(f)}...")
    # Run each example as a subprocess, capture output
    # Since some might be GUI, let's run them with an environment variable to disable GUI
    env = os.environ.copy()
    env["HEADLESS_TEST"] = "1"
    try:
        res = subprocess.run([sys.executable, f], env=env, capture_output=True, text=True, timeout=10)
        if res.returncode != 0:
            print(f"FAILED: {os.path.basename(f)}\n{res.stderr}\n{res.stdout}")
        else:
            print(f"SUCCESS: {os.path.basename(f)}")
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT (likely blocked on GUI): {os.path.basename(f)}")
