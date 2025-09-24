import subprocess

# === Step 1: Run calibration and undistortion ===
print("Running colorScale.py...")
subprocess.run(["python", "colorScale.py"])

# === Step 2: Run detection and pose estimation ===
print("Running detect.py...")
subprocess.run(["python", "detect.py"])

print("Full pipeline completed.")
