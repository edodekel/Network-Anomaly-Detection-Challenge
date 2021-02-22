import subprocess

print("Starting first")
subprocess.call(["python", "first.py"])
print("Starting second")
subprocess.call(["python", "second.py"])
print("Starting third")
subprocess.call(["python", "third.py"])

print("models are ready")
