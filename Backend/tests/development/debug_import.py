import sys
print("--- Starting Import Test ---")
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

try:
    print("Attempting to import 'rembg'...")
    from rembg import remove
    print("Successfully imported 'rembg'.")
except Exception as e:
    print(f"An error occurred during import: {e}")

print("--- Import Test Finished ---")
