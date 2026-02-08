"""Test watchdog import"""
import sys
print(f"Python: {sys.version}")
print(f"Python executable: {sys.executable}")
print()

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    print("✅ SUCCESS: Watchdog imported successfully!")
    print(f"   Observer: {Observer}")
    print(f"   FileSystemEventHandler: {FileSystemEventHandler}")
except ImportError as e:
    print(f"❌ ERROR: Failed to import watchdog")
    print(f"   {e}")
    sys.exit(1)

print("\nTest completed!")
