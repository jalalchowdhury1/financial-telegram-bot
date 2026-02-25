"""
Compatibility Redirection Script
Ensures Render deployments continue to work if the manually configured
Start Command still points to this file.
"""
import sys
import os

# Add current directory to path to ensure bot package is found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("🚀 [Compatibility Mode] Redirecting to modular bot package: bot.main")
    
    # We use runpy to execute the module as if it were run with -m
    import runpy
    try:
        runpy.run_module("bot.main", run_name="__main__")
    except ImportError as e:
        print(f"❌ Error loading bot.main: {e}")
        print("Falling back to subprocess...")
        import subprocess
        subprocess.run([sys.executable, "-m", "bot.main"])
