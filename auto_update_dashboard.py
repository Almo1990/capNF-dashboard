"""
AUTO-UPDATE DASHBOARD SERVICE
Monitors Data/ folder for new .tsv files and automatically:
1. Runs the CapNF pipeline
2. Generates updated dashboard HTML files
3. Deploys to GitHub Pages

Usage:
    python auto_update_dashboard.py

Or use the provided batch file:
    start_auto_watcher.bat
"""

import time
import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except ImportError:
    print("❌ Error: watchdog library not installed")
    print("📦 Installing watchdog...")
    subprocess.run([sys.executable, "-m", "pip", "install", "watchdog"])
    print("✅ Installation complete. Please restart this script.")
    sys.exit(1)


class DataFileHandler(FileSystemEventHandler):
    """Handles file system events in the Data/ folder"""

    def __init__(self, base_path):
        self.base_path = base_path
        self.processing = False
        self.last_processed = None

    def on_created(self, event):
        """Triggered when a new file is created"""
        if event.is_directory:
            return

        if event.src_path.endswith(".tsv"):
            # Avoid processing the same file multiple times
            if self.processing or event.src_path == self.last_processed:
                return

            print("\n" + "=" * 60)
            print(f"📂 New data file detected: {os.path.basename(event.src_path)}")
            print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 60 + "\n")

            # Wait a moment to ensure file is fully written
            time.sleep(2)

            self.processing = True
            self.last_processed = event.src_path

            try:
                self.run_pipeline()
                self.deploy_to_github()
                print("\n" + "=" * 60)
                print("✅ SUCCESS! Dashboard is now live and updated")
                print("🌐 View at: https://Almo1990.github.io/capNF-dashboard/")
                print("=" * 60 + "\n")
            except Exception as e:
                print(f"\n❌ Error during processing: {e}")
                print("Please check the logs and try running manually.\n")
            finally:
                self.processing = False

    def run_pipeline(self):
        """Execute the main pipeline"""
        print("🔄 Step 1/2: Running CapNF pipeline...")
        print("-" * 60)

        result = subprocess.run(
            [sys.executable, "main.py"],
            cwd=self.base_path,
            capture_output=False,
            text=True,
        )

        if result.returncode != 0:
            raise Exception(f"Pipeline failed with return code {result.returncode}")

        print("-" * 60)
        print("✅ Pipeline completed successfully\n")

    def deploy_to_github(self):
        """Deploy updated dashboard files to GitHub Pages (gh-pages branch)"""
        print("☁️ Step 2/2: Deploying to GitHub Pages...")
        print("-" * 60)

        plots_dir = os.path.join(self.base_path, "combined_data_plots")
        if not os.path.exists(plots_dir):
            print("⚠️ Warning: combined_data_plots/ folder not found")
            return

        # Auto-detect git: check PATH first, then MinGit in user home
        git_exe = "git"
        import shutil as _shutil
        if _shutil.which("git") is None:
            mingit = os.path.join(Path.home(), "MinGit", "cmd", "git.exe")
            if os.path.exists(mingit):
                git_exe = mingit
                os.environ["PATH"] = str(Path(mingit).parent) + os.pathsep + os.environ.get("PATH", "")
            else:
                print("⚠️ Warning: Git not found (install Git or MinGit)")
                print("-" * 60 + "\n")
                return

        # Check if git is configured
        result = subprocess.run(
            [git_exe, "status"], cwd=self.base_path, capture_output=True, text=True
        )

        if result.returncode != 0:
            print("⚠️ Warning: Git repository not initialized")
            return

        # First, commit source changes to main branch
        subprocess.run(
            ["git", "add", "combined_data_plots/", "outputs/*.json"],
            cwd=self.base_path,
            capture_output=True,
        )
        commit_msg = f"Auto-update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        subprocess.run(
            ["git", "commit", "-m", commit_msg],
            cwd=self.base_path,
            capture_output=True,
        )
        subprocess.run(
            ["git", "push", "origin", "main"],
            cwd=self.base_path,
            capture_output=True,
        )

        # Now deploy to gh-pages branch
        # Use a temporary directory approach to avoid disrupting the working tree
        import tempfile
        import shutil

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Clone the repo into a temp directory (shallow, just gh-pages)
            clone_result = subprocess.run(
                [
                    "git", "clone", "--branch", "gh-pages", "--single-branch",
                    "--depth", "1", self.base_path, tmp_dir,
                ],
                capture_output=True, text=True,
            )

            if clone_result.returncode != 0:
                print("⚠️ gh-pages branch not found, creating it...")
                # Initialize a fresh repo in temp dir
                subprocess.run(["git", "init"], cwd=tmp_dir, capture_output=True)
                subprocess.run(
                    ["git", "checkout", "--orphan", "gh-pages"],
                    cwd=tmp_dir, capture_output=True,
                )
                # Set the remote
                subprocess.run(
                    ["git", "remote", "add", "origin",
                     "https://github.com/Almo1990/capNF-dashboard.git"],
                    cwd=tmp_dir, capture_output=True,
                )

            # Remove old HTML files from temp dir
            for f in os.listdir(tmp_dir):
                if f.endswith(".html"):
                    os.remove(os.path.join(tmp_dir, f))

            # Copy all HTML files from combined_data_plots/ to temp dir root
            for f in os.listdir(plots_dir):
                if f.endswith(".html"):
                    shutil.copy2(
                        os.path.join(plots_dir, f),
                        os.path.join(tmp_dir, f),
                    )

            # Commit and push
            subprocess.run(["git", "add", "-A"], cwd=tmp_dir, capture_output=True)
            deploy_msg = f"Deploy dashboard: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            commit_result = subprocess.run(
                ["git", "commit", "-m", deploy_msg],
                cwd=tmp_dir, capture_output=True, text=True,
            )

            if "nothing to commit" in commit_result.stdout:
                print("ℹ️ No dashboard changes to deploy")
                print("-" * 60 + "\n")
                return

            push_result = subprocess.run(
                ["git", "push", "origin", "gh-pages"],
                cwd=tmp_dir, capture_output=True, text=True,
            )

            if push_result.returncode == 0:
                print("✅ Deployed to GitHub Pages")
                print("   Dashboard will be live in ~1 minute")
                print("   🌐 https://Almo1990.github.io/capNF-dashboard/")
            else:
                print("⚠️ Push to gh-pages failed")
                print(f"   Error: {push_result.stderr}")

        print("-" * 60 + "\n")


def main():
    """Main entry point"""
    # Get base path (CapNF directory)
    base_path = Path(__file__).parent.absolute()
    data_folder = base_path / "Data"

    # Check if Data folder exists
    if not data_folder.exists():
        print(f"❌ Error: Data folder not found at {data_folder}")
        print("Please ensure you're running this from the CapNF directory")
        sys.exit(1)

    # Display startup banner
    print("\n" + "=" * 60)
    print("  🚀 CapNF Auto-Update Dashboard Service")
    print("=" * 60)
    print(f"\n📁 Monitoring folder: {data_folder}")
    print("📊 GitHub Pages: https://Almo1990.github.io/capNF-dashboard/")
    print("\n💡 Instructions:")
    print("   1. Drop new .tsv files into the Data/ folder")
    print("   2. Pipeline will run automatically (2-5 minutes)")
    print("   3. Dashboard updates online automatically")
    print("\n⚠️  Keep this window open for auto-updates to work")
    print("   Press Ctrl+C to stop monitoring\n")
    print("=" * 60 + "\n")
    print("👀 Watching for new data files...\n")

    # Set up file watcher
    event_handler = DataFileHandler(base_path)
    observer = Observer()
    observer.schedule(event_handler, str(data_folder), recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("🛑 Stopping auto-update service...")
        print("=" * 60 + "\n")
        observer.stop()

    observer.join()


if __name__ == "__main__":
    main()
