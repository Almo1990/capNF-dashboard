"""
Deploy dashboard HTML files to GitHub Pages (gh-pages branch).

This script:
1. Auto-detects git (system git or MinGit portable)
2. Clones the gh-pages branch from GitHub into a temp directory
3. Copies updated HTML files from combined_data_plots/
4. Commits and pushes to gh-pages on GitHub

Usage:
    python deploy_to_pages.py              # deploy only gh-pages
    python deploy_to_pages.py --main-too   # also push source changes to main

Can be called from:
    - Run_CapNF_Pipeline.bat
    - auto_update_dashboard.py
    - Manually from the command line
"""

import os
import sys
import shutil
import subprocess
import tempfile
import webbrowser
from datetime import datetime
from pathlib import Path

GITHUB_REPO_URL = "https://github.com/Almo1990/capnf-dashboard.git"
PAGES_URL = "https://Almo1990.github.io/capNF-dashboard/"


def find_git():
    """Find git executable: system git first, then MinGit portable."""
    # Check system git
    git = shutil.which("git")
    if git:
        return git

    # Check MinGit in user home
    mingit = Path.home() / "MinGit" / "cmd" / "git.exe"
    if mingit.exists():
        # Add to PATH so subprocess calls with 'git' also work
        os.environ["PATH"] = (
            str(mingit.parent) + os.pathsep + os.environ.get("PATH", "")
        )
        return str(mingit)

    return None


def run_git(args, cwd=None, capture=True):
    """Run a git command and return the result."""
    git_exe = find_git()
    if not git_exe:
        raise RuntimeError("Git not found. Install Git or MinGit.")

    cmd = [git_exe] + args
    if not capture:
        result = subprocess.run(cmd, cwd=cwd, text=True)
    else:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
        )
    return result


def deploy_gh_pages(base_path):
    """Deploy HTML files from combined_data_plots/ to gh-pages branch on GitHub."""
    plots_dir = os.path.join(base_path, "combined_data_plots")
    if not os.path.exists(plots_dir):
        print("  [ERROR] combined_data_plots/ folder not found")
        return False

    html_files = [f for f in os.listdir(plots_dir) if f.endswith(".html")]
    if not html_files:
        print("  [ERROR] No HTML files found in combined_data_plots/")
        return False

    print(f"  Found {len(html_files)} HTML files to deploy")

    # Clone gh-pages from GitHub (NOT from local repo)
    # git clone needs a non-existing target, so use a unique subdir
    tmp_parent = tempfile.mkdtemp(prefix="capnf-deploy-")
    tmp_dir = os.path.join(tmp_parent, "repo")
    try:
        print("  Cloning gh-pages branch from GitHub...")
        clone_result = run_git(
            [
                "clone",
                "--branch",
                "gh-pages",
                "--single-branch",
                "--depth",
                "1",
                GITHUB_REPO_URL,
                tmp_dir,
            ],
            capture=False,
        )

        if clone_result.returncode != 0:
            print("  gh-pages branch not found, creating it...")
            run_git(["init"], cwd=tmp_dir, capture=False)
            run_git(["checkout", "--orphan", "gh-pages"], cwd=tmp_dir, capture=False)
            run_git(
                ["remote", "add", "origin", GITHUB_REPO_URL], cwd=tmp_dir, capture=False
            )

        # Configure git user in temp repo
        run_git(
            ["config", "user.email", "almo1990@users.noreply.github.com"], cwd=tmp_dir
        )
        run_git(["config", "user.name", "Almo1990"], cwd=tmp_dir)

        # Remove old HTML files
        for f in os.listdir(tmp_dir):
            if f.endswith(".html"):
                os.remove(os.path.join(tmp_dir, f))

        # Copy new HTML files
        for f in html_files:
            shutil.copy2(
                os.path.join(plots_dir, f),
                os.path.join(tmp_dir, f),
            )

        # Stage, commit, push
        run_git(["add", "-A"], cwd=tmp_dir)

        commit_msg = f"Deploy dashboard: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        commit_result = run_git(["commit", "-m", commit_msg], cwd=tmp_dir)

        if commit_result.returncode != 0:
            if "nothing to commit" in (commit_result.stdout or ""):
                print("  No changes to deploy (dashboard already up to date)")
                return True
            print(f"  [ERROR] Commit failed: {commit_result.stderr}")
            return False

        print("  Pushing to gh-pages on GitHub...")
        push_result = run_git(
            ["push", "origin", "gh-pages"], cwd=tmp_dir, capture=False
        )

        if push_result.returncode == 0:
            print(f"  [OK] Deployed to GitHub Pages!")
            print(f"  Dashboard will be live in ~1 minute at: {PAGES_URL}")
            print(f"  Opening dashboard in your default browser...")
            # Open GitHub Pages URL in browser
            try:
                webbrowser.open(PAGES_URL)
            except Exception as e:
                print(f"  ⚠️  Could not open browser automatically: {e}")
                print(f"  Please open manually: {PAGES_URL}")
            return True
        else:
            print(f"  [ERROR] Push failed (exit code {push_result.returncode})")
            return False

    finally:
        # Clean up temp directory
        shutil.rmtree(tmp_parent, ignore_errors=True)


def push_main_branch(base_path):
    """Push all pipeline-relevant changes to main branch (respects .gitignore)."""
    print("  Pushing source changes to main...")
    run_git(["add", "-A"], cwd=base_path)
    commit_msg = f"Auto-update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    commit_result = run_git(["commit", "-m", commit_msg], cwd=base_path)
    if commit_result.returncode != 0:
        if "nothing to commit" in (commit_result.stdout or ""):
            print("  [OK] Main branch already up to date")
            return
    result = run_git(["push", "origin", "main"], cwd=base_path, capture=False)
    if result.returncode == 0:
        print("  [OK] Main branch updated")
    else:
        print(f"  [WARNING] Main push failed (exit code {result.returncode})")


def main():
    base_path = str(Path(__file__).parent.absolute())

    print("\n[Deploy] Starting GitHub Pages deployment...")

    # Check git is available
    git = find_git()
    if not git:
        print("  [ERROR] Git not found!")
        print("  Install Git from https://git-scm.com/download/win")
        print("  Or MinGit (portable, no admin): see GITHUB_SETUP.txt")
        sys.exit(1)
    print(f"  Using git: {git}")

    # Optionally push main branch first
    if "--main-too" in sys.argv:
        push_main_branch(base_path)

    # Deploy to gh-pages
    success = deploy_gh_pages(base_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
