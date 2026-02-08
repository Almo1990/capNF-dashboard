# Version Control Guide for CapNF Dashboard

## Quick Reference

### Current Version
- **Latest Tag**: v1.0.0-mobile
- **Latest Branch**: stable-mobile-v1.0.0
- **Repository**: https://github.com/Almo1990/capNF-dashboard
- **Live Dashboard**: https://almo1990.github.io/capNF-dashboard/

## Helper Scripts

### 1. git_commit_and_push.bat
Quick commit and push changes to GitHub.

**Usage:**
```bash
git_commit_and_push.bat "Your commit message here"
```

**Example:**
```bash
git_commit_and_push.bat "Added new KPI calculation"
```

### 2. git_create_tag.bat
Create and push version tags for stable releases.

**Usage:**
```bash
git_create_tag.bat "v1.1.0" "Description of this version"
```

**Example:**
```bash
git_create_tag.bat "v1.1.0" "Added temperature correction feature"
```

### 3. git_create_branch.bat
Create and push feature branches for experimental work.

**Usage:**
```bash
git_create_branch.bat "branch-name"
```

**Example:**
```bash
git_create_branch.bat "feature-ml-predictions"
```

## Version Control Workflow

### Daily Work (Small Changes)
1. Make your code changes
2. Run: `git_commit_and_push.bat "Brief description of changes"`
3. Done! Changes are on GitHub.

### Testing New Features
1. Create feature branch: `git_create_branch.bat "feature-xyz"`
2. Switch to branch: `C:\Users\abusa\MinGit\cmd\git.exe checkout feature-xyz`
3. Make changes and test
4. Commit often: `git_commit_and_push.bat "Work in progress on feature xyz"`
5. When ready, merge back to main:
   ```bash
   C:\Users\abusa\MinGit\cmd\git.exe checkout main
   C:\Users\abusa\MinGit\cmd\git.exe merge feature-xyz
   C:\Users\abusa\MinGit\cmd\git.exe push origin main
   ```

### Stable Releases
When you have a working version you want to preserve:
1. Test thoroughly
2. Create version tag: `git_create_tag.bat "v1.x.0" "Description"`
3. Optionally create backup branch: `git_create_branch.bat "stable-v1.x.0"`

## Version Naming Convention

### Tags
- **v1.0.0**: Major release (big changes, new features)
- **v1.1.0**: Minor release (small features, improvements)
- **v1.0.1**: Patch (bug fixes only)

### Branches
- **main**: Active development, latest code
- **stable-vX.X.X**: Long-term backups of working versions
- **feature-xyz**: Experimental features being developed

## Restoring Previous Versions

### View Available Versions
```bash
# See all tags (stable versions)
C:\Users\abusa\MinGit\cmd\git.exe tag -l

# See all branches
C:\Users\abusa\MinGit\cmd\git.exe branch -a
```

### Restore a Tagged Version
```bash
# Go to specific version
C:\Users\abusa\MinGit\cmd\git.exe checkout v1.0.0-mobile

# Look around, test it (read-only mode)

# Return to latest
C:\Users\abusa\MinGit\cmd\git.exe checkout main
```

### Restore from Backup Branch
```bash
# Switch to backup branch
C:\Users\abusa\MinGit\cmd\git.exe checkout stable-mobile-v1.0.0

# Return to main
C:\Users\abusa\MinGit\cmd\git.exe checkout main
```

## Current Versions

### v1.0.0-mobile (Latest)
- iPhone/mobile compatibility fully implemented
- Hamburger menu navigation
- Touch-friendly interface
- iOS safe area support
- Sidebar scroll fix for reliable menu clicks
- All mobile features in source template (src/dashboard_app.py)
- Auto-watcher generates mobile-friendly dashboards

### Backup Branch: stable-mobile-v1.0.0
- Same as v1.0.0-mobile
- Long-term backup for safety

## Best Practices

1. **Commit Often**: Better to have many small commits than one giant one
2. **Clear Messages**: Write commit messages that explain WHAT and WHY
3. **Test Before Tagging**: Only create version tags when code is stable
4. **Use Branches for Experiments**: Don't risk breaking main branch
5. **Tag Stable Versions**: Mark working versions so you can always return
6. **Push Regularly**: Keep GitHub up-to-date as backup

## Emergency Recovery

If something breaks and you need to restore:

1. **Recent change broke something**:
   ```bash
   C:\Users\abusa\MinGit\cmd\git.exe log --oneline -10
   # Note the commit hash of last working version
   C:\Users\abusa\MinGit\cmd\git.exe reset --hard <commit-hash>
   C:\Users\abusa\MinGit\cmd\git.exe push origin main --force
   ```

2. **Need to go back to last tagged version**:
   ```bash
   C:\Users\abusa\MinGit\cmd\git.exe checkout v1.0.0-mobile
   C:\Users\abusa\MinGit\cmd\git.exe checkout -b recovery-from-v1.0.0
   C:\Users\abusa\MinGit\cmd\git.exe push origin recovery-from-v1.0.0
   ```

3. **Files accidentally deleted**:
   ```bash
   C:\Users\abusa\MinGit\cmd\git.exe restore .
   ```

## GitHub Repository

- **URL**: https://github.com/Almo1990/capNF-dashboard
- **Live Site**: https://almo1990.github.io/capNF-dashboard/
- **Branches**:
  - `main`: Current development
  - `gh-pages`: Deployed dashboard (auto-updated by deploy_to_pages.py)
  - `stable-mobile-v1.0.0`: Backup of mobile-friendly version

## Need Help?

Common scenarios:

- **"I want to try something risky"**: Create feature branch first
- **"This version works perfectly"**: Create a version tag
- **"Something broke, go back"**: Checkout previous tag or branch
- **"What changed recently?"**: Run `C:\Users\abusa\MinGit\cmd\git.exe log --oneline -10`
- **"I need to update GitHub"**: Run `git_commit_and_push.bat "message"`
