# Git & IDE Workflow Guide for Multi-Section Projects with Claude Code

## 背景 Context

我在jupyter notebook里写project, 这个project有14个sections，
1）然后我希望每一个section我做完之后我可以看到这个section我改动过什么代码，我指的是整个section的改变，比如说这个section我可能做过11次改变我只想看到这个section最后的样子和section最开始的样子的对比，而不是中间的11次的改动都看见。 
2）然后我还想在所有14个section完成之后把我的version one保存下来，然后对比说version 1和version zero，也就是空的文档之间，全局有什么变化（e.g. 全部14个section的最后改变）
3）但在每一个section改动过程中，我又不想让AI直接的就replace掉我以前的所有东西，我想看到他到底改了具体的什么小细节（不算真实改动的提案预览），然后我要想这个具体的细节我要不要接受，我想实现这个

目标：我想实现三个颗粒度的保存：
1） 最小颗粒度：像cursor那样，每要propose change的时候，IDE会自动跳出来他想存什么，然后想添加的line就变成绿色然后想删掉的line就变成红色。 然后这些change我想accept的话就accept不accept就可以delete， 而且这些change都还没有处于保存的状态，所以不算staging 
2）我accept exchange之后我把整个section给完成了之后我想保存这个section，所以我想实现一个section的存储 
3）全部section的完成了之后整个文件的存储这样子的话我可以看到14个section整体的变化，然后commit

---

## 最后自己的感想

以后就用jupytext, 然后在.py文件里做改动。 建立一个branch, 改完一个Section我就在这个Branch里commit一稿，然后最后弄完全部的section之后commit一稿大的，e.g. Version2 之类的。之后确定没问题之后merge到main去，然后main里再直接squash，这样的话就不会又branch里面的所有细节的每个section的commit history，但每个Section的commit history又保存了下来。然后去改v3的时候就从现在的main(只有一个大commit version2)去建一个branch, 然后重复上面的步骤，每一个改动commit一下，然后最后改出v3之后再merge回main，然后再squash.

### Branch Strategy Visualization
```
# 开发分支 (feature-branch)
commit: 完成第1节 - 项目介绍
commit: 完成第2节 - 技术架构
commit: 完成第3节 - 实现细节
commit: 修复第2节的图表
commit: 完成第4节 - 测试结果

# merge 到 main 后，main 也有相同记录
# 然后在 main 上 squash 后：
commit: 文档结构和核心内容
commit: 测试结果和修复

# 但 feature-branch 还是原来的细分 commits
```

---

## Methods We Experimented Today

### Method 1: Jupytext + Git Commits (最终选择的方法)

**What it is:** Convert Jupyter notebook to paired Python script for proper version control

**Setup Steps:**
```bash
# 1. Install jupytext
pip install jupytext

# 2. Create paired .py file
jupytext --set-formats ipynb,py:percent your_notebook.ipynb

# 3. Add .ipynb to gitignore (save space)
echo "*.ipynb" >> .gitignore
```

**Workflow Steps:**
1. Open the `.py` file (not `.ipynb`) in your IDE
2. Claude edits the `.py` file
3. You see green/red diff in IDE before accepting
4. Accept/reject changes line by line
5. Save the `.py` file
6. **Sync back to notebook:** `jupytext --sync your_notebook.ipynb`
7. Commit the section: `git commit -m "sec03: Data preprocessing"`

**Key Point:** After editing .py file, must sync back to .ipynb for running cells

---

### Method 2: Git + VS Code Timeline

**What it is:** Use git for version control + VS Code's local history for section tracking

**Setup Steps:**
```bash
# Initialize git
git init
git add .
git commit -m "Initial state"
```

**Workflow Steps:**
1. Claude makes changes to notebook directly
2. After each section, **Ctrl+S** to save
3. VS Code Timeline (left panel) shows "File Saved" entries
4. Right-click Timeline entries → "Compare with Previous" to see section changes
5. After all sections, commit: `git commit -m "Version 1 complete"`

**Pros:** Simple, visual timeline
**Cons:** Timeline saves take disk space, not in git

---

### Method 3: Staging Method

**What it is:** Stage changes after each section, commit once at end

**Workflow Steps:**
1. Complete section 1
2. `git add .` (stages section 1)
3. Complete section 2  
4. `git add .` (stages section 2)
5. Continue for all sections
6. View staged: `git diff --staged`
7. Final commit: `git commit -m "All sections complete"`

**Pros:** One clean commit
**Cons:** Can't easily revert individual sections

---

### Method 4: Many Small Commits

**What it is:** Commit after each section, optionally squash later

**Workflow Steps:**
```bash
# For each section
git add .
git commit -m "sec01: Introduction"

# After all sections, optionally squash
git rebase -i HEAD~14
# Mark all but first as 'squash'
```

**Pros:** Full history, can revert any section
**Cons:** Many commits (need cleanup)

---

## Key Learnings

### Timeline vs Git
- **Timeline**: Tracks file saves regardless of git context
- **Git diff**: Understands branches and commits
- Switching branches creates Timeline entries (confusing but normal)

### Disk Space
- VS Code Timeline: Creates full file snapshots (wastes space)
- Git: Only saves differences (delta compression)
- Disable Timeline if using git: `"workbench.localHistory.enabled": false`

### Jupytext Sync
- **Critical**: After editing .py file, must sync back to .ipynb
- Manual: `jupytext --sync notebook.ipynb`
- Automatic: Install Jupytext VS Code extension

---

## Three Complete Pipeline Methods

### Method 1: Jupytext + Branch Strategy (推荐方法)

**Complete Pipeline Following 最后自己的感想**

#### Initial Setup (One-time)
```bash
# Install jupytext
pip install jupytext

# Create paired .py file
jupytext --set-formats ipynb,py:percent notebook.ipynb

# Add .ipynb to gitignore (save space)
echo "*.ipynb" >> .gitignore
```

#### Version 2 Development
```bash
# 1. Create development branch from main
git checkout main
git checkout -b v2-dev

# 2. Work on each section
#    - Open .py file in IDE
#    - Claude proposes changes → see green/red diff
#    - Accept/reject line by line
#    - Save .py file
#    - Sync back: jupytext --sync notebook.ipynb
#    - Commit section
git commit -m "sec01: Data loading"
git commit -m "sec02: Preprocessing"
# ... repeat for all 14 sections

# 3. Final v2 commit in dev branch
git commit -m "v2: Complete all sections"

# 4. Merge to main with squash
git checkout main
git merge --squash v2-dev
git commit -m "v2: Complete implementation"

# 5. Dev branch preserves detailed history
# Main branch has clean single commit
```

#### Version 3 Development
```bash
# Start from clean main (only has v2 as single commit)
git checkout main
git checkout -b v3-dev

# Repeat the process
# Each section gets a commit in v3-dev
# Finally squash merge to main
```

**Result:**
- Main branch: v1 → v2 → v3 (clean history)
- Dev branches: Preserve all section-by-section commits
- Can always go back to see detailed changes

---

### Method 2: Using Local Saves (Timeline Method)

**For those who prefer visual Timeline over git commits**

#### Setup
```bash
# Initialize git for final commits only
git init
git add .
git commit -m "v0: Initial state"
```

#### Workflow
```
颗粒度1: Preview
├─ Claude proposes changes
├─ See diff in IDE
└─ Accept/reject

颗粒度2: Section Save
├─ Complete section
├─ Ctrl+S to save
├─ Timeline creates "File Saved" entry
└─ Compare with previous save in Timeline

颗粒度3: Global Version
├─ All sections complete
├─ git add .
└─ git commit -m "v1: Complete implementation"
```

#### Using Timeline
- Open VS Code Explorer → TIMELINE section
- Each "File Saved" = one section completion
- Right-click → "Compare with Previous Saved"
- See exactly what changed in that section

**Pros:** Visual, easy to use
**Cons:** Takes disk space, not in git history

---

### Method 3: Git Commits Instead of Local Saves

**Replace Timeline with git commits to save disk space**

#### Setup
```bash
# Disable VS Code local history to save space
# In settings.json:
{
    "workbench.localHistory.enabled": false
}

# Initialize git
git init
git commit -m "v0: Initial state"
```

#### Workflow Options

##### Option A: Many Commits (Then Squash)
```bash
# Each section gets a commit
git commit -m "sec01: Introduction"
git commit -m "sec02: Data loading"
# ... 14 commits total

# Later squash for clean history
git rebase -i HEAD~14
```

##### Option B: Staging Method
```bash
# Complete section 1
git add .  # Stage but don't commit

# Complete section 2
git add .  # Accumulate in staging

# View progress
git diff --staged  # See all staged sections
git diff          # See current work

# Final commit
git commit -m "v1: All sections complete"
```

**Understanding Staging:**
- `git add` = "I'm done with this section, save it to staging area"
- Staging area = temporary holding area before commit
- Can accumulate multiple sections before committing
- Pros: One clean commit at end
- Cons: Can't easily revert individual sections