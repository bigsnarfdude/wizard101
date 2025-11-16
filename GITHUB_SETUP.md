# GitHub Setup Guide

Quick guide to push this repository to GitHub as bigsnarfdude.

## 🚀 Quick Setup

### 1. Create GitHub Repository

Go to: https://github.com/new

**Settings:**
- Repository name: `wizard101`
- Description: `Educational AI safety projects - Learn how safety reasoners work through hands-on implementation`
- Visibility: **Public** (for educational sharing)
- ⚠️ **DO NOT** initialize with README (we already have one)

### 2. Push to GitHub

```bash
# Add remote (replace with your actual URL)
git remote add origin https://github.com/bigsnarfdude/wizard101.git

# Rename branch to main (optional, recommended)
git branch -M main

# Push to GitHub
git push -u origin main
```

### 3. Configure Repository Settings

On GitHub, go to repository → Settings:

#### Topics (for discoverability)
Add these topics:
- `ai-safety`
- `machine-learning`
- `education`
- `content-moderation`
- `chain-of-thought`
- `reasoning`
- `python`
- `openai`
- `safety-reasoner`

#### About Section
```
Educational AI safety projects demonstrating how modern safety reasoners work.
Includes toy implementation with chain-of-thought reasoning, multi-policy
classification, and comprehensive learning guides.
```

Website: (leave blank or add your site)

#### Features
Enable:
- ✅ Issues (for bug reports and questions)
- ✅ Discussions (for learning community)
- ❌ Wiki (not needed, we have docs)
- ❌ Projects (not needed yet)

## 📝 Optional: Create GitHub Actions

Add automated testing (optional):

Create `.github/workflows/test.yml`:
```yaml
name: Test Safety Reasoner

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'
    - name: Run tests
      run: |
        cd toy-safety-reasoner
        python3 safety_reasoner.py
        python3 examples.py all
```

## 🎯 Post-Setup Tasks

### 1. Add Repository Description
On main page, click "⚙️" next to About and add:
- Description
- Website (if any)
- Topics

### 2. Enable Discussions
Settings → Features → ✅ Discussions

Create categories:
- 💡 Ideas - New features and improvements
- 🙋 Q&A - Questions about the project
- 📚 Show and Tell - Share your implementations
- 💬 General - Anything else

### 3. Create Issue Templates

GitHub → Settings → Features → Issues → Set up templates

**Bug Report Template:**
```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Run command '...'
2. Enter content '...'
3. See error

**Expected behavior**
What you expected to happen.

**Environment:**
- OS: [e.g. macOS, Linux, Windows]
- Python version: [e.g. 3.8]

**Additional context**
Add any other context about the problem.
```

**Feature Request Template:**
```markdown
**Is your feature request related to a problem?**
A clear description of the problem.

**Describe the solution you'd like**
What you want to happen.

**Describe alternatives you've considered**
Other approaches you've thought about.

**Additional context**
Any other context, mockups, or examples.
```

### 4. Add Social Preview Image

Settings → General → Social preview

Upload: `safetyReasoner.png` (1280x640px recommended)

This shows when sharing on social media.

### 5. Pin Repository (Optional)

On your profile → Repositories → Customize pins
- Pin wizard101 to show on your profile

## 📢 Sharing Your Repository

### README Badges (Optional)

Add to top of README.md:
```markdown
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.6+-blue.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)
```

### Social Media Posts

**Twitter/X:**
```
🎓 New educational project: Wizard101 - Learn how AI safety reasoners work!

✅ Chain-of-thought reasoning
✅ Multi-policy classification
✅ Interactive demos
✅ Based on OpenAI research

Perfect for anyone learning about responsible AI 🤖

https://github.com/bigsnarfdude/wizard101
```

**LinkedIn:**
```
Excited to share an educational project on AI safety!

Wizard101 demonstrates how modern safety reasoners work through a hands-on
toy implementation. Based on OpenAI's gpt-oss-safeguard research.

Features:
- Transparent chain-of-thought reasoning
- Multi-policy content classification
- Interactive learning demos
- Comprehensive documentation

Great for anyone interested in responsible AI development and content moderation systems.

Check it out: https://github.com/bigsnarfdude/wizard101

#AIEthics #MachineLearning #Education
```

## 🔗 Useful GitHub Features

### Insights
- Watch stars/forks growth
- See which files get viewed most
- Track community engagement

### Releases
When you add features:
```bash
git tag -a v1.0.0 -m "Initial release"
git push origin v1.0.0
```

Then create release on GitHub with notes.

### Sponsors (Optional)
If you want to accept donations for educational content.

## ✅ Verification Checklist

After setup, verify:
- [ ] Repository is public
- [ ] README displays correctly
- [ ] Image (safetyReasoner.png) shows in README
- [ ] Topics are added
- [ ] License shows correctly
- [ ] Code is searchable
- [ ] Issues are enabled
- [ ] Contributing guide is visible

## 🎯 Next Steps

1. **Share with community**
   - Post on social media
   - Share in relevant Discord/Slack communities
   - Post on Reddit (r/MachineLearning, r/learnmachinelearning)

2. **Engage with users**
   - Respond to issues
   - Answer questions in discussions
   - Accept pull requests

3. **Iterate and improve**
   - Add requested features
   - Improve documentation based on feedback
   - Create video tutorials

4. **Build community**
   - Encourage contributions
   - Highlight user projects
   - Create learning challenges

---

Ready to go live? Run the push command above! 🚀
