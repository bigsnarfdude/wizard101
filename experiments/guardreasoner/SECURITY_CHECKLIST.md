# Security Checklist - Pre-Commit Review

**Last Audit**: 2025-11-18
**Status**: ✅ SAFE TO PUSH

---

## ✅ Completed Security Fixes

### API Keys & Credentials
- [x] **Removed Gemini API key** from all files
  - Fixed: `scripts/generate_with_gemini.py`
  - Fixed: `scripts/augment_dataset.py`
  - Fixed: `DATA_SYNTHESIS_PLAN.md`
  - Fixed: `GEMINI_DATA_GENERATION.md`
- [x] **Created `.env.example`** with placeholder values
- [x] **Updated `.gitignore`** to exclude:
  - `.env` and `.env.local` files
  - `*.key` and `*.pem` files
  - Password files (`*_password`, `CASTLE.password`, etc.)

### Environment Variable Usage
All scripts now use:
```python
import os
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")
```

---

## Files Safe to Commit (New/Modified)

### New Files ✅
- `experiments/guardreasoner/.env.example` - Example only, no real credentials
- `experiments/guardreasoner/DATA_SYNTHESIS_PLAN.md` - Uses environment variables
- `experiments/guardreasoner/GEMINI_DATA_GENERATION.md` - Uses environment variables
- `experiments/guardreasoner/scripts/generate_with_gemini.py` - Uses environment variables
- `experiments/guardreasoner/scripts/augment_dataset.py` - Uses environment variables
- `experiments/guardreasoner/scripts/quick_generate_10k_samples.py` - Clean (no credentials)

### Modified Files ✅
- `.gitignore` - Added environment variable protection

---

## Sensitive Information NOT in Repository ✅

### API Keys
- ❌ Gemini API key: `AIzaSy...` (REMOVED)
- ❌ OpenAI API key (never added)
- ❌ HuggingFace tokens (never added)

### Server Credentials
- ❌ nigel.birs.ca credentials (not in guardreasoner/)
- ❌ castle.birs.ca credentials (not in guardreasoner/)
- ❌ andromeda.birs.ca credentials (not in guardreasoner/)
- ❌ SSH passwords (not in guardreasoner/)

### Personal Information
- ❌ Email addresses with passwords (not in guardreasoner/)
- ❌ User passwords (not in guardreasoner/)
- ❌ Server access credentials (not in guardreasoner/)

---

## How to Use the New Scripts Securely

### 1. Set Environment Variable (Required)

**Local development:**
```bash
# Option A: Export in shell
export GEMINI_API_KEY='your-actual-api-key-here'

# Option B: Create .env file (gitignored)
echo "GEMINI_API_KEY=your-actual-api-key-here" > experiments/guardreasoner/.env

# Then load it:
source experiments/guardreasoner/.env
```

**For long-term use:**
```bash
# Add to ~/.bashrc or ~/.zshrc
echo 'export GEMINI_API_KEY="your-actual-api-key-here"' >> ~/.zshrc
source ~/.zshrc
```

### 2. Verify Environment Variable

```bash
# Check it's set
echo $GEMINI_API_KEY

# Should show your key (not empty)
```

### 3. Run Scripts

```bash
# Now scripts will work
python scripts/generate_with_gemini.py --input data.json --output results.json

# If you get "GEMINI_API_KEY not set" error:
# - Make sure you exported the variable
# - Check spelling: GEMINI_API_KEY (all caps)
```

---

## Pre-Push Checklist

Before running `git push`, verify:

- [ ] No hardcoded API keys in any files
  ```bash
  grep -r "AIzaSy" experiments/guardreasoner/
  # Should return: no matches
  ```

- [ ] No passwords in committed files
  ```bash
  grep -r "password.*=" experiments/guardreasoner/*.py
  # Should return: only variable names, no actual passwords
  ```

- [ ] `.env` file is gitignored
  ```bash
  git check-ignore .env
  # Should return: .env
  ```

- [ ] `.env.example` exists with placeholder values
  ```bash
  cat experiments/guardreasoner/.env.example
  # Should show: GEMINI_API_KEY=your-gemini-api-key-here
  ```

- [ ] All scripts use environment variables
  ```bash
  grep -l "os.getenv" experiments/guardreasoner/scripts/*.py
  # Should list all Python scripts
  ```

---

## What's Protected in .gitignore

```gitignore
# Environment variables and secrets
.env
.env.local
.env.*.local
*.key
*.pem
*_password
*_PASSWORD
CASTLE.password
ANDROMEDA.password
birszoom.password
```

---

## Emergency: If Credentials Were Pushed

If you accidentally pushed credentials to GitHub:

1. **Immediately rotate credentials**
   ```bash
   # Go to Google AI Studio and regenerate API key
   # https://aistudio.google.com/app/apikey
   ```

2. **Remove from git history** (use BFG or git-filter-repo)
   ```bash
   # Install BFG
   brew install bfg

   # Remove sensitive data
   bfg --replace-text passwords.txt wizard101.git

   # Force push (WARNING: rewrites history)
   git push --force
   ```

3. **Notify team** if repository is shared

---

## Security Best Practices

### ✅ DO
- Use environment variables for all secrets
- Keep `.env` files local (never commit)
- Provide `.env.example` with placeholders
- Rotate API keys regularly
- Use read-only API keys when possible
- Review diffs before committing (`git diff`)

### ❌ DON'T
- Hardcode API keys in source code
- Commit `.env` files
- Share credentials in Slack/email
- Use production credentials in development
- Commit server passwords or SSH keys

---

## Current Status

**Repository security: ✅ SAFE**

All new files use environment variables correctly. No credentials are hardcoded.

**Ready to commit!**

```bash
git add experiments/guardreasoner/.env.example
git add experiments/guardreasoner/DATA_SYNTHESIS_PLAN.md
git add experiments/guardreasoner/GEMINI_DATA_GENERATION.md
git add experiments/guardreasoner/scripts/generate_with_gemini.py
git add experiments/guardreasoner/scripts/augment_dataset.py
git add experiments/guardreasoner/scripts/quick_generate_10k_samples.py
git add .gitignore

git commit -m "Add Gemini-powered data generation pipeline

- Create reasoning trace generator using Gemini 2.0 Flash
- Add dataset augmentation scripts (3-5× multiplier)
- Document cost-effective data generation ($23 vs $15,750)
- Include security best practices (environment variables)
- Add .env.example for setup guidance"

git push origin main
```
