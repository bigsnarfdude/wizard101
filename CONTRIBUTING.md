# Contributing to Wizard101

Thank you for your interest in contributing to this educational AI safety project!

## 🎯 Project Goals

This repository aims to:
- Teach AI safety concepts through hands-on implementation
- Demonstrate transparent reasoning in AI systems
- Provide a foundation for building responsible AI applications
- Make complex research accessible to learners

## 🤝 How to Contribute

### Types of Contributions Welcome

1. **Documentation Improvements**
   - Clarify existing explanations
   - Add examples and tutorials
   - Fix typos and formatting
   - Translate content

2. **Code Enhancements**
   - Improve algorithm accuracy
   - Optimize performance
   - Add new reasoning strategies
   - Better error handling

3. **New Features**
   - Additional safety policies
   - Multi-language support
   - LLM integration examples
   - Visualization tools

4. **Educational Content**
   - Learning exercises
   - Video tutorials
   - Use case examples
   - Comparison studies

5. **Bug Fixes**
   - Fix edge cases
   - Improve robustness
   - Better test coverage

### Getting Started

1. **Fork the repository**
   ```bash
   # Click "Fork" on GitHub
   git clone https://github.com/YOUR_USERNAME/wizard101.git
   cd wizard101
   ```

2. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/bug-description
   ```

3. **Make your changes**
   - Keep changes focused and atomic
   - Follow existing code style
   - Add comments where helpful
   - Update documentation

4. **Test your changes**
   ```bash
   cd toy-safety-reasoner
   python3 safety_reasoner.py  # Run basic tests
   python3 examples.py all      # Run all examples
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "Clear description of what you changed"
   ```

6. **Push and create Pull Request**
   ```bash
   git push origin feature/your-feature-name
   # Then create PR on GitHub
   ```

## 📝 Code Style Guidelines

### Python Code
- Follow PEP 8 style guide
- Use descriptive variable names
- Add docstrings to functions
- Keep functions focused (single responsibility)
- Comment complex logic

**Example:**
```python
def evaluate_content(content: str, policy: Dict) -> PolicyEvaluation:
    """
    Evaluate content against a single safety policy.

    Args:
        content: Text to evaluate
        policy: Policy definition with rules and examples

    Returns:
        PolicyEvaluation with classification and reasoning
    """
    # Implementation here
```

### Documentation
- Use clear, simple language
- Include examples
- Explain the "why" not just the "what"
- Add links to relevant research

### Commit Messages
```
Type: Brief description (50 chars or less)

More detailed explanation if needed. Wrap at 72 characters.
Explain what changed and why, not how.

- Bullet points for multiple changes
- Reference issues: "Fixes #123"
```

**Types:**
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `refactor:` Code restructuring
- `test:` Test additions/changes
- `chore:` Maintenance tasks

## 🧪 Testing

Before submitting:

1. **Run existing tests**
   ```bash
   python3 safety_reasoner.py
   python3 examples.py all
   ```

2. **Test your changes**
   - Try different reasoning levels
   - Test edge cases
   - Verify documentation accuracy

3. **Check for regressions**
   - Ensure existing functionality still works
   - Compare output before/after changes

## 📚 Documentation Standards

### README Updates
- Keep getting started section simple
- Update features list when adding capabilities
- Add to roadmap if appropriate

### Code Documentation
- Docstrings for all public functions
- Inline comments for complex logic
- Examples in docstrings where helpful

### Learning Guides
- Start with simple concepts
- Build to advanced topics
- Include hands-on exercises
- Provide clear learning outcomes

## 🔍 Review Process

1. **Automated checks**
   - Code runs without errors
   - No obvious bugs
   - Style is consistent

2. **Maintainer review**
   - Changes align with project goals
   - Code quality is good
   - Documentation is clear

3. **Feedback and iteration**
   - Address review comments
   - Update as needed
   - Re-request review

## 🎓 Educational Focus

Remember this is an **educational project**:

- **Clarity over cleverness** - Readable code teaches better
- **Simplicity over optimization** - Understanding matters more than speed
- **Documentation over assumptions** - Explain everything
- **Examples over theory** - Show, don't just tell

## ❓ Questions?

- **Check existing documentation** - Might already be answered
- **Open an issue** - For questions, ideas, or discussion
- **Start a discussion** - For broader topics

## 🙏 Recognition

Contributors will be:
- Listed in project acknowledgments
- Credited in commit history
- Appreciated by the learning community!

## 📜 Code of Conduct

### Our Pledge

This is a learning environment. We pledge to make participation:
- Welcoming and inclusive
- Harassment-free
- Respectful of different viewpoints
- Focused on education

### Expected Behavior

- Be kind and patient
- Assume good intentions
- Accept constructive criticism gracefully
- Focus on what's best for learners

### Unacceptable Behavior

- Harassment or discrimination
- Trolling or insulting comments
- Personal attacks
- Publishing others' private information

### Enforcement

Maintainers will:
- Address issues fairly
- Remove inappropriate content
- Ban repeat offenders if necessary

## 🚀 Priority Areas

Current focus areas where contributions are especially welcome:

1. **LLM Integration Examples**
   - OpenAI API integration
   - Anthropic Claude examples
   - Local model usage (Llama, etc.)

2. **Multi-language Support**
   - Translation of content
   - Language-specific policies
   - Cross-lingual examples

3. **Adversarial Testing**
   - Jailbreak resistance
   - Edge case discovery
   - Robustness evaluation

4. **Educational Content**
   - Video tutorials
   - Interactive exercises
   - Jupyter notebooks

5. **Production Examples**
   - Discord/Slack bots
   - Web interfaces
   - API implementations

## 📊 Success Metrics

Good contributions:
- ✅ Make concepts clearer
- ✅ Add learning value
- ✅ Improve code quality
- ✅ Enhance documentation
- ✅ Fix bugs or issues

Thank you for helping make AI safety education accessible!

---

*Questions? Open an issue or discussion on GitHub.*
