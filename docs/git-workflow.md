# Git Workflow and Version Control Best Practices

This document outlines the Git workflow and version control best practices for the Physical AI & Humanoid Robotics Book project. Following these practices will ensure smooth collaboration and maintainable documentation/code.

## Branch Strategy

We follow a simplified Git flow model appropriate for documentation projects:

```
main (production-ready content)
├── module-1-development
├── module-2-development
├── module-3-development
└── module-4-development
```

- **main**: Production-ready content that is published to the documentation site
- **module-* branches**: Development branches for each module
- **feature branches**: Short-lived branches for specific features or fixes

## Commit Message Guidelines

We follow conventional commits format:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Types
- `feat`: New module content, chapters, or features
- `fix`: Corrections to content or code examples
- `docs`: Documentation-only changes
- `style`: Formatting, missing semi-colons, etc. (no code change)
- `refactor`: Code changes that neither fix a bug nor add a feature
- `test`: Adding missing tests or correcting existing tests
- `chore`: Other changes that don't modify content files

### Examples
```
feat(module1): Add chapter on ROS 2 architecture

docs: Update installation instructions for ROS 2 Iron
fix: Correct code example in topic publisher section
```

## Content Organization

### Documentation Structure
```
docs/
├── module1-ros2/
│   ├── index.md
│   ├── architecture.md
│   ├── nodes.md
│   ├── topics-services-actions.md
│   ├── urdf.md
│   ├── launch-files.md
│   ├── ai-integration.md
│   └── exercises/
│       ├── exercise-1.md
│       └── exercise-2.md
├── module2-simulation/
├── module3-ai/
└── module4-vla/
```

### Commit Organization
- Group related changes in single commits
- Each module should have self-contained commits
- Keep commits focused on a single aspect (e.g., don't mix content changes with style changes)

## Working with Branches

### Creating a New Module Branch
```bash
git checkout main
git pull origin main
git checkout -b module-2-development
```

### Creating a Feature Branch
```bash
git checkout module-2-development
git pull origin module-2-development
git checkout -b feature/new-chapter
```

### Merging Changes
```bash
# Complete your feature
git add .
git commit -m "feat: Add new chapter content"
git push origin feature/new-chapter

# On GitHub, create a PR from feature/new-chapter to module-2-development
# After review and approval, merge the PR

# Update your local module branch
git checkout module-2-development
git pull origin module-2-development
```

## Pull Request Process

1. Ensure your branch is up-to-date with the target branch
2. Write clear commit messages following the guidelines
3. Create a descriptive PR title and description
4. Include what changes were made and why
5. Request reviews from relevant maintainers
6. Address feedback and update the PR as needed

## Code and Content Review

### Documentation Review Checklist
- [ ] Technical accuracy of content
- [ ] Code examples run successfully
- [ ] Formatting follows Docusaurus MDX standards
- [ ] Links are valid and functional
- [ ] Exercises have clear acceptance criteria
- [ ] Content is appropriate for target audience

### Technical Requirements
- All code examples must work in clean ROS 2 Iron environment
- Docusaurus builds must complete without errors
- Markdown formatting must be valid
- Images and assets must be optimized

## Handling Large Changes

For large content additions:
1. Break work into logical commits
2. Each commit should represent a complete unit (e.g., a complete chapter section)
3. Test build frequently to catch issues early
4. Get early feedback on approach before completing large sections

## Recovery Procedures

### Undoing a Commit (if not pushed)
```bash
git reset --soft HEAD~1  # Keep changes staged
# or
git reset --hard HEAD~1  # Discard changes completely
```

### Undoing a Push
```bash
git push --force-with-lease origin your-branch
```
⚠️ Use with caution, especially on shared branches

### Reverting a Merged Change
```bash
git revert <commit-hash>
git push origin main
```

## Special Considerations for Documentation

### Content Versioning
- Changes to published content should consider backward compatibility
- Breaking changes to examples should be noted in release notes
- Major content reorganizations should be scheduled appropriately

### Asset Management
- Optimize images and media files
- Use descriptive filenames
- Maintain consistent directory structure for assets

## Troubleshooting Common Issues

### Merge Conflicts
1. Identify conflicted files: `git status`
2. Open conflicted files and look for conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`)
3. Resolve conflicts manually
4. Add resolved files: `git add .`
5. Complete the merge: `git commit`

### Docusaurus Build Errors
- Check for valid Markdown syntax
- Verify all internal links
- Ensure code blocks have proper language identifiers
- Validate frontmatter in MDX files

Following these practices will help maintain a clean, collaborative, and maintainable documentation project for the Physical AI & Humanoid Robotics Book.