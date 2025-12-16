# 001: Documentation Framework and Technology Stack

Date: 2025-01-15

## Status

Accepted

## Context

We need to create a comprehensive educational resource on Physical AI & Humanoid Robotics consisting of 4 modules. The documentation must be interactive, include code examples, support technical content with diagrams, and be deployable to GitHub Pages. The target audience is beginner-to-intermediate developers who need clear, reproducible examples with step-by-step guidance.

## Decision

We will use Docusaurus with MDX as the documentation framework, with the following technology stack:
- Framework: Docusaurus for documentation generation
- Format: MDX (Markdown with React components) for content creation
- Deployment: GitHub Pages for hosting
- Primary Language: Python 3.10-3.11 for ROS 2 Iron compatibility
- Additional Technologies: JavaScript/TypeScript for Docusaurus customization, Shell scripting for automation

## Consequences

Positive:
- Docusaurus provides excellent support for technical documentation with integrated search, versioning, and responsive design
- MDX allows embedding React components in Markdown for interactive documentation
- GitHub Pages provides seamless integration with GitHub, free hosting, and CI/CD capabilities
- Python 3.10-3.11 ensures compatibility with ROS 2 Iron
- Structured approach allows for consistent module formatting

Negative:
- Learning curve for team members unfamiliar with Docusaurus/MDX
- Dependency on Node.js ecosystem for documentation site
- Potential performance issues with large documentation sets
- Need to maintain both Python (ROS 2) and JavaScript (Docusaurus) codebases

## Alternatives

1. GitBook: More limited customization and less interactive capabilities, but simpler for basic documentation needs
2. Sphinx: Python-focused documentation tool, but less suitable for mixed-language educational content
3. Hugo: Static site generator with good performance, but less interactive and more complex for embedding React components
4. Custom solution with separate documentation and code repositories: More complex to maintain, but would offer complete control

## References

- plan.md: Technical Context section
- research.md: Docusaurus Documentation Framework decision
- data-model.md: CodeExample entity definition