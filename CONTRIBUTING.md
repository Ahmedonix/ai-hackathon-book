# Contributing to Physical AI & Humanoid Robotics Book

Thank you for your interest in contributing to the Physical AI & Humanoid Robotics Book! This project aims to provide a comprehensive educational resource for humanoid robotics with ROS 2, simulation, AI perception, and VLA integration.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [How to Contribute](#how-to-contribute)
- [Style Guidelines](#style-guidelines)
- [Submitting Changes](#submitting-changes)
- [Questions](#questions)

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/0/code_of_conduct/). By participating, you are expected to uphold this code.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/ai-hackathon-book.git`
3. Create a new branch: `git checkout -b feature/amazing-feature`
4. Set up your development environment by following the [Quickstart Guide](./specs/main/quickstart.md)
5. Make your changes
6. Test your changes according to the [Testing Guidelines](#testing-guidelines)
7. Commit your changes: `git commit -m 'Add some amazing feature'`
8. Push to the branch: `git push origin feature/amazing-feature`
9. Open a Pull Request

## Project Structure

```
ai-hackathon-book/
├── docs/                       # Docusaurus documentation
│   ├── module1-ros2/           # Module 1 content
│   ├── module2-simulation/     # Module 2 content
│   ├── module3-ai/             # Module 3 content
│   └── module4-vla/            # Module 4 content
├── src/                        # Source code and examples
│   └── ros2_ws/                # ROS 2 workspace
├── specs/                      # Feature specifications
│   ├── 001-ros2-module/        # ROS 2 module spec
│   ├── 002-digital-twin-sim/   # Simulation module spec
│   ├── 003-ai-robot-brain/     # AI module spec
│   ├── 004-vla-cognitive-robotics/  # VLA module spec
│   └── main/                   # Main project specs
└── history/                    # Project history (ADRs, prompts)
```

## How to Contribute

### Writing Documentation

- Follow the established module structure for consistency
- Include practical examples and hands-on exercises
- Ensure all code examples are tested and valid
- Write with grade 8-10 clarity level

### Code Examples

- Follow ROS 2 Iron best practices
- Include proper error handling
- Add comments explaining key concepts
- Ensure code runs in simulation environment

### Reporting Issues

When reporting issues, please include:

- A clear title and description
- Steps to reproduce the issue
- Expected vs. actual behavior
- Environment information (OS, ROS version, etc.)

### Testing Guidelines

- All code examples must run successfully in the simulation environment
- Documentation should build without errors
- Exercises should have clear acceptance criteria

## Style Guidelines

### Documentation Style

- Use clear, concise language
- Break complex concepts into digestible sections
- Include diagrams and visual aids where helpful
- Follow the established format for each module

### Code Style

- Follow ROS 2 Python style guidelines
- Use descriptive variable and function names
- Include docstrings for functions and classes
- Maintain consistency with existing code patterns

## Submitting Changes

1. Ensure all tests pass and documentation builds correctly
2. Update any documentation affected by your changes
3. Add or update tests if applicable
4. Follow the PR template when creating your pull request
5. One of the maintainers will review your PR

## Questions

If you have questions about contributing, feel free to open an issue or reach out to the maintainers.

Thank you for contributing to the Physical AI & Humanoid Robotics Book!