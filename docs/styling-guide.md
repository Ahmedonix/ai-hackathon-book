# Docusaurus Theme and Styling: Physical AI & Humanoid Robotics Book

## Overview

This document outlines the styling and theme updates needed for the Docusaurus documentation to create a cohesive educational look that's appropriate for the Physical AI & Humanoid Robotics curriculum. The design should be professional, accessible, and conducive to learning technical content.

## Color Palette

### Primary Colors
- `--ifm-color-primary`: #2563eb (Primary blue for links, buttons)
- `--ifm-color-primary-dark`: #1d4ed8 (Darker blue for hover states)
- `--ifm-color-primary-darker`: #1e40af (Even darker for active states)
- `--ifm-color-primary-darkest`: #1e3a8a (Darkest for strong emphasis)

### Secondary Colors
- `--ifm-color-secondary`: #64748b (Slate for secondary elements)
- `--ifm-color-success`: #16a34a (Green for success/positive elements)
- `--ifm-color-info`: #0ea5e9 (Cyan for informational elements)
- `--ifm-color-warning`: #f59e0b (Amber for warnings)
- `--ifm-color-danger`: #dc2626 (Red for errors/danger)

### Background Colors
- `--ifm-background-color`: #ffffff (Main background)
- `--ifm-background-surface-color`: #f8fafc (Surface containers)
- `--ifm-footer-background-color`: #0f172a (Dark footer background)
- `--ifm-navbar-background-color`: #1e293b (Dark navbar background)

## Typography

### Font Stack
```css
:root {
  --ifm-font-family-base: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
  --ifm-font-size-base: 16px;
  --ifm-line-height-base: 1.7;
}
```

### Heading Styles
- `h1`: 2.5rem (36px), font-weight 700, letter-spacing -0.02em
- `h2`: 2rem (32px), font-weight 600, letter-spacing -0.01em
- `h3`: 1.5rem (24px), font-weight 600
- `h4`: 1.25rem (20px), font-weight 600
- `h5`: 1.1rem (18px), font-weight 500
- `h6`: 1rem (16px), font-weight 500

### Code Typography
- `code`: Consolas, Monaco, 'Andale Mono', monospace
- Font-size: 0.9em for inline code
- Font-size: 0.85em for code blocks

## Layout and Spacing

### Container Widths
- Max-width: 1440px for main content
- Content padding: 2rem on desktop, 1rem on mobile
- Sidebar width: 280px on desktop

### Spacing Scale
- 1 unit = 0.25rem (4px)
- Common spacings: 0.5rem, 1rem, 1.5rem, 2rem, 2.5rem, 3rem

## Navigation Styling

### Navbar
- Dark background with light text for high contrast
- Logo prominently displayed with course branding
- Navigation items with clear hover/focus states
- Search bar integrated with consistent styling

### Sidebar
- Clean, organized structure matching module progression
- Active section clearly highlighted
- Collapsible sections for sub-topics
- Consistent indentation for hierarchy

## Content Styling

### Text Styling
- Body text: 1.1rem, line-height 1.7, max-width 72ch for readability
- Emphasis: Use primary color for important links/keywords
- Blockquotes: Distinct styling for quotes and callout boxes

### Code Blocks
- Syntax highlighting with consistent color scheme
- Line numbers for longer code examples
- Copy buttons for code snippets
- Light background for better readability

### Callout Boxes
```css
.admonition {
  border-left: 4px solid var(--ifm-color-primary);
  background-color: #f0f9ff;
  border-radius: 0.5rem;
  padding: 1rem;
  margin: 1.5rem 0;
}

.admonition--note {
  border-color: #0ea5e9;
  background-color: #f0f9ff;
}

.admonition--tip {
  border-color: #10b981;
  background-color: #ecfdf5;
}

.admonition--caution {
  border-color: #f59e0b;
  background-color: #fffbeb;
}

.admonition--danger {
  border-color: #ef4444;
  background-color: #fef2f2;
}
```

## Accessibility Features

### Color Contrast
- All text meets WCAG 2.1 AA contrast standards
- Sufficient contrast between background and text (4.5:1 minimum)
- Additional contrast for large text (3:1 minimum)

### Keyboard Navigation
- Clear focus indicators for all interactive elements
- Logical tab order through content
- Skip links for navigation

### Screen Reader Support
- Proper heading hierarchy (H1 → H6)
- Semantic HTML structure
- ARIA labels where needed

## Mobile Responsiveness

### Breakpoints
- Mobile: up to 768px
- Tablet: 768px to 1024px
- Desktop: 1024px and above

### Mobile Adjustments
- Collapsible navigation menu
- Stacked content layout
- Appropriate touch target sizes (44px minimum)
- Readable font sizes without zooming

## Custom Components

### Module Cards
```css
.module-card {
  background: white;
  border-radius: 0.75rem;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
  padding: 1.5rem;
  margin: 1rem 0;
  border-top: 0.25rem solid var(--ifm-color-primary);
}

.module-card h3 {
  color: var(--ifm-color-primary);
  margin-top: 0;
}
```

### Interactive Elements
- Buttons with consistent sizing and spacing
- Hover and active states for all interactive elements
- Loading states for API-dependent components
- Clear visual feedback for user actions

## Implementation

### Custom CSS File
Create `src/css/custom.css` with the above styles:

```css
/**
 * Custom CSS for Physical AI & Humanoid Robotics Book
 */
 
/* Color palette adjustments */
:root {
  --ifm-color-primary: #2563eb;
  --ifm-color-primary-dark: #1d4ed8;
  --ifm-color-primary-darker: #1e40af;
  --ifm-color-primary-darkest: #1e3a8a;
  --ifm-color-secondary: #64748b;
  --ifm-color-success: #16a34a;
  --ifm-color-info: #0ea5e9;
  --ifm-color-warning: #f59e0b;
  --ifm-color-danger: #dc2626;
  --ifm-background-color: #ffffff;
  --ifm-background-surface-color: #f8fafc;
  --ifm-footer-background-color: #0f172a;
  --ifm-navbar-background-color: #1e293b;
}

/* Typography enhancements */
.markdown h1,
.markdown h2,
.markdown h3,
.markdown h4,
.markdown h5,
.markdown h6 {
  font-family: 'Inter', sans-serif;
  font-weight: 600;
}

.markdown h1 {
  font-size: 2.5rem;
  margin-bottom: 1.5rem;
  letter-spacing: -0.02em;
}

.markdown h2 {
  font-size: 2rem;
  margin-bottom: 1.25rem;
  letter-spacing: -0.01em;
}

/* Code block styling */
.docusaurus-highlight-code-line {
  background-color: rgba(0, 0, 0, 0.1);
  display: block;
  margin: 0 calc(-1 * var(--ifm-pre-padding));
  padding: 0 var(--ifm-pre-padding);
}

html[data-theme='dark'] .docusaurus-highlight-code-line {
  background-color: rgba(0, 0, 0, 0.3);
}

/* Admonition styling */
.admonition {
  border-left: 4px solid var(--ifm-color-primary);
  background-color: #f0f9ff;
  border-radius: 0.5rem;
  padding: 1rem;
  margin: 1.5rem 0;
}

.admonition--note {
  border-color: #0ea5e9;
  background-color: #f0f9ff;
}

.admonition--tip {
  border-color: #10b981;
  background-color: #ecfdf5;
}

.admonition--caution {
  border-color: #f59e0b;
  background-color: #fffbeb;
}

.admonition--danger {
  border-color: #ef4444;
  background-color: #fef2f2;
}

/* Module card styling */
.module-card {
  background: white;
  border-radius: 0.75rem;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
  padding: 1.5rem;
  margin: 1rem 0;
  border-top: 0.25rem solid var(--ifm-color-primary);
}

.module-card h3 {
  color: var(--ifm-color-primary);
  margin-top: 0;
}

/* Mobile responsiveness */
@media (max-width: 996px) {
  .module-card {
    padding: 1rem;
  }
  
  .markdown h1 {
    font-size: 2rem;
  }
  
  .markdown h2 {
    font-size: 1.5rem;
  }
}
```

### Theme Configuration
Update `docusaurus.config.js` with the custom CSS:

```js
module.exports = {
  // ... other config
  stylesheets: [
    {
      href: 'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap',
      type: 'text/css',
      rel: 'stylesheet',
    },
  ],
  themeConfig: {
    // ... other theme config
    colorMode: {
      defaultMode: 'light',
      disableSwitch: false,
      respectPrefersColorScheme: true,
    },
  },
  themes: ['@docusaurus/theme-classic'],
  plugins: [
    // ... other plugins
  ],
  styles: [require.resolve('./src/css/custom.css')],
};
```

This styling creates a professional, educational look appropriate for the Physical AI & Humanoid Robotics curriculum, with appropriate color schemes, typography, and layout that facilitates learning of technical content.