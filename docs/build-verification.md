# Documentation Build and Deployment Verification: Physical AI & Humanoid Robotics Book

## Overview

This document provides comprehensive procedures and verification steps for building and deploying the Physical AI & Humanoid Robotics curriculum documentation. It includes setup procedures, build processes, quality assurance checks, and deployment validation to ensure the documentation is consistently available and properly formatted.

## Section 1: Environment Setup and Prerequisites

### 1.1 Development Environment Requirements

#### Operating System
- **Primary Support**: Ubuntu 22.04 LTS or later
- **Alternative**: macOS 10.15+ or Windows 10+ with WSL2
- **Architecture**: x86_64 (Intel/AMD 64-bit) or ARM64

#### System Requirements
```bash
# Minimum system specifications
- RAM: 8GB (16GB recommended)
- Storage: 20GB free space
- Processor: Dual-core (Quad-core recommended)
- Internet: Broadband connection for dependencies
```

#### Required Software Stack
```bash
# Essential tools and versions
- Node.js: v18.0 or later
- npm: v8.0 or later  
- Git: v2.30 or later
- Python: v3.8 or later
- Java: OpenJDK 11 or later (if needed for other tools)
```

### 1.2 Installation Verification Commands

```bash
# Verify system prerequisites
echo "=== System Prerequisites Verification ==="

echo "1. Checking OS and version:"
uname -a

echo "2. Checking available disk space:"
df -h ~ | tail -1

echo "3. Checking available memory:"
free -h

echo "4. Checking Node.js installation:"
node -v || echo "ERROR: Node.js not found"

echo "5. Checking npm installation:"  
npm -v || echo "ERROR: npm not found"

echo "6. Checking Git installation:"
git --version || echo "ERROR: Git not found"

echo "7. Checking Python installation:"
python3 -c "import sys; print(f'Python version: {sys.version}')"
```

### 1.3 Repository Setup

```bash
# Clone and setup repository
mkdir -p ~/speckit-curriculum
cd ~/speckit-curriculum

# Clone the repository
git clone https://github.com/your-org/ai-hackathon-book.git .
git checkout main

# Verify repository structure
echo "=== Repository Structure ==="
ls -la
echo "=== Documentation Directories ==="
find . -name "*.md" -type f | head -10
```

## Section 2: Local Development Setup

### 2.1 Dependency Installation

```bash
# Install project dependencies
echo "=== Installing Documentation Dependencies ==="

# Navigate to project root
cd ~/speckit-curriculum

# Install npm dependencies
npm install

# If using Python tools
pip3 install -r requirements.txt  # if exists

# Verify Docusaurus installation
npx docusaurus --version
```

### 2.2 Local Development Server

```bash
# Start local development server
echo "=== Starting Local Development Server ==="

# Build the site for development
npm run build

# Start development server
npx docusaurus start --port 3000 --host 0.0.0.0

# Verification commands (to run in separate terminal)
echo "=== Verifying Development Server ==="
sleep 5  # Wait for server to start
curl -s -o /dev/null -w "HTTP Status: %{http_code}\n" http://localhost:3000
```

### 2.3 Configuration Validation

```bash
# Validate Docusaurus configuration
echo "=== Validating Docusaurus Configuration ==="

npx docusaurus deploy --dry-run || echo "Configuration validation failed"

# Check site configuration
npx docusaurus swizzle @docusaurus/theme-classic Footer --eject --force
```

## Section 3: Build Process Verification

### 3.1 Production Build

```bash
# Execute production build
echo "=== Starting Production Build ==="

# Clean previous build if exists
rm -rf build/

# Execute build process
npm run build

# Verify build output
echo "=== Build Output Verification ==="
ls -la build/
echo "=== Checking for essential build files ==="
test -f build/index.html && echo "✓ index.html exists" || echo "✗ index.html missing"
test -f build/sitemap.xml && echo "✓ sitemap.xml exists" || echo "✗ sitemap.xml missing"
test -d build/docs && echo "✓ docs/ directory exists" || echo "✗ docs/ directory missing"
```

### 3.2 Build Quality Assurance

#### HTML Validation
```bash
# Validate HTML structure of generated pages
echo "=== HTML Validation ==="

# Install HTML validator globally if not present
npm install -g @html-validate/cli

# Validate key pages
html-validate build/index.html --rules 'element-permitted-content: error' 'element-required-attributes: error' || echo "HTML validation issues found"

html-validate build/docs/intro.html --rules 'element-permitted-content: error' 'element-required-attributes: error' || echo "Intro HTML validation issues found"
```

#### Link Verification
```bash
# Verify internal links are working
echo "=== Link Verification ==="

# Install link checker
npm install -g broken-link-checker

# Check for broken links (excluding external for now)
blc http://localhost:3000 -ro --filter-level 3 || echo "Broken links detected"
```

#### Accessibility Audit
```bash
# Perform accessibility audit
echo "=== Accessibility Audit ==="

# Install axe-cli for accessibility testing
npm install -g @axe-core/cli

# Run accessibility audit on key pages
axe http://localhost:3000 --output axe-results.json --include '#main' || echo "Accessibility issues detected"
cat axe-results.json
```

### 3.3 Content Validation

#### Markdown Validation
```bash
# Validate Markdown structure
echo "=== Markdown Validation ==="

# Install markdownlint-cli
npm install -g markdownlint-cli

# Validate all documentation files
markdownlint docs/**/*.md --config .markdownlint.json || echo "Markdown style issues found"

# Check for common documentation errors
echo "=== Checking Documentation Quality ==="
# Check for missing alt text in images
grep -r '![](' docs/ --include="*.md" || echo "Found images without alt text"

# Check for broken internal links
grep -r ']\((' docs/ --include="*.md" || echo "Found potentially broken relative links"
```

#### Content Completeness Check
```bash
# Validate content completeness
echo "=== Content Completeness Verification ==="

# Check for documentation headers in key files
for file in docs/module1-ros2/*.md docs/module2-simulation/*.md docs/module3-ai/*.md docs/module4-vla/*.md; do
    if [ -f "$file" ]; then
        echo "Checking $file"
        head -20 "$file" | grep -q "^\#" && echo "  ✓ Has header" || echo "  ✗ Missing header"
        grep -q "##.*Overview\|##.*Introduction" "$file" && echo "  ✓ Has overview/introduction" || echo "  ✗ Missing overview/introduction"
    fi
done
```

## Section 4: Deployment Preparation

### 4.1 Build Optimization

```bash
# Optimize build for deployment
echo "=== Optimizing Build for Deployment ==="

# Analyze bundle size
npx docusaurus build
npx webpack-bundle-analyzer build --mode static --no-open || echo "Bundle analyzer failed"

# Check build size
BUILD_SIZE=$(du -sh build/ | cut -f1)
echo "Build size: $BUILD_SIZE"
if [ "$(echo "$BUILD_SIZE" | sed 's/[^0-9.]//g' | awk '{print int($1)}')" -gt 100 ]; then
    echo "⚠ Warning: Build size exceeds 100MB"
else
    echo "✓ Build size is acceptable"
fi
```

### 4.2 Deployment Configuration Verification

```bash
# Verify deployment configuration
echo "=== Deployment Configuration ==="

# Check if deployment configuration exists
if [ -f "docusaurus.config.js" ]; then
    echo "✓ docusaurus.config.js exists"
    
    # Verify deployment URL is configured
    grep -q "baseUrl" docusaurus.config.js && echo "✓ baseUrl configured" || echo "✗ baseUrl missing"
    
    # Verify site URL is configured
    grep -q "url" docusaurus.config.js && echo "✓ url configured" || echo "✗ url missing"
    
    # Verify organization and project names
    grep -q "organizationName" docusaurus.config.js && echo "✓ organizationName configured" || echo "✗ organizationName missing"
    grep -q "projectName" docusaurus.config.js && echo "✓ projectName configured" || echo "✗ projectName missing"
else
    echo "✗ docusaurus.config.js not found"
    exit 1
fi
```

### 4.3 Pre-deployment Testing

```bash
# Run pre-deployment tests
echo "=== Pre-deployment Testing ==="

# Start local server to test production build
npx serve -s build &

# Wait for server to start
sleep 3
SERVER_PID=$!

# Perform basic tests
echo "Testing home page..."
curl -f http://localhost:3000/ > /dev/null && echo "✓ Home page loads" || echo "✗ Home page failed"

echo "Testing documentation pages..."
curl -f http://localhost:3000/docs/intro > /dev/null && echo "✓ Intro page loads" || echo "✗ Intro page failed"

# Kill server process
kill $SERVER_PID

echo "=== Pre-deployment Health Check Complete ==="
```

## Section 5: Deployment Verification

### 5.1 GitHub Pages Deployment

```bash
# GitHub Pages deployment script
echo "=== GitHub Pages Deployment ==="

# Verify Git configuration
git remote -v

# Check if we're on the correct branch
CURRENT_BRANCH=$(git branch --show-current)
echo "Current branch: $CURRENT_BRANCH"

# Execute deployment (assuming GitHub Actions or similar CI/CD setup)
# The actual deployment would typically be handled by CI/CD
echo "Deployment would execute: npm run deploy"
echo "This pushes build/ directory to GitHub Pages branch"
```

### 5.2 Deployment Verification Checklist

#### Post-Deployment Validation
```bash
# Post-deployment verification script
echo "=== Post-Deployment Verification ==="

# Set the deployment URL (adjust as needed)
DEPLOYMENT_URL="https://speckit.github.io/ai-hackathon-book"

# Test primary pages
test_page() {
    local path="$1"
    local name="$2"
    local url="${DEPLOYMENT_URL}${path}"
    
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$url")
    if [ "$HTTP_CODE" -eq 200 ]; then
        echo "✓ $name (${path}): Status $HTTP_CODE"
        return 0
    else
        echo "✗ $name (${path}): Status $HTTP_CODE"
        return 1
    fi
}

# Test core pages
test_page "/" "Home Page"
test_page "/docs/intro" "Documentation Introduction"
test_page "/docs/module1-ros2/" "Module 1 Landing"
test_page "/docs/module2-simulation/" "Module 2 Landing"
test_page "/docs/module3-ai/" "Module 3 Landing"
test_page "/docs/module4-vla/" "Module 4 Landing"

# Test sitemap
curl -f "${DEPLOYMENT_URL}/sitemap.xml" > /dev/null && echo "✓ Sitemap accessible" || echo "✗ Sitemap not accessible"

# Test robots.txt
curl -f "${DEPLOYMENT_URL}/robots.txt" > /dev/null && echo "✓ robots.txt accessible" || echo "✗ robots.txt not accessible"
```

### 5.3 Functional Testing of Deployed Site

#### Navigation and Search Testing
```bash
# Test site navigation and functionality
echo "=== Functional Testing of Deployed Site ==="

# Test navigation structure
NAV_TESTS=(
    "/docs/intro"
    "/docs/category/module-1-ros-2-fundamentals"
    "/docs/module1-ros2/"
    "/docs/module2-simulation/"
    "/docs/module3-ai/"
    "/docs/module4-vla/"
)

for test_path in "${NAV_TESTS[@]}"; do
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "${DEPLOYMENT_URL}${test_path}")
    if [ "$HTTP_CODE" -eq 200 ]; then
        echo "✓ Navigation: ${test_path} - Status $HTTP_CODE"
    else
        echo "✗ Navigation: ${test_path} - Status $HTTP_CODE"
    fi
done

# Test search functionality (if available)
echo "Testing search endpoint..."
SEARCH_ENDPOINT="${DEPLOYMENT_URL}/api/search"
if curl -f "$SEARCH_ENDPOINT" > /dev/null 2>&1; then
    echo "✓ Search API accessible"
else
    echo "? Search API status unknown (may not be configured)"
fi
```

#### Performance Testing
```bash
# Basic performance testing
echo "=== Performance Testing ==="

# Test page load times
PAGE_LOAD_TESTS=(
    "/"
    "/docs/intro"
    "/docs/module1-ros2/"
)

for page in "${PAGE_LOAD_TESTS[@]}"; do
    START_TIME=$(date +%s.%N)
    curl -s -o /dev/null "${DEPLOYMENT_URL}${page}"
    END_TIME=$(date +%s.%N)
    LOAD_TIME=$(echo "$END_TIME - $START_TIME" | bc)
    
    if (( $(echo "$LOAD_TIME < 5.0" | bc -l) )); then
        echo "✓ ${page}: Loaded in ${LOAD_TIME}s"
    else
        echo "⚠ ${page}: Slow load time (${LOAD_TIME}s)"
    fi
done
```

## Section 6: Content Integrity Verification

### 6.1 Asset Verification

```bash
# Verify all assets are properly deployed
echo "=== Asset Verification ==="

# Check CSS loads properly
CSS_CHECK=$(curl -s -o /dev/null -w "%{http_code}" "${DEPLOYMENT_URL}/styles.css")
echo "CSS file: Status $CSS_CHECK"

# Check JS bundles
JS_CHECK=$(curl -s -o /dev/null -w "%{http_code}" "${DEPLOYMENT_URL}/assets/js/main.*.js")
echo "JS bundle: Status $JS_CHECK"

# Check images directory structure
IMAGE_COUNT=$(curl -s "${DEPLOYMENT_URL}/docs/assets/img/" | grep -c "\.png\|\.jpg\|\.jpeg\|\.gif\|\.svg")
echo "Images found: $IMAGE_COUNT"
```

### 6.2 Search Index Verification

```bash
# Verify search index is properly built
echo "=== Search Index Verification ==="

# Check if search index exists
if curl -f "${DEPLOYMENT_URL}/search-index.json" > /dev/null 2>&1; then
    echo "✓ Search index exists"
    INDEX_SIZE=$(curl -s "${DEPLOYMENT_URL}/search-index.json" | wc -c)
    echo "Search index size: $INDEX_SIZE bytes"
elif curl -f "${DEPLOYMENT_URL}/assets/js/search-index.*.js" > /dev/null 2>&1; then
    echo "✓ Search index exists (JS format)"
    INDEX_SIZE=$(curl -s "${DEPLOYMENT_URL}/assets/js/search-index.*.js" | wc -c)
    echo "Search index size: $INDEX_SIZE bytes"
else
    echo "? Search index not found (this may be intentional)"
fi
```

### 6.3 Cross-Module Navigation Testing

```bash
# Verify cross-module navigation works
echo "=== Cross-Module Navigation Testing ==="

# Test links between modules
CROSS_LINKS=(
    "${DEPLOYMENT_URL}/docs/module1-ros2/#next-steps"
    "${DEPLOYMENT_URL}/docs/module2-simulation/#prerequisites"
    "${DEPLOYMENT_URL}/docs/module3-ai/#integration"
    "${DEPLOYMENT_URL}/docs/module4-vla/#capstone-project"
)

for link in "${CROSS_LINKS[@]}"; do
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$link")
    if [ "$HTTP_CODE" -eq 200 ]; then
        echo "✓ Cross-link: Status $HTTP_CODE"
    else
        echo "✗ Cross-link: Status $HTTP_CODE ($link)"
    fi
done
```

## Section 7: Automated Verification Scripts

### 7.1 Build Verification Script

```bash
#!/bin/bash
# build-verification.sh - Automated build verification script

set -euo pipefail

echo "========================================="
echo "Physical AI & Humanoid Robotics Curriculum"
echo "Build Verification Script"
echo "========================================="

TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
echo "Verification started at: $TIMESTAMP"

# Configuration
PROJECT_ROOT="${1:-$HOME/speckit-curriculum}"
BUILD_LOG="build-verification-$(date +%Y%m%d_%H%M%S).log"
SUCCESS=true

# Redirect all output to log and terminal
exec > >(tee -a "$BUILD_LOG") 2>&1

echo ""
echo "--- Environment Verification ---"
# Check system requirements
check_prerequisites() {
    local errors=0
    
    if ! command -v node >/dev/null 2>&1; then
        echo "[ERROR] Node.js not installed"
        errors=$((errors + 1))
    else
        NODE_VERSION=$(node -v)
        echo "[OK] Node.js $NODE_VERSION installed"
    fi
    
    if ! command -v npm >/dev/null 2>&1; then
        echo "[ERROR] npm not installed"
        errors=$((errors + 1))
    else
        echo "[OK] npm installed"
    fi
    
    if ! command -v git >/dev/null 2>&1; then
        echo "[ERROR] Git not installed"
        errors=$((errors + 1))
    else
        echo "[OK] Git $(git --version) installed"
    fi
    
    if [ $errors -gt 0 ]; then
        echo "[CRITICAL] $errors prerequisites missing. Exiting."
        exit 1
    fi
}

# Check project structure
check_structure() {
    echo ""
    echo "--- Project Structure Verification ---"
    
    if [ ! -f "$PROJECT_ROOT/package.json" ]; then
        echo "[ERROR] package.json not found in $PROJECT_ROOT"
        SUCCESS=false
        return 1
    else
        echo "[OK] package.json found"
    fi
    
    if [ ! -f "$PROJECT_ROOT/docusaurus.config.js" ]; then
        echo "[ERROR] docusaurus.config.js not found in $PROJECT_ROOT"
        SUCCESS=false
        return 1
    else
        echo "[OK] docusaurus.config.js found"
    fi
    
    if [ ! -d "$PROJECT_ROOT/docs" ]; then
        echo "[ERROR] docs directory not found in $PROJECT_ROOT"
        SUCCESS=false
        return 1
    else
        echo "[OK] docs/ directory found with $(find "$PROJECT_ROOT/docs" -name "*.md" | wc -l) markdown files"
    fi
}

# Execute build process
execute_build() {
    echo ""
    echo "--- Build Process ---"
    
    cd "$PROJECT_ROOT"
    
    # Clean previous build
    if [ -d build ]; then
        echo "Cleaning previous build..."
        rm -rf build/
    fi
    
    # Install dependencies
    echo "Installing dependencies..."
    npm ci
    
    # Execute build
    echo "Starting build process..."
    if npm run build; then
        echo "[OK] Build completed successfully"
    else
        echo "[ERROR] Build failed"
        SUCCESS=false
        return 1
    fi
    
    # Verify build output
    if [ -f "$PROJECT_ROOT/build/index.html" ]; then
        echo "[OK] Build output verified - index.html exists"
    else
        echo "[ERROR] Build output missing - index.html not found"
        SUCCESS=false
        return 1
    fi
}

# Run verifications
check_prerequisites
check_structure
execute_build

echo ""
echo "========================================="
echo "Build Verification Summary"
echo "========================================="

if [ "$SUCCESS" = true ]; then
    echo "✓ ALL CHECKS PASSED"
    echo "Build verification completed successfully"
    exit 0
else
    echo "✗ BUILD VERIFICATION FAILED"
    echo "One or more checks failed. Please review the log: $BUILD_LOG"
    exit 1
fi
```

### 7.2 Deployment Verification Script

```bash
#!/bin/bash
# deployment-verification.sh - Automated deployment verification script

set -euo pipefail

echo "========================================="
echo "Physical AI & Humanoid Robotics Curriculum"
echo "Deployment Verification Script"
echo "========================================="

# Configuration - adjust URLs as needed
DEPLOYMENT_URL="${1:-https://speckit.github.io/ai-hackathon-book}"
HEALTH_CHECK_INTERVAL="${2:-30}"  # seconds
MAX_RETRIES="${3:-5}"
VERIFICATION_LOG="deployment-verification-$(date +%Y%m%d_%H%M%S).log"

echo "Deployment URL: $DEPLOYMENT_URL"
echo "Health check interval: ${HEALTH_CHECK_INTERVAL}s"
echo "Max retries: $MAX_RETRIES"

# Redirect to log and terminal
exec > >(tee -a "$VERIFICATION_LOG") 2>&1

SUCCESS=true
FAILED_CHECKS=0
TOTAL_CHECKS=0

# Increment counters
increment_counters() {
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    if [ "$1" = "FAIL" ]; then
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
        SUCCESS=false
    fi
}

# Generic health check function
health_check() {
    local url="$1"
    local description="$2"
    local expected_status="${3:-200}"
    local retries="${4:-0}"
    
    local attempt=1
    local status_code
    
    while [ $attempt -le $((retries + 1)) ]; do
        status_code=$(curl -s -o /dev/null -w "%{http_code}" "$url" || echo "000")
        
        if [ "$status_code" = "$expected_status" ]; then
            echo "[OK] $description - $url (Status: $status_code)"
            increment_counters "PASS"
            return 0
        elif [ $attempt -le $retries ]; then
            echo "  [TRY $attempt/$retries] $description - Status: $status_code. Retrying in ${HEALTH_CHECK_INTERVAL}s..."
            sleep $HEALTH_CHECK_INTERVAL
            attempt=$((attempt + 1))
        else
            echo "[FAIL] $description - $url (Status: $status_code, Expected: $expected_status)"
            increment_counters "FAIL"
            return 1
        fi
    done
}

# Content verification function
content_check() {
    local url="$1"
    local description="$2"
    local expected_content="$3"
    
    local response
    response=$(curl -s "$url")
    
    if echo "$response" | grep -q "$expected_content"; then
        echo "[OK] $description - Contains expected content"
        increment_counters "PASS"
    else
        echo "[FAIL] $description - Missing expected content: $expected_content"
        increment_counters "FAIL"
    fi
}

echo ""
echo "--- Deployment Health Checks ---"

# Core page availability
health_check "$DEPLOYMENT_URL/" "Home page" "200" "$MAX_RETRIES"
health_check "$DEPLOYMENT_URL/docs/intro" "Documentation intro" "200" "$MAX_RETRIES"
health_check "$DEPLOYMENT_URL/docs/module1-ros2/" "Module 1 landing" "200" "$MAX_RETRIES"

# Static assets
health_check "$DEPLOYMENT_URL/styles.css" "CSS file" "200" "$MAX_RETRIES"
health_check "$DEPLOYMENT_URL/manifest.json" "Manifest file" "200" "$MAX_RETRIES"

# API endpoints (if applicable)
health_check "$DEPLOYMENT_URL/sitemap.xml" "Sitemap" "200" "$MAX_RETRIES"
health_check "$DEPLOYMENT_URL/robots.txt" "Robots.txt" "200" "$MAX_RETRIES"

echo ""
echo "--- Content Verification ---"

# Verify key content exists on important pages
content_check "$DEPLOYMENT_URL/" "Home page content" "Physical AI &amp; Humanoid Robotics"
content_check "$DEPLOYMENT_URL/docs/intro" "Intro page content" "curriculum"
content_check "$DEPLOYMENT_URL/docs/module1-ros2/" "Module 1 content" "ROS 2"

echo ""
echo "--- Performance Testing ---"

# Test load times
PERFORMANCE_PAGES=(
    "$DEPLOYMENT_URL/"
    "$DEPLOYMENT_URL/docs/intro"
)

for page in "${PERFORMANCE_PAGES[@]}"; do
    local start_time end_time load_time
    
    start_time=$(date +%s.%N)
    curl -s -o /dev/null "$page"
    end_time=$(date +%s.%N)
    
    load_time=$(echo "$end_time - $start_time" | bc)
    
    if (( $(echo "$load_time < 5.0" | bc -l) )); then
        echo "[OK] $page loaded in ${load_time}s"
    else
        echo "[WARN] $page slow load time (${load_time}s)"
    fi
done

echo ""
echo "========================================="
echo "Deployment Verification Summary"
echo "========================================="
echo "Total checks performed: $TOTAL_CHECKS"
echo "Failed checks: $FAILED_CHECKS"
echo "Passed checks: $((TOTAL_CHECKS - FAILED_CHECKS))"

if [ "$SUCCESS" = true ]; then
    echo ""
    echo "✓ ALL CHECKS PASSED"
    echo "Deployment verification completed successfully"
    echo "Log saved to: $VERIFICATION_LOG"
    exit 0
else
    echo ""
    echo "✗ DEPLOYMENT VERIFICATION FAILED"
    echo "$FAILED_CHECKS out of $TOTAL_CHECKS checks failed"
    echo "Please review the log: $VERIFICATION_LOG"
    exit 1
fi
```

## Section 8: Continuous Integration Setup

### 8.1 GitHub Actions Workflow

```yaml
# .github/workflows/documentation.yml
name: Documentation Build and Verification

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build-and-verify:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '18'
        cache: 'npm'
    
    - name: Install Dependencies
      run: |
        npm ci
        
    - name: Build Documentation
      run: |
        npm run build
      
    - name: Upload Build Artifacts
      uses: actions/upload-artifact@v3
      with:
        name: documentation-build
        path: build/
        retention-days: 30
    
    - name: Run Build Verification Script
      run: |
        chmod +x scripts/build-verification.sh
        ./scripts/build-verification.sh ${{ github.workspace }}
    
    - name: Verify Content Integrity
      run: |
        # Check for critical files
        if [ ! -f "build/index.html" ]; then
          echo "Critical error: index.html not found in build"
          exit 1
        fi
        if [ ! -f "build/sitemap.xml" ]; then
          echo "Warning: sitemap.xml not found"
        fi
        
        # Count documentation pages
        DOC_COUNT=$(find build/docs -name "*.html" | wc -l)
        echo "Built $DOC_COUNT documentation pages"
        if [ "$DOC_COUNT" -lt 50 ]; then
          echo "Warning: Only $DOC_COUNT pages built, expected more"
        fi
```

### 8.2 Deployment Workflow

```yaml
# .github/workflows/deploy.yml
name: Deploy Documentation

on:
  push:
    branches: [ main ]

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: "pages"
  cancel-in-progress: true

jobs:
  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v3
      
      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
          cache: 'npm'
      
      - name: Install Dependencies
        run: npm ci
      
      - name: Build Documentation
        run: npm run build
      
      - name: Setup Pages
        uses: actions/configure-pages@v3
      
      - name: Upload to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v2
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: build/
      
      - name: Verify Deployment
        run: |
          chmod +x scripts/deployment-verification.sh
          ./scripts/deployment-verification.sh "${{ steps.deployment.outputs.page_url }}" "10" "3"
```

## Section 9: Monitoring and Alerting

### 9.1 Site Uptime Monitoring

```bash
# uptime-monitor.sh - Site monitoring script
#!/bin/bash

SITE_URL="https://speckit.github.io/ai-hackathon-book"
LOG_FILE="/var/log/doc-site-monitor.log"
HEALTH_ENDPOINT="${SITE_URL}/"
EMAIL_ALERTS=false  # Set to true to enable email alerts

# Log function
log_message() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] $message" | tee -a "$LOG_FILE"
}

# Check site status
check_site() {
    local status_code
    status_code=$(curl -s -o /dev/null -w "%{http_code}" "$HEALTH_ENDPOINT")
    
    if [ "$status_code" -eq 200 ]; then
        log_message "Site is UP (Status: $status_code)"
        return 0
    else
        log_message "Site is DOWN (Status: $status_code)"
        # Add alerting logic here
        return 1
    fi
}

# Run continuous monitoring
while true; do
    check_site
    sleep 300  # Check every 5 minutes
done
```

### 9.2 Performance Monitoring

```javascript
// performance-monitor.js - Frontend performance monitoring snippet
// To be added to documentation pages for real-world performance data

(function() {
    // Track Core Web Vitals and other performance metrics
    const perfObserver = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
            // Log performance entries
            console.log(`Performance Entry: ${entry.name} - ${entry.duration}ms`);
            
            // Track Largest Contentful Paint (LCP)
            if (entry.entryType === 'largest-contentful-paint' && entry.startTime) {
                console.log(`LCP: ${entry.startTime}ms`);
            }
            
            // Track First Input Delay (FID)
            if (entry.entryType === 'first-input' && entry.processingStart) {
                const fid = entry.processingStart - entry.startTime;
                console.log(`FID: ${fid}ms`);
            }
        }
    });
    
    try {
        perfObserver.observe({ entryTypes: ['largest-contentful-paint', 'first-input'] });
    } catch (e) {
        console.warn('Performance Observer not supported:', e);
    }
    
    // Track page load performance
    window.addEventListener('load', function() {
        const loadTime = window.performance.timing.loadEventEnd - window.performance.timing.navigationStart;
        console.log(`Page Load Time: ${loadTime}ms`);
        
        // Send performance metrics to analytics (if available)
        if (typeof gtag !== 'undefined') {
            gtag('event', 'timing_complete', {
                'name': 'page_load',
                'value': loadTime,
                'event_category': 'performance'
            });
        }
    });
})();
```

## Section 10: Troubleshooting and Resolution

### 10.1 Common Build Issues and Solutions

```bash
# Troubleshooting script - Common build issues and diagnostics

echo "=== Documentation Build Troubleshooting ==="

# Check for common issues
echo ""
echo "1. Checking Node.js version compatibility..."
NODE_VERSION=$(node -v | sed 's/v//' | cut -d. -f1)
if [ "$NODE_VERSION" -lt 16 ]; then
    echo "⚠ Node.js version might be too old (< 16): $(node -v)"
    echo "Consider upgrading to Node.js 18 or later"
else
    echo "✓ Node.js version is compatible: $(node -v)"
fi

echo ""
echo "2. Checking disk space..."
FREE_SPACE=$(df -h . | awk 'NR==2 {print $4}' | sed 's/[^0-9]//g')
if [ "$FREE_SPACE" -lt 1 ]; then
    echo "⚠ Low disk space: less than 1GB available"
else
    echo "✓ Sufficient disk space available"
fi

echo ""
echo "3. Checking memory usage during build..."
# This would need to run during actual build process
# Adding to monitoring during build

echo ""
echo "4. Verifying npm cache..."
npm cache verify

echo ""
echo "5. Checking for conflicting dependencies..."
# Check for duplicate packages
DUPLICATE_PKGS=$(npm ls --depth=0 2>&1 | grep -c "UNMET" || true)
if [ "$DUPLICATE_PKGS" -gt 0 ]; then
    echo "⚠ Found $DUPLICATE_PKGS unmet dependencies"
    npm ls --depth=0
else
    echo "✓ No unmet dependencies found"
fi

echo ""
echo "6. Testing simple build..."
cd ~/speckit-curriculum
echo "Creating minimal config for test..." > temp-test.md
if npm run build > /dev/null 2>&1; then
    echo "✓ Basic build test passed"
    rm -f temp-test.md
else
    echo "✗ Basic build test failed"
    echo "Check package.json scripts and dependencies"
fi
```

### 10.2 Deployment Issue Resolution

#### Common Deployment Problems
```bash
# Deployment troubleshooting checklist

echo "=== Deployment Troubleshooting Checklist ==="

echo ""
echo "1. Verify GitHub Pages is enabled in repository settings"
echo "   Settings → Pages → Source should be 'Deploy from a branch'"
echo "   Branch should be 'gh-pages' and folder '(/root)'"

echo ""
echo "2. Check that the site builds successfully locally:"
echo "   npm run build"
echo "   npx serve -s build"

echo ""
echo "3. Verify docusaurus.config.js has correct deployment settings:"
echo "   url: 'https://speckit.github.io'"
echo "   baseUrl: '/ai-hackathon-book/'"
echo "   organizationName: 'speckit'"
echo "   projectName: 'ai-hackathon-book'"

echo ""
echo "4. Check that the deploy command is configured:"
echo "   In package.json, 'deploy' script should exist"
echo "   Should run: docusaurus deploy"
```

## Conclusion

This comprehensive verification process ensures that the Physical AI & Humanoid Robotics curriculum documentation is built properly, deployed successfully, and remains available to students and educators. Regular execution of these verification procedures will maintain the quality and availability of the curriculum materials.

The automated scripts provided can be integrated into CI/CD pipelines to ensure continuous verification of the build and deployment processes, catching potential issues before they affect end users.