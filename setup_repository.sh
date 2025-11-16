#!/bin/bash
# Git repository initialization script for Eksperimen_SML_Lukas

echo "Initializing Git repository for Eksperimen_SML_Lukas..."

# Initialize git repository
git init

# Create .gitignore file
cat > .gitignore << EOL
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyCharm
.idea/

# Jupyter Notebook
.ipynb_checkpoints

# pyenv
.python-version

# Environment variables
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# VSCode
.vscode/

# macOS
.DS_Store

# Temporary files
*.tmp
*.temp
EOL

# Add all files
git add .

# Initial commit
git commit -m "Initial commit: Diabetes ML preprocessing project with automated pipeline"

echo "Repository initialized successfully!"
echo ""
echo "Next steps:"
echo "1. Create a GitHub repository named 'Eksperimen_SML_Lukas'"
echo "2. Connect your local repository to GitHub:"
echo "   git remote add origin https://github.com/YOUR_USERNAME/Eksperimen_SML_Lukas.git"
echo "3. Push to GitHub:"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "The GitHub Actions workflow will automatically run when you push changes!"
