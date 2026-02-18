#!/bin/bash
# Setup script to create .env file from GitHub secrets

echo "🔧 Setting up .env file..."

# Get secrets from GitHub (requires gh CLI)
TELEGRAM_TOKEN_VAL=$(gh secret list --json name,createdAt | jq -r '.[] | select(.name=="TELEGRAM_TOKEN") | .name')
TELEGRAM_CHAT_ID_VAL=$(gh secret list --json name,createdAt | jq -r '.[] | select(.name=="TELEGRAM_CHAT_ID") | .name')

# Note: GitHub secrets are write-only, so we need to prompt user for values
echo ""
echo "⚠️  GitHub secrets are write-only and cannot be read."
echo "Please provide your credentials:"
echo ""

read -p "Enter your TELEGRAM_TOKEN: " TELEGRAM_TOKEN
read -p "Enter your TELEGRAM_CHAT_ID: " TELEGRAM_CHAT_ID
read -p "Enter your GITHUB_TOKEN: " GITHUB_TOKEN

# Create .env file
cat > .env << EOF
# Telegram Bot Configuration
TELEGRAM_TOKEN=$TELEGRAM_TOKEN
TELEGRAM_CHAT_ID=$TELEGRAM_CHAT_ID

# GitHub Configuration
GITHUB_TOKEN=$GITHUB_TOKEN
EOF

echo ""
echo "✅ .env file created successfully!"
echo "🚀 You can now run: python3 telegram_bot.py"
