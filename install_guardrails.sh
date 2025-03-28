#!/bin/bash

# Ensure the required environment variable is set
if [ -z "$GUARDRAILS_API_KEY" ]; then
  echo "Error: GUARDRAILS_API_KEY is not set."
  exit 1
fi

# Configure Guardrails
guardrails configure --token "$GUARDRAILS_API_KEY" --disable-metrics --disable-remote-inferencing
if [ $? -ne 0 ]; then
  echo "Error: Failed to configure Guardrails."
  exit 1
fi

echo "Installing Guardrails hub components..."

# Install Guardrails hub modules
HUB_MODULES=(
  "hub://guardrails/nsfw_text"
  "hub://scb-10x/correct_language"
)

for module in "${HUB_MODULES[@]}"; do
  guardrails hub install "$module"
  if [ $? -ne 0 ]; then
    echo "Error: Failed to install $module."
    exit 1
  fi
done

echo "Guardrails hub components installed successfully."