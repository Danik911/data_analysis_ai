# =============================
# clear-all-data.ps1
# Purpose: Clear all data from the Redis database (irreversible!).
# Usage (PowerShell):
#   .\clear-all-data.ps1
# Prerequisites: Docker must be running and able to access a Redis container.
# Notes: This script is for PowerShell. Do not use & or && as in bash. Prompts for confirmation before deleting.
# =============================

# Ask for confirmation
$confirmation = Read-Host "This will clear ALL data from your Redis database. Are you sure? (yes/no)"
if ($confirmation -ne "yes") {
    Write-Host "Operation cancelled." -ForegroundColor Yellow
    exit
}

# Execute FLUSHALL command to clear all data
$result = docker run --rm redis:latest redis-cli -h host.docker.internal -p 6379 FLUSHALL

# Report the result
if ($result -eq "OK") {
    Write-Host "Successfully cleared all data from Redis database" -ForegroundColor Green
} else {
    Write-Host "Error clearing data: $result" -ForegroundColor Red
}