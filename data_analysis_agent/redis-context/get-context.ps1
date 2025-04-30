# =============================
# get-context.ps1
# Purpose: Retrieve the value for a given key from Redis.
# Usage (PowerShell):
#   .\get-context.ps1 -key "yourKey"
# Prerequisites: Docker must be running and able to access a Redis container.
# Notes: This script is for PowerShell. Do not use & or && as in bash.
# =============================

param(
    [Parameter(Mandatory=$true)]
    [string]$key
)

# Retrieve the value from Redis
$result = docker run --rm redis:latest redis-cli -h host.docker.internal -p 6379 GET $key

# Report the result
if ($result) {
    Write-Host "Value for key '$key': $result"
} else {
    Write-Host "No value found for key: $key"
}