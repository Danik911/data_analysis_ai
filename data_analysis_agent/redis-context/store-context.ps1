# =============================
# store-context.ps1
# Purpose: Store a key-value pair in Redis, optionally with an expiration time (in seconds).
# Usage (PowerShell):
#   .\store-context.ps1 -key "yourKey" -value "yourValue" [-expireSeconds 3600]
# Prerequisites: 
#   - Docker must be running and able to access a Redis container.
#   - Redis container should be started with: docker run -d --name redis-test -p 6379:6379 redis:latest
#   - Ensure you use 'powershell' (not 'pwsh') if 'pwsh' is not available on your system.
#   - The script connects to host.docker.internal:6379; make sure this is accessible from your OS (Windows).
# Notes: This script is for PowerShell. Do not use & or && as in bash.
# =============================

param(
    [Parameter(Mandatory=$true)]
    [string]$key,
    
    [Parameter(Mandatory=$true)]
    [string]$value,
    
    [Parameter(Mandatory=$false)]
    [int]$expireSeconds = 0
)

# Store the value in Redis
if ($expireSeconds -gt 0) {
    # Set with expiration
    $result = docker run --rm redis:latest redis-cli -h host.docker.internal -p 6379 SETEX $key $expireSeconds $value
} else {
    # Set without expiration
    $result = docker run --rm redis:latest redis-cli -h host.docker.internal -p 6379 SET $key $value
}

# Report the result
if ($result -eq "OK") {
    Write-Host "Successfully stored value for key: $key"
    if ($expireSeconds -gt 0) {
        Write-Host "This key will expire in $expireSeconds seconds"
    }
} else {
    Write-Host "Error storing value: $result"
}