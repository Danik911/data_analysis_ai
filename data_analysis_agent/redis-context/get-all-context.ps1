# =============================
# get-all-context.ps1
# Purpose: Retrieve all key-value pairs from Redis matching a pattern, format for LLMs, and optionally save to file.
# Usage (PowerShell):
#   .\get-all-context.ps1 [-pattern "*"] [-formatForLLM] [-saveToFile] [-outputFile "filename.txt"]
# Prerequisites: Docker must be running and able to access a Redis container.
# Notes: This script is for PowerShell. Do not use & or && as in bash.
# =============================

param(
    [Parameter(Mandatory=$false)]
    [string]$pattern = "*",

    [Parameter(Mandatory=$false)]
    [switch]$formatForLLM = $true,

    [Parameter(Mandatory=$false)]
    [switch]$saveToFile = $false,

    [Parameter(Mandatory=$false)]
    [string]$outputFile = "redis-context.txt"
)

# Function to escape quotes in Redis values for better display
function Escape-RedisString {
    param([string]$str)
    return $str -replace '"', '\"'
}

# Retrieve all keys from Redis that match the pattern
Write-Host "Retrieving all Redis keys matching pattern: '$pattern'"
$keys = docker run --rm redis:latest redis-cli -h host.docker.internal -p 6379 KEYS $pattern
if (-not $keys) {
    Write-Host "No keys found in Redis matching pattern: '$pattern'"
    exit
}

# Convert string output to array (if it's a single line, make it an array)
if ($keys -is [string]) {
    $keys = @($keys)
}

Write-Host "Found $($keys.Count) keys in Redis"

# Initialize results collection
$results = @()

# Retrieve each key's value and format it
foreach ($key in $keys) {
    # Skip empty lines
    if ([string]::IsNullOrWhiteSpace($key)) {
        continue
    }
    
    # Get the value of the key
    $value = docker run --rm redis:latest redis-cli -h host.docker.internal -p 6379 GET $key
    
    if ($formatForLLM) {
        # Format for LLM context
        $contextItem = @{
            Key = $key
            Value = $value
        }
        $results += $contextItem
    } else {
        # Simple console display
        Write-Host "Key: $key"
        Write-Host "Value: $value"
        Write-Host "----------------------------------------"
    }
}

# Output formatted context
if ($formatForLLM) {
    $contextOutput = @"
### Redis Context Information
Total keys: $($results.Count)

"@

    # Add each key-value pair with proper formatting
    foreach ($item in $results) {
        $safeValue = Escape-RedisString $item.Value
        $contextOutput += @"
#### Key: $($item.Key)
$safeValue

"@
    }

    if ($saveToFile) {
        # Save to file
        Set-Content -Path $outputFile -Value $contextOutput
        Write-Host "Saved formatted context to $outputFile"
    } else {
        # Display in console
        Write-Host $contextOutput
    }
}

# Return object data for piping
if ($formatForLLM) {
    return $results
}