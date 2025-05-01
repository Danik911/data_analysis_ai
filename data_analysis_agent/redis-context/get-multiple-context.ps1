# =============================
# get-multiple-context.ps1
# Purpose: Retrieve values for multiple keys from Redis in a single operation.
# Usage (PowerShell):
#   .\get-multiple-context.ps1 -keys "key1,key2,key3"
#   .\get-multiple-context.ps1 -keys "key1,key2,key3" -format "json"
#   .\get-multiple-context.ps1 -keys "key1,key2,key3" -saveToFile -outputFile "results.json"
# Prerequisites: 
#   - Docker must be running and able to access a Redis container.
#   - Redis container should be started with: docker run -d --name redis-test -p 6379:6379 redis:latest
#   - Ensure you use 'powershell' (not 'pwsh') if 'pwsh' is not available on your system.
#   - The script connects to host.docker.internal:6379; make sure this is accessible from your OS (Windows).
# Notes: This script is for PowerShell. Do not use & or && as in bash; use semicolons (;) instead for command chaining.
# =============================

param(
    [Parameter(Mandatory=$true)]
    [string]$keys,
    
    [Parameter(Mandatory=$false)]
    [ValidateSet("text", "json")]
    [string]$format = "text",
    
    [Parameter(Mandatory=$false)]
    [switch]$saveToFile = $false,
    
    [Parameter(Mandatory=$false)]
    [string]$outputFile = "redis-keys-output.json"
)

# Convert comma-separated keys to array
$keyArray = $keys -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" }

# Check if we have any valid keys
if ($keyArray.Count -eq 0) {
    Write-Host "No valid keys provided. Please specify at least one key." -ForegroundColor Red
    exit 1
}

Write-Host "Retrieving values for $($keyArray.Count) keys from Redis..." -ForegroundColor Cyan

# Initialize results collection
$results = @{}
$foundCount = 0
$missingCount = 0

# Retrieve each key's value
foreach ($key in $keyArray) {
    # Get the value from Redis
    $value = docker run --rm redis:latest redis-cli -h host.docker.internal -p 6379 GET $key
    
    # Store result
    if ($value) {
        $results[$key] = $value
        $foundCount++
    } else {
        $results[$key] = $null
        $missingCount++
    }
}

# Output results based on format
if ($format -eq "json") {
    # Convert to JSON format
    $jsonOutput = ConvertTo-Json $results -Depth 3
    
    if ($saveToFile) {
        $jsonOutput | Set-Content -Path $outputFile
        Write-Host "Results saved to $outputFile" -ForegroundColor Green
    } else {
        Write-Host $jsonOutput
    }
} else {
    # Text output format
    Write-Host "`nResults:" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Gray
    
    foreach ($key in $keyArray) {
        Write-Host "Key: $key" -ForegroundColor Yellow
        if ($results[$key]) {
            Write-Host "Value: $($results[$key])" -ForegroundColor Green
        } else {
            Write-Host "Value: <not found>" -ForegroundColor Red
        }
        Write-Host "----------------------------------------" -ForegroundColor Gray
    }
    
    # Summary
    Write-Host "`nSummary:" -ForegroundColor Cyan
    Write-Host "Found: $foundCount key(s)" -ForegroundColor Green
    Write-Host "Missing: $missingCount key(s)" -ForegroundColor $(if ($missingCount -gt 0) { "Red" } else { "Green" })
}

# Return results object for piping or further processing
return $results