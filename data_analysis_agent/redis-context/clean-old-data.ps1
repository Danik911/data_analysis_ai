# =============================
# clean-old-data.ps1
# Purpose: Remove Redis keys older than 7 days, based on key naming or TTL estimation.
# Usage (PowerShell):
#   .\clean-old-data.ps1
# Prerequisites: Docker must be running and able to access a Redis container. Keys should include a date or timestamp for best results.
# Notes: This script is for PowerShell. Do not use & or && as in bash. Prompts for confirmation before deleting.
# =============================

# Script to clean Redis data older than a week

# Function to check if a string is a valid date
function Test-DateFormat {
    param (
        [Parameter(Mandatory=$true)]
        [string]$DateString
    )
    
    try {
        $null = [DateTime]::ParseExact($DateString, "yyyyMMdd", $null)
        return $true
    } catch {
        return $false
    }
}

# Ask for confirmation
$confirmation = Read-Host "This will remove data older than 7 days from your Redis database. Continue? (yes/no)"
if ($confirmation -ne "yes") {
    Write-Host "Operation cancelled." -ForegroundColor Yellow
    exit
}

# Get current date for comparison
$currentDate = Get-Date
$cutoffDate = $currentDate.AddDays(-7)

Write-Host "Searching for keys older than $($cutoffDate.ToString('yyyy-MM-dd'))" -ForegroundColor Cyan

# First, scan for all keys
$keysResult = $(docker run --rm redis:latest redis-cli -h host.docker.internal -p 6379 KEYS "*")
$keys = $keysResult -split "`n"

$totalKeys = $keys.Count
$removedKeys = 0

# Check each key for timestamp in name or metadata
foreach ($key in $keys) {
    $key = $key.Trim()
    
    if ($key -eq "") { continue }
    
    # Check for date pattern in key name
    # Assuming keys with dates follow pattern: prefix_YYYYMMDD_suffix
    $keyParts = $key -split "_"
    $foundDate = $false
    
    foreach ($part in $keyParts) {
        if ($part.Length -eq 8 -and (Test-DateFormat $part)) {
            $dateStr = $part
            $keyDate = [DateTime]::ParseExact($dateStr, "yyyyMMdd", $null)
            
            if ($keyDate -lt $cutoffDate) {
                $foundDate = $true
                break
            }
        }
    }
    
    # Also check keys with timestamp in their name (unix timestamp)
    if (-not $foundDate -and $key -match "\d{10}") {
        $timestampMatch = $key | Select-String -Pattern "\d{10}"
        $timestamp = $timestampMatch.Matches[0].Value
        
        # Convert Unix timestamp to DateTime
        $keyDate = (Get-Date "1970-01-01").AddSeconds([long]$timestamp)
        
        if ($keyDate -lt $cutoffDate) {
            $foundDate = $true
        }
    }
    
    # If it's an old key based on naming convention, delete it
    if ($foundDate) {
        $deleteResult = docker run --rm redis:latest redis-cli -h host.docker.internal -p 6379 DEL $key
        
        if ($deleteResult -eq "1" -or $deleteResult -eq "0") {
            Write-Host "Deleted old key: $key (created on $($keyDate.ToString('yyyy-MM-dd')))" -ForegroundColor Green
            $removedKeys++
        } else {
            Write-Host "Error deleting key ${key}: $deleteResult" -ForegroundColor Red
        }
    }
    
    # Check TTL for keys without date in name
    # Keys with -1 have no expiration, keys with -2 don't exist
    $ttlResult = docker run --rm redis:latest redis-cli -h host.docker.internal -p 6379 TTL $key
    
    # If key has TTL info, check when it was created (approximation)
    if ($ttlResult -match "^\d+$") {
        $ttl = [int]$ttlResult
        # Get type to determine max TTL by type
        $typeResult = docker run --rm redis:latest redis-cli -h host.docker.internal -p 6379 TYPE $key
        
        # Estimate original TTL by type (common defaults)
        $originalTTL = switch -Regex ($typeResult) {
            "string" { 86400 * 30 } # 30 days for strings
            "hash|list|set|zset" { 86400 * 90 } # 90 days for complex data
            default { 86400 * 14 } # 14 days default
        }
        
        # Estimate creation time by subtracting TTL from original TTL
        $estimatedSecondsAgo = $originalTTL - $ttl
        $estimatedKeyDate = $currentDate.AddSeconds(-$estimatedSecondsAgo)
        
        if ($estimatedKeyDate -lt $cutoffDate) {
            $deleteResult = docker run --rm redis:latest redis-cli -h host.docker.internal -p 6379 DEL $key
            
            if ($deleteResult -eq "1" -or $deleteResult -eq "0") {
                Write-Host "Deleted key with old TTL: $key (est. created $($estimatedKeyDate.ToString('yyyy-MM-dd')))" -ForegroundColor Green
                $removedKeys++
            } else {
                Write-Host "Error deleting key ${key}: $deleteResult" -ForegroundColor Red
            }
        }
    }
}

# Report summary
Write-Host "`nCleanup summary:" -ForegroundColor Cyan
Write-Host "Total keys scanned: $totalKeys" -ForegroundColor White
Write-Host "Keys removed: $removedKeys" -ForegroundColor White

if ($removedKeys -eq 0) {
    Write-Host "No keys older than 7 days were found." -ForegroundColor Yellow
}