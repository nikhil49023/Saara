# Saara CLI Launcher for PowerShell
# Usage: .\saara.ps1 run
# Or: .\saara.ps1 distill document.md --type reasoning

param(
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$Arguments
)

# Suppress Python warnings for cleaner output
$env:PYTHONWARNINGS = "ignore"

# Run the CLI
python -m saara.cli @Arguments
