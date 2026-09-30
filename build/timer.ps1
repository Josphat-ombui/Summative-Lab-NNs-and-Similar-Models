param([int]$Probe = 1)
$pidFile = Join-Path $PSScriptRoot '..\full_run.pid'
$startTs  = Join-Path $PSScriptRoot '..\full_run.start_ts'
$log      = Join-Path $PSScriptRoot '..\full_run_log.txt'
$errLog   = Join-Path $PSScriptRoot '..\full_run_err.txt'
$start    = Get-Date (Get-Content $startTs -ErrorAction SilentlyContinue)
$p = Get-Process -Id (Get-Content $pidFile -ErrorAction SilentlyContinue) -ErrorAction SilentlyContinue
$now = Get-Date
$el = $now - $start
$alive = $null -ne $p
$cpu = if ($alive) { [math]::Round(($p.CPU)/60.0, 1) } else { 0 }
$stage = ''
$lines = if (Test-Path $log) { Get-Content $log -Tail 14 } else { @() }
$lastLine = ($lines | Select-Object -Last 1)

# crude stage map based on the last meaningful log line
$t = ($lines -join "`n")
if ($t -match 'CNN model saved|Loaded final CNN for evaluation') { $stage = 'Part 2 done (CNN trained)' }
elseif ($t -match 'Fine-tuned val_accuracy|unfreezing|Unfreezing') { $stage = 'Part 2.3 fine-tuning CNN' }
elseif ($t -match 'Training the classification head|Stage 1') { $stage = 'Part 2.3 head warm-up' }
elseif ($t -match 'headC|headB|headA|head sweep|Best head variant') { $stage = 'Part 2.2 head sweep' }
elseif ($t -match 'Extracting frozen|Feature extraction|Loaded cached ImageNet') { $stage = 'Part 2.1 feature extraction' }
elseif ($t -match 'policy|split|Image split|pipeline') { $stage = 'Part 1 pipelines' }
elseif ($t -match 'cell|OK|FAILED') { $stage = 'finalising' }
if (-not $stage -and -not $alive) { $stage = 'finished/exited' }

# estimate: 2.2h nominal budget
$budgetMin = 132
$elMin = $el.TotalMinutes
$rem = ($budgetMin - $elMin)
$remStr = if ($rem -gt 0) { '{0:00}:{1:00}' -f [int]($rem/60), [int]($rem%60) } else { '00:00' }
Write-Host ('[{0}] elapsed {1:00}:{2:00}:{3:00} | ETA ~{4} | cpu {5} min | stage: {6}' -f
    $now.ToString('HH:mm:ss'), $el.Hours, $el.Minutes, $el.Seconds, $remStr, $cpu, $stage)
if ($alive) { Write-Host ('   PID {0}  (running)' -f $p.Id) } else { Write-Host '   (process finished / not running)' }
if ($lastLine) { Write-Host ('   last log line : {0}' -f $lastLine) }