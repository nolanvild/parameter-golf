# sweep_mlp_power.ps1 — MLP nonlinearity sweep: x^1 through x^5, 3 seeds each
# Run from the parameter-golf directory with the venv activated:
#   .venv\Scripts\activate
#   .\sweep_mlp_power.ps1

$powers = 1, 2, 3, 4, 5
$seeds  = 1337, 42, 123

foreach ($power in $powers) {
    foreach ($seed in $seeds) {
        $runId = "power${power}_seed${seed}"
        Write-Host "=== $runId ===" -ForegroundColor Cyan

        $env:RUN_ID               = $runId
        $env:MLP_POWER            = "$power"
        $env:SEED                 = "$seed"
        $env:ITERATIONS           = "500"
        $env:TRAIN_BATCH_TOKENS   = "4096"
        $env:TRAIN_SEQ_LEN        = "256"
        $env:VAL_BATCH_SIZE       = "4096"
        $env:MAX_VAL_TOKENS       = "65536"
        $env:VAL_LOSS_EVERY       = "0"
        $env:WARMUP_STEPS         = "2"
        $env:WARMDOWN_ITERS       = "5"
        $env:TRAIN_LOG_EVERY      = "5"
        $env:MAX_WALLCLOCK_SECONDS = "120"
        $env:NUM_LAYERS           = "3"
        $env:MODEL_DIM            = "128"
        $env:NUM_HEADS            = "4"
        $env:NUM_KV_HEADS         = "2"

        python train_gpt.py
    }
}

# --- Summary table ---
Write-Host "`n=== RESULTS ===" -ForegroundColor Yellow
Write-Host ("{0,-28} {1,10} {2,10}" -f "run", "val_bpb", "val_loss")
Write-Host ("-" * 52)

foreach ($power in $powers) {
    $bpbs = @()
    foreach ($seed in $seeds) {
        $runId   = "power${power}_seed${seed}"
        $logFile = "logs/$runId.txt"
        if (Test-Path $logFile) {
            $line = Get-Content $logFile | Select-String "final_int8_zlib_roundtrip val_loss" | Select-Object -Last 1
            if ($line) {
                $bpb  = [regex]::Match($line, "val_bpb:([\d.]+)").Groups[1].Value
                $loss = [regex]::Match($line, "val_loss:([\d.]+)").Groups[1].Value
                Write-Host ("{0,-28} {1,10} {2,10}" -f $runId, $bpb, $loss)
                if ($bpb) { $bpbs += [double]$bpb }
            }
        } else {
            Write-Host ("{0,-28} {1,10}" -f $runId, "MISSING") -ForegroundColor Red
        }
    }
    if ($bpbs.Count -gt 0) {
        $avg = ($bpbs | Measure-Object -Average).Average
        Write-Host ("{0,-28} {1,10}" -f "  power=$power avg ($($bpbs.Count)/3)", ("{0:F4}" -f $avg)) -ForegroundColor Green
    }
    Write-Host ""
}
