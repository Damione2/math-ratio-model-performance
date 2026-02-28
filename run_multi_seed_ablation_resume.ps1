# run_multi_seed_ablation_resume.ps1
# Resume-safe ablation driver - skips existing folders

$python = "python"
$script = "run_ablation_experiment.py"
$validator = "validate_and_record.py"

$models = @(
    "unsloth/Qwen2.5-1.5B",
    "unsloth/Qwen2.5-Math-1.5B",
    "unsloth/Qwen2.5-Coder-1.5B"
)

$ratios = @(0, 10, 25, 50, 100)
$seeds  = @(0, 1, 2, 3)

$baseOut = "experiments"

foreach ($model in $models) {

    $modelName = $model.Replace("unsloth/", "").Replace("/", "_")
    $modelDir  = "$baseOut/$modelName"

    Write-Host "==============================================="
    Write-Host "Model: $modelName"
    Write-Host "==============================================="

    foreach ($ratio in $ratios) {
        foreach ($seed in $seeds) {

            $outDir = "$modelDir/math_${ratio}_seed${seed}"

            if (Test-Path $outDir) {
                Write-Host "Skipping existing: $outDir"
                continue
            }

            Write-Host ""
            Write-Host ">>> Running: model=$model ratio=$ratio seed=$seed"
            Write-Host "    Output: $outDir"

            New-Item -ItemType Directory -Force -Path $outDir | Out-Null

            & $python $script `
                --math-ratio $ratio `
                --seed $seed `
                --out $outDir `
                --llm $model

            $rc = $LASTEXITCODE
            if ($rc -ne 0) {
                Write-Host "Error running experiment (rc=$rc)"
                continue
            }

            $manifest = Join-Path $outDir "run_manifest.json"
            if (Test-Path $manifest) {
                Write-Host "Run completed. Validating and recording..."
                & $python $validator --run-dir $outDir
                if ($LASTEXITCODE -ne 0) {
                    Write-Host "Validator reported an issue"
                } else {
                    Write-Host "Results recorded"
                }
            } else {
                Write-Host "No run_manifest.json found - skipping record step"
            }
        }
    }
}

Write-Host ""
Write-Host "All scheduled ablation runs processed."