# ============================================
# run_multi_seed_ablation.ps1
# Runs ablation for 3 Qwen models x 5 math ratios x 4 seeds
# Automatically SKIPS models whose output folder already exists.
# ============================================

$python = "python"
$script = "run_ablation_experiment.py"

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

    if (Test-Path $modelDir) {
        Write-Host "Skipping $modelName - data already exists at $modelDir"
        continue
    }

    Write-Host "==============================================="
    Write-Host "Running ablation for model: $modelName"
    Write-Host "==============================================="

    foreach ($ratio in $ratios) {
        foreach ($seed in $seeds) {

            $outDir = "$modelDir/math_${ratio}_seed${seed}"

            Write-Host ""
            Write-Host ">>> Running: model=$model ratio=$ratio seed=$seed"
            Write-Host "    Output: $outDir"

            & $python $script `
                --math-ratio $ratio `
                --seed $seed `
                --out $outDir `
                --llm $model

            if ($LASTEXITCODE -ne 0) {
                Write-Host "Error running experiment for $model ratio=$ratio seed=$seed"
            }
        }
    }
}

Write-Host ""
Write-Host "All ablation runs completed."