# RL fine-tuning runs compared with the closed-form pi* in the report.
# Safe to rerun: finished runs (with policy.pt) are skipped.
$py = ".\.venv\Scripts\python.exe"
$env:PYTHONPATH = "scripts"
$runs = @(
    @{ out = "checkpoints/rl/ratio_b1_exact"; args = @("--reward", "ratio", "--beta", "1") },
    @{ out = "checkpoints/rl/ratio_b0.3_exact"; args = @("--reward", "ratio", "--beta", "0.3") },
    @{ out = "checkpoints/rl/ratio_b0.3_sampled"; args = @("--reward", "ratio", "--beta", "0.3", "--method", "sampled", "--group", "16") },
    @{ out = "checkpoints/rl/like_b1_exact"; args = @("--reward", "like", "--beta", "1") },
    @{ out = "checkpoints/rl/relisten_b0.3_exact"; args = @("--reward", "relisten", "--beta", "0.3") }
)
foreach ($r in $runs) {
    if (Test-Path "$($r.out)/policy.pt") { continue }
    & $py scripts/rl_train.py @($r.args) --out $r.out
}

if (-not (Test-Path "checkpoints/rl/positive_ce/policy.pt")) { & $py scripts/finetune_positive.py --out checkpoints/rl/positive_ce }
