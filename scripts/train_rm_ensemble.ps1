# Reward-model ensemble: three members initialised from the base GPT-2 on different
# user subsets, plus one member trained from scratch for comparison.
# Safe to rerun: finished members are skipped, interrupted ones continue from last.pt.
$py = ".\.venv\Scripts\python.exe"
$members = @(
    @{ out = "checkpoints/rm/m0"; args = @("--seed", "0") },
    @{ out = "checkpoints/rm/m1"; args = @("--seed", "1") },
    @{ out = "checkpoints/rm/m2"; args = @("--seed", "2") },
    @{ out = "checkpoints/rm/scratch10"; args = @("--seed", "10", "--scratch") }
)
foreach ($m in $members) {
    if (Test-Path "$($m.out)/done") { continue }
    $extra = @()
    if (Test-Path "$($m.out)/last.pt") { $extra = @("--resume") }
    & $py scripts/train_rm.py @($m.args) --epochs 2 --out $m.out @extra
}
