param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('humaneval', 'mbpp')]
    [string]$Benchmark,

    [Parameter(Mandatory = $true)]
    [string]$Samples,

    [string]$Output = 'outputs/eval_results.json',
    [string]$Image = 'code-lora-evalplus:0.3.1',
    [int]$Parallel = 4
)

$ErrorActionPreference = 'Stop'
$samplesPath = (Resolve-Path -LiteralPath $Samples).Path
$outputPath = [System.IO.Path]::GetFullPath($Output)
$outputDirectory = Split-Path -Parent $outputPath
$outputName = Split-Path -Leaf $outputPath
New-Item -ItemType Directory -Force $outputDirectory | Out-Null

$dockerArgs = @(
    'run', '--rm',
    '--network', 'none',
    '--read-only',
    '--cap-drop', 'ALL',
    '--security-opt', 'no-new-privileges',
    '--memory', '12g',
    '--cpus', '8',
    '--pids-limit', '512',
    '--tmpfs', '/tmp:rw,noexec,nosuid,size=2g',
    '--mount', "type=bind,source=$samplesPath,target=/samples.jsonl,readonly",
    '--mount', "type=bind,source=$outputDirectory,target=/results",
    $Image,
    'python', '-m', 'evalplus.evaluate',
    '--dataset', $Benchmark,
    '--samples', '/samples.jsonl',
    '--parallel', $Parallel.ToString(),
    '--output-file', "/results/$outputName"
)

docker @dockerArgs
exit $LASTEXITCODE
