param(
    [string]$Destination = "models/wilor",
    [string]$ModelBaseUrl = "https://huggingface.co/rolpotamias/WiLoR/resolve/dcf093e9c3e82a63e1a61c851dc20cb4612ccb5f",
    [string]$SpaceBaseUrl = "https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/99fe3d7acff8104ecca1055df7467709506c2fa6/pretrained_models",
    [ValidateRange(1, 32)]
    [int]$Connections = 12
)

$ErrorActionPreference = "Stop"

$files = @(
    @{
        Name = "detector.pt"
        Size = [int64]53582271
        Sha256 = "5ef3df44e42d2db52d4ffe91f83a22ce9925e2acc9abebf453f2c5d22e380033"
        Url = "$ModelBaseUrl/detector.pt"
    },
    @{
        Name = "wilor_final.ckpt"
        Size = [int64]2564989533
        Sha256 = "3e97aafc7dd08d883a4cc5a027df61fdb6fda6136dbd1319405413862ada6bb2"
        Url = "$ModelBaseUrl/wilor_final.ckpt"
    }
)

function Assert-WithinDestination {
    param(
        [string]$Candidate,
        [string]$DestinationRoot
    )

    $resolvedRoot = [System.IO.Path]::GetFullPath($DestinationRoot)
    $resolvedCandidate = [System.IO.Path]::GetFullPath($Candidate)
    $rootWithSeparator = $resolvedRoot.TrimEnd(
        [System.IO.Path]::DirectorySeparatorChar
    ) + [System.IO.Path]::DirectorySeparatorChar
    if (-not $resolvedCandidate.StartsWith(
        $rootWithSeparator,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "下载目标超出 Destination: $resolvedCandidate"
    }
}

function Test-VerifiedFile {
    param(
        [string]$Path,
        [int64]$ExpectedSize,
        [string]$ExpectedSha256
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $false
    }
    $item = Get-Item -LiteralPath $Path
    if ($item.Length -ne $ExpectedSize) {
        return $false
    }
    $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    return $actual -eq $ExpectedSha256
}

function Download-VerifiedFile {
    param(
        [string]$Url,
        [string]$Path,
        [int64]$ExpectedSize,
        [string]$ExpectedSha256,
        [int]$ConnectionCount
    )

    if (Test-VerifiedFile $Path $ExpectedSize $ExpectedSha256) {
        Write-Host "Verified existing file: $Path"
        return
    }

    $partsDirectory = "$Path.parts"
    Assert-WithinDestination $partsDirectory $Destination
    New-Item -ItemType Directory -Path $partsDirectory -Force | Out-Null

    $chunkSize = [int64][math]::Ceiling($ExpectedSize / $ConnectionCount)
    $downloads = @()
    for ($index = 0; $index -lt $ConnectionCount; $index++) {
        $start = [int64]$index * $chunkSize
        if ($start -ge $ExpectedSize) {
            break
        }
        $end = [math]::Min($ExpectedSize - 1, $start + $chunkSize - 1)
        $expectedPartSize = $end - $start + 1
        $partPath = Join-Path $partsDirectory ("part-{0:D3}" -f $index)
        Assert-WithinDestination $partPath $Destination

        if (
            (Test-Path -LiteralPath $partPath -PathType Leaf) -and
            (Get-Item -LiteralPath $partPath).Length -eq $expectedPartSize
        ) {
            $downloads += @{
                Index = $index
                Path = $partPath
                ExpectedSize = $expectedPartSize
                Process = $null
            }
            continue
        }
        if (Test-Path -LiteralPath $partPath) {
            Remove-Item -LiteralPath $partPath -Force
        }

        $curlArguments = @(
            "--http1.1",
            "-L",
            "--fail",
            "--retry", "8",
            "--retry-all-errors",
            "--range", "$start-$end",
            "--output", $partPath,
            $Url
        )
        $process = Start-Process `
            -FilePath "curl.exe" `
            -ArgumentList $curlArguments `
            -PassThru `
            -WindowStyle Hidden
        $downloads += @{
            Index = $index
            Path = $partPath
            ExpectedSize = $expectedPartSize
            Process = $process
        }
    }

    foreach ($download in $downloads) {
        if ($null -ne $download.Process) {
            $download.Process.WaitForExit()
            if ($download.Process.ExitCode -ne 0) {
                throw "分片下载失败: $($download.Path), exit=$($download.Process.ExitCode)"
            }
        }
        $actualSize = (Get-Item -LiteralPath $download.Path).Length
        if ($actualSize -ne $download.ExpectedSize) {
            throw (
                "分片大小错误: {0}, expected={1}, actual={2}" -f
                $download.Path,
                $download.ExpectedSize,
                $actualSize
            )
        }
    }

    $temporaryPath = "$Path.assembling"
    Assert-WithinDestination $temporaryPath $Destination
    if (Test-Path -LiteralPath $temporaryPath) {
        Remove-Item -LiteralPath $temporaryPath -Force
    }
    $outputStream = [System.IO.File]::Open(
        $temporaryPath,
        [System.IO.FileMode]::CreateNew,
        [System.IO.FileAccess]::Write,
        [System.IO.FileShare]::None
    )
    try {
        foreach (
            $download in (
                $downloads | Sort-Object { [int]$_["Index"] }
            )
        ) {
            $partStream = [System.IO.File]::OpenRead($download.Path)
            try {
                $partStream.CopyTo($outputStream)
            }
            finally {
                $partStream.Dispose()
            }
        }
    }
    finally {
        $outputStream.Dispose()
    }

    $assembledSize = (Get-Item -LiteralPath $temporaryPath).Length
    if ($assembledSize -ne $ExpectedSize) {
        Remove-Item -LiteralPath $temporaryPath -Force
        throw "合并文件大小错误: expected=$ExpectedSize, actual=$assembledSize"
    }
    $actualSha256 = (
        Get-FileHash -LiteralPath $temporaryPath -Algorithm SHA256
    ).Hash.ToLowerInvariant()
    if ($actualSha256 -ne $ExpectedSha256) {
        Remove-Item -LiteralPath $temporaryPath -Force
        throw "SHA-256 不匹配: expected=$ExpectedSha256, actual=$actualSha256"
    }

    Move-Item -LiteralPath $temporaryPath -Destination $Path -Force
    foreach ($download in $downloads) {
        Remove-Item -LiteralPath $download.Path -Force
    }
    Remove-Item -LiteralPath $partsDirectory -Force
    Write-Host "Downloaded and verified: $Path"
}

$destinationPath = [System.IO.Path]::GetFullPath($Destination)
New-Item -ItemType Directory -Path $destinationPath -Force | Out-Null

foreach ($file in $files) {
    $targetPath = Join-Path $destinationPath $file.Name
    Assert-WithinDestination $targetPath $destinationPath
    Download-VerifiedFile `
        -Url $file.Url `
        -Path $targetPath `
        -ExpectedSize $file.Size `
        -ExpectedSha256 $file.Sha256 `
        -ConnectionCount $Connections
}

$configPath = Join-Path $destinationPath "model_config.yaml"
Assert-WithinDestination $configPath $destinationPath
curl.exe `
    --http1.1 `
    -L `
    --fail `
    --retry 8 `
    --retry-all-errors `
    --output $configPath `
    "$SpaceBaseUrl/model_config.yaml"
if ($LASTEXITCODE -ne 0) {
    throw "model_config.yaml 下载失败: exit=$LASTEXITCODE"
}

Write-Host "WiLoR model download complete: $destinationPath"
