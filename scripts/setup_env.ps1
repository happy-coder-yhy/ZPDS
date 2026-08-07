<#
.SYNOPSIS
ZPDS 一键环境部署：主环境 .venv + WiLoR 推理环境 wilor_env + 模型资产 + .env 模板。

.DESCRIPTION
在干净机器上从零搭建 ZPDS 完整运行环境。所有步骤幂等（已存在则跳过）。
详细清单见 docs/部署依赖清单.md。

.PARAMETER ZPDSRoot
ZPDS 项目根目录（默认：脚本所在目录的上一级，即仓库根）。

.PARAMETER WilorRoot
WiLoR 源码目录（默认 E:/ZSPD/WiLoR）。requirements.txt 与 mano_data 在此目录内。

.PARAMETER WilorEnvRoot
WiLoR 推理 venv 位置（默认 E:/ZSPD/wilor_env，与 WiLoR 源码同级，位于项目根之外）。

.PARAMETER PythonExe
使用的 Python 解释器（默认：`python`；建议 3.11.x）。

.PARAMETER Force
强制重建已存在的 venv（默认复用）。

.PARAMETER SkipModels
跳过全部模型下载（网络受限时使用；模型缺失会导致 WiLoR/脱敏不可用）。

.PARAMETER SkipWilor
跳过 wilor_env 创建、WiLoR 模型下载与预检（仅需主环境时使用）。

.EXAMPLE
powershell -ExecutionPolicy Bypass -File scripts/setup_env.ps1
powershell -ExecutionPolicy Bypass -File scripts/setup_env.ps1 -SkipModels -WilorRoot D:/WiLoR
#>
param(
    [string]$ZPDSRoot = "",
    [string]$WilorRoot = "E:/ZSPD/WiLoR",
    [string]$WilorEnvRoot = "E:/ZSPD/wilor_env",
    [string]$PythonExe = "python",
    [switch]$Force,
    [switch]$SkipModels,
    [switch]$SkipWilor
)

# 注意：不用 "Stop" —— PowerShell 5.1 下 native 命令（pip/python）的任何 stderr
# 输出（如 sitecustomize 的 FutureWarning）都会变成终止性错误；统一改为
# 显式检查 $LASTEXITCODE。
$ErrorActionPreference = "Continue"

# ---------------------------------------------------------------------------
# 0. 定位 ZPDS 根目录与工具函数
# ---------------------------------------------------------------------------
if (-not $ZPDSRoot) {
    $ZPDSRoot = Split-Path (Split-Path $MyInvocation.MyCommand.Path -Parent) -Parent
}
$ZPDSRoot = [System.IO.Path]::GetFullPath($ZPDSRoot)

$MainVenv = Join-Path $ZPDSRoot ".venv"
$WilorVenv = $WilorEnvRoot
$ModelsDir = Join-Path $ZPDSRoot "zpds/privacy/models"
$EnvFile = Join-Path $ZPDSRoot ".env"

function Write-Step([string]$Msg) { Write-Host "`n=== $Msg ===" -ForegroundColor Cyan }
function Write-Ok([string]$Msg)   { Write-Host "  [OK] $Msg" -ForegroundColor Green }
function Write-Skip([string]$Msg) { Write-Host "  [SKIP] $Msg" -ForegroundColor DarkGray }
function Write-Warn([string]$Msg) { Write-Host "  [WARN] $Msg" -ForegroundColor Yellow }

function Resolve-Python {
    # 返回 @{ Exe; Args }：优先 3.11（与生产环境一致），依次回退 3.10/3.12。
    # 用 cmd /c 包裹探测：native 命令 stderr 在 PowerShell 5.1 + ErrorActionPreference=Stop
    # 下会变成终止性错误，cmd 包装可避免。
    param([string]$Requested)
    $candidates = @()
    if ($Requested) { $candidates += $Requested }
    $candidates += @("py -3.11", "py -3.10", "py -3.12", "python3.11", "python3.10", "python3", "python")
    foreach ($c in $candidates) {
        $ver = (cmd /c "$c --version 2>&1" | Out-String).Trim()
        if ($LASTEXITCODE -eq 0 -and $ver -match "Python 3\.(10|11|12)") {
            $parts = $c -split " "
            Write-Ok "使用 Python: $ver ($c)"
            return @{ Exe = $parts[0]; Args = @($parts[1..($parts.Count - 1)] | Where-Object { $_ }) }
        }
    }
    throw "未找到 Python 3.10~3.12（推荐 3.11）。请安装后重试，或用 -PythonExe 指定。"
}

function Copy-NumpyPatch([string]$SitePackages) {
    $patch = @"
import numpy as np
# Restore deprecated numpy type aliases that chumpy needs
for _name, _target in [
    ('bool', np.bool_),
    ('int', np.int_),
    ('float', np.float64),
    ('complex', np.complex128),
    ('object', np.object_),
    ('unicode', np.str_),
    ('str', np.str_),
]:
    if not hasattr(np, _name):
        setattr(np, _name, _target)
"@
    $target = Join-Path $SitePackages "sitecustomize.py"
    if (Test-Path $target) { Write-Skip "numpy 别名补丁已存在: $target" }
    else {
        Set-Content -Path $target -Value $patch -Encoding UTF8
        Write-Ok "numpy 别名补丁已写入: $target"
    }
}

function Download-File {
    # 下载失败不终止部署（模型可稍后手动拷贝），返回 $false。
    param([string]$Url, [string]$Destination, [string]$Label)
    if (Test-Path $Destination) { Write-Skip "$Label 已存在: $Destination"; return $true }
    Write-Host "  下载 $Label ..." -ForegroundColor DarkGray
    try {
        Invoke-WebRequest -Uri $Url -OutFile $Destination -UseBasicParsing -MaximumRedirection 10
    } catch {
        Remove-Item $Destination -ErrorAction SilentlyContinue
        Write-Warn "下载失败: $Label（$Url）"
        Write-Warn "  可稍后重试，或在外网机器下载后手动拷贝到 $Destination"
        return $false
    }
    if ((Get-Item $Destination).Length -eq 0) {
        Remove-Item $Destination -ErrorAction SilentlyContinue
        Write-Warn "下载文件为空: $Label（$Url）"
        return $false
    }
    Write-Ok "$Label -> $Destination ($([math]::Round((Get-Item $Destination).Length/1MB,1)) MB)"
    return $true
}

# ---------------------------------------------------------------------------
# 1. 前置检查
# ---------------------------------------------------------------------------
Write-Step "前置检查"
if (-not (Test-Path $ZPDSRoot)) { throw "ZPDS 根目录不存在: $ZPDSRoot" }
$pyInfo = Resolve-Python $PythonExe
$pyExe = $pyInfo.Exe
$pyArgs = @($pyInfo.Args)
Write-Ok "ZPDS 根目录: $ZPDSRoot"

# ---------------------------------------------------------------------------
# 2. 主环境 .venv
# ---------------------------------------------------------------------------
Write-Step "主环境 .venv（数据管线）"
$mainPy = Join-Path $MainVenv "Scripts/python.exe"
if ((Test-Path $mainPy) -and -not $Force) {
    Write-Skip ".venv 已存在，复用（-Force 可重建并重装依赖）"
} else {
    if (Test-Path $MainVenv) { Remove-Item $MainVenv -Recurse -Force }
    Write-Host "  创建 venv: $MainVenv"
    & $pyExe @pyArgs -m venv $MainVenv
    if ($LASTEXITCODE -ne 0) { throw "创建 .venv 失败" }
    & $mainPy -m pip install --upgrade pip -q
    Write-Host "  安装 pyproject 依赖（mcap,hdf5,hands,scene,privacy,dev）..."
    & $mainPy -m pip install -e "$ZPDSRoot[mcap,hdf5,hands,scene,privacy,dev]" -q
    if ($LASTEXITCODE -ne 0) { throw "主环境依赖安装失败" }
}
# numpy < 2 锁定 + 补丁（若 extras 安装引入 numpy 2.x 则降级）
& $mainPy -m pip install "numpy<2" -q 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) { Write-Warn "numpy<2 锁定失败（版本可能不满足 chumpy 兼容）" }
Copy-NumpyPatch (Join-Path $MainVenv "Lib/site-packages")
& $mainPy -c "import cv2, numpy, pandas, pyarrow, mcap; print('  主环境 import 检查 OK (numpy', numpy.__version__ + ')')"

# ---------------------------------------------------------------------------
# 3. WiLoR 推理环境 wilor_env
# ---------------------------------------------------------------------------
if ($SkipWilor) {
    Write-Step "wilor_env（已跳过 -SkipWilor）"
} else {
    Write-Step "wilor_env（WiLoR 推理）"
    if (-not (Test-Path $WilorRoot)) { throw "WiLoR 源码目录不存在: $WilorRoot（请先 clone，或传 -WilorRoot）" }
    $wPy = Join-Path $WilorVenv "Scripts/python.exe"
    if ((Test-Path $wPy) -and -not $Force) {
        Write-Skip "wilor_env 已存在，复用（-Force 可重建并重装依赖）"
    } else {
        if (Test-Path $WilorVenv) { Remove-Item $WilorVenv -Recurse -Force }
        Write-Host "  创建 venv: $WilorVenv"
        & $pyExe @pyArgs -m venv $WilorVenv
        if ($LASTEXITCODE -ne 0) { throw "创建 wilor_env 失败" }
        & $wPy -m pip install --upgrade pip -q
        Write-Host "  安装 torch 2.5.1+cu121 / torchvision 0.20.1+cu121（约 2.5GB，较慢）..."
        & $wPy -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121 -q
        if ($LASTEXITCODE -ne 0) {
            Write-Warn "cu121 torch 安装失败，回退 CPU 版 torch 2.5.1（WiLoR 可推理但慢）"
            & $wPy -m pip install torch==2.5.1 torchvision==0.20.1 -q
        }
        Write-Host "  安装 WiLoR requirements..."
        & $wPy -m pip install -r (Join-Path $WilorRoot "requirements.txt") -q
        if ($LASTEXITCODE -ne 0) { throw "WiLoR 依赖安装失败（chumpy 需编译，可重试一次）" }
        & $wPy -m pip install "numpy<2" -q 2>&1 | Out-Null
        Copy-NumpyPatch (Join-Path $WilorVenv "Lib/site-packages")
    }
    & $wPy -c "import torch; print('  wilor_env torch', torch.__version__, '| CUDA:', torch.cuda.is_available())"

    if (-not $SkipModels) {
        Write-Step "模型：WiLoR（detector.pt + wilor_final.ckpt）"
        $dlScript = Join-Path $ZPDSRoot "scripts/download_wilor_models.ps1"
        $ptDir = Join-Path $WilorRoot "pretrained_models"
        $hasModels = (Test-Path (Join-Path $ptDir "detector.pt")) -and
                     (Test-Path (Join-Path $ptDir "wilor_final.ckpt"))
        if ($hasModels) {
            Write-Skip "WiLoR 模型已存在: $ptDir（跳过下载）"
        } elseif (Test-Path $dlScript) {
            Write-Warn "WiLoR 模型缺失。下载源为 HuggingFace，国内网络可能不通："
            Write-Warn "  可先设置 HTTPS_PROXY 环境变量重试，或在外网机器下载后拷贝到 $ptDir"
            & powershell -ExecutionPolicy Bypass -File $dlScript -Destination $ptDir
            if ($LASTEXITCODE -ne 0) {
                Write-Warn "WiLoR 模型下载失败（可稍后重试或手动拷贝）"
            } else { Write-Ok "WiLoR 模型下载完成" }
        } else { Write-Warn "未找到 $dlScript，跳过 WiLoR 模型下载" }
    }
    if (-not $SkipModels) {
        Write-Step "模型：MANO 数据检查"
        $mano = Join-Path $WilorRoot "mano_data"
        if (Test-Path (Join-Path $mano "MANO_RIGHT.pkl")) { Write-Ok "mano_data 已就绪: $mano" }
        else {
            Write-Warn "mano_data 缺失。需从 MANO 官网（https://mano.is.tue.mpg.de/）申请后放置到: $mano"
            Write-Warn "  （MANO 模型需许可，无法自动下载；缺失时 WiLoR 推理不可用）"
        }
    }
}

# ---------------------------------------------------------------------------
# 4. 模型：隐私脱敏（face + yolo）
# ---------------------------------------------------------------------------
if (-not $SkipModels) {
    Write-Step "模型：隐私脱敏（zpds/privacy/models/）"
    New-Item -ItemType Directory -Path $ModelsDir -Force | Out-Null
    Download-File -Url "https://huggingface.co/AdamCodd/YOLOv11n-face-detection/resolve/main/yolov11n-face.pt" `
        -Destination (Join-Path $ModelsDir "yolov11n-face.pt") -Label "yolov11n-face.pt"
    Download-File -Url "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt" `
        -Destination (Join-Path $ModelsDir "yolo.pt") -Label "yolo.pt（文本区域提议）"
} else {
    Write-Step "模型下载（已跳过 -SkipModels）"
}

# ---------------------------------------------------------------------------
# 5. .env 模板
# ---------------------------------------------------------------------------
Write-Step ".env 密钥配置"
if (Test-Path $EnvFile) {
    Write-Skip ".env 已存在: $EnvFile"
} else {
    Set-Content -Path $EnvFile -Value @"
# ZPDS 密钥配置（勿提交 git）
DASHSCOPE_API_KEY=
VLM_MODEL=qwen-vl-max
"@ -Encoding UTF8
    Write-Warn "已生成 .env 模板，请填写 DASHSCOPE_API_KEY / VLM_MODEL 后使用"
}

# ---------------------------------------------------------------------------
# 6. 自检
# ---------------------------------------------------------------------------
Write-Step "自检"
& (Join-Path $MainVenv "Scripts/python.exe") -c "import cv2, numpy, pandas, pyarrow, mcap, easyocr, ultralytics; print('  主环境: 全部模块 OK')"
if (-not $SkipWilor) {
    $wPy = Join-Path $WilorVenv "Scripts/python.exe"
    & $wPy -c "import torch; assert torch.__version__.startswith('2.5.1'); print('  wilor_env: torch', torch.__version__, 'OK')"
    if (Test-Path (Join-Path $ZPDSRoot "scripts/check_wilor_readiness.py")) {
        # 预检脚本依赖 cwd = ZPDS 根（scripts 包 + config.yaml）
        Push-Location $ZPDSRoot
        & $wPy -m scripts.check_wilor_readiness 2>&1 | Out-Null
        $readinessExit = $LASTEXITCODE
        Pop-Location
        if ($readinessExit -eq 0) { Write-Ok "WiLoR 资产预检通过" }
        else { Write-Warn "WiLoR 资产预检未通过（通常为模型缺失，详见上方提示）" }
    }
}

Write-Host "`n==================== 部署完成 ====================" -ForegroundColor Green
Write-Host "主环境:   $MainVenv"
Write-Host "WiLoR:    $WilorVenv  (源码 $WilorRoot)"
Write-Host "脱敏模型: $ModelsDir"
Write-Host "密钥:     $EnvFile"
Write-Host "下一步:   参考 docs/部署依赖清单.md §5 验证命令"
