Param(
    [switch]$Help,
    [switch]$Build,
    [switch]$NoCache
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location -Path $ScriptRoot

$script:ComposeCommand = $null
$script:ComposeArgs = @()

function Log-Message {
    param(
        [Parameter(Mandatory = $true)][string]$Level,
        [Parameter(Mandatory = $true)][string]$Message,
        [ConsoleColor]$Color = [ConsoleColor]::White
    )

    $timestamp = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
    Write-Host "[$timestamp] [$Level] $Message" -ForegroundColor $Color
}

function Log-Info {
    param([string]$Message)
    Log-Message -Level 'INFO' -Message $Message -Color [ConsoleColor]::Green
}

function Log-Warning {
    param([string]$Message)
    Log-Message -Level 'WARNING' -Message $Message -Color [ConsoleColor]::Yellow
}

function Log-Error {
    param([string]$Message)
    Log-Message -Level 'ERROR' -Message $Message -Color [ConsoleColor]::Red
}

function Show-Help {
    Write-Host @'
NarratoAI Docker 一键部署脚本 (Windows 版)

用法:
    powershell -ExecutionPolicy Bypass -File .\deploy_windows.ps1 [参数]

参数:
    -Help        显示帮助信息
    -Build       强制重新构建镜像
    -NoCache     构建时不使用缓存

示例:
    .\deploy_windows.ps1
    .\deploy_windows.ps1 -Build
    .\deploy_windows.ps1 -Build -NoCache
'@
}

function Initialize-ComposeCommand {
    if (Get-Command 'docker-compose' -ErrorAction SilentlyContinue) {
        $script:ComposeCommand = 'docker-compose'
        $script:ComposeArgs = @()
        return
    }

    try {
        docker compose version | Out-Null
        $script:ComposeCommand = 'docker'
        $script:ComposeArgs = @('compose')
    }
    catch {
        Log-Error 'Docker Compose 未安装或未在 PATH 中，请先安装 Docker Desktop 或独立的 Docker Compose。'
        exit 1
    }
}

function Invoke-Compose {
    param([string[]]$Args)

    $fullArgs = @()
    if ($script:ComposeArgs) {
        $fullArgs += $script:ComposeArgs
    }
    if ($Args) {
        $fullArgs += $Args
    }

    & $script:ComposeCommand @fullArgs
    if ($LASTEXITCODE -ne 0) {
        throw "命令执行失败: $($script:ComposeCommand) $($fullArgs -join ' ')"
    }
}

function Check-Requirements {
    Log-Info '检查系统要求...'

    if (-not (Get-Command 'docker' -ErrorAction SilentlyContinue)) {
        Log-Error '未找到 Docker，请先安装 Docker Desktop。'
        exit 1
    }

    Initialize-ComposeCommand

    try {
        docker info | Out-Null
    }
    catch {
        Log-Error 'Docker 服务未运行，请启动 Docker Desktop 后重试。'
        exit 1
    }
}

function Ensure-Config {
    if (-not (Test-Path 'config.toml')) {
        if (Test-Path 'config.example.toml') {
            Log-Warning '未找到 config.toml，正在复制示例配置...'
            Copy-Item 'config.example.toml' 'config.toml'
            Log-Info '已生成 config.toml，请根据需要更新 API 配置。'
        }
        else {
            Log-Error '未找到 config.example.toml，无法自动生成配置文件。'
            exit 1
        }
    }
}

function Test-DockerImage {
    param([string]$ImageName)

    $imageId = docker images -q $ImageName | Select-Object -First 1
    if ($null -eq $imageId) {
        return $false
    }
    return -not [string]::IsNullOrWhiteSpace(($imageId | Out-String).Trim())
}

function Build-Image {
    Log-Info '构建 Docker 镜像...'
    $args = @('build')
    if ($NoCache) {
        $args += '--no-cache'
    }
    Invoke-Compose -Args $args
}

function Start-Services {
    Log-Info '启动 NarratoAI 服务...'
    try {
        Invoke-Compose -Args @('down')
    }
    catch {
        Log-Warning "关闭现有服务失败: $($_.Exception.Message)"
    }
    Invoke-Compose -Args @('up', '-d')
}

function Wait-ForService {
    Log-Info '等待服务就绪...'
    $maxAttempts = 30
    $invokeCommand = Get-Command Invoke-WebRequest
    $supportsBasicParsing = $invokeCommand.Parameters.ContainsKey('UseBasicParsing')
    for ($attempt = 1; $attempt -le $maxAttempts; $attempt++) {
        try {
            $invokeParams = @{ Uri = 'http://localhost:8501/_stcore/health'; TimeoutSec = 5; ErrorAction = 'Stop' }
            if ($supportsBasicParsing) {
                $invokeParams['UseBasicParsing'] = $true
            }
            $response = Invoke-WebRequest @invokeParams
            if ($response.StatusCode -eq 200) {
                Log-Info '服务已就绪。'
                return $true
            }
        }
        catch {
            Start-Sleep -Seconds 2
        }
    }
    Log-Warning '服务启动超时，请检查日志。'
    return $false
}

function Show-DeploymentInfo {
    Log-Info 'NarratoAI 部署完成！'
    Write-Host '访问地址: http://localhost:8501'
    Write-Host '常用命令:'
    Write-Host '  查看日志: docker compose logs -f 或 docker-compose logs -f'
    Write-Host '  停止服务: docker compose down 或 docker-compose down'
    Write-Host '  重启服务: docker compose restart 或 docker-compose restart'
}

try {
    if ($Help) {
        Show-Help
        exit 0
    }

    Log-Info '开始 NarratoAI Docker 部署 (Windows)...'

    Check-Requirements
    Ensure-Config

    $imageName = 'narratoai:latest'
    if ($Build -or -not (Test-DockerImage -ImageName $imageName)) {
        Build-Image
    }

    Start-Services

    if (Wait-ForService) {
        Show-DeploymentInfo
    }
    else {
        Log-Error '部署过程中出现问题，请使用日志命令排查。'
        try {
            Invoke-Compose -Args @('logs', '--tail', '20')
        }
        catch {
            Log-Warning "获取日志失败: $($_.Exception.Message)"
        }
        exit 1
    }
}
catch {
    Log-Error $_.Exception.Message
    exit 1
}
