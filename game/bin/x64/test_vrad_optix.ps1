param([int]$TimeoutExtensionMinutes = 0)

if ($PSVersionTable.PSVersion.Major -lt 7 -or ($PSVersionTable.PSVersion.Major -eq 7 -and $PSVersionTable.PSVersion.Minor -lt 6)) {
    Throw "This script requires PowerShell 7.6 or later. Your version: $($PSVersionTable.PSVersion)"
}

$MOD_DIR = "..\..\mod_hl2mp"
$LOG_FILE = "test_vrad_optix.log"
$MAP_NAME = "validation"

$REF_DIR = "bsp_unit_tests\reference"
$CONTROL_DIR = "bsp_unit_tests\control"
$TEST_DIR = "bsp_unit_tests\test"

$REF_LOG = "$REF_DIR\$MAP_NAME.log"
$CONTROL_LOG = "$CONTROL_DIR\$MAP_NAME.log"
$CUDA_LOG = "$TEST_DIR\$MAP_NAME.log"

$GAME_EXE = "E:\Steam\steamapps\common\Source SDK Base 2013 Multiplayer\hl2.exe"
$GAME_MAPS = "E:\Steam\steamapps\common\Source SDK Base 2013 Multiplayer\sourcetest\maps"
$GAME_SCREENSHOTS = "E:\Steam\steamapps\common\Source SDK Base 2013 Multiplayer\sourcetest\screenshots"

# Clean up any existing tool processes
$tools = @("vbsp", "vvis", "vrad", "vvis_cuda", "vrad_rtx")
foreach ($tool in $tools) {
    Get-Process -Name $tool -ErrorAction SilentlyContinue | Stop-Process -Force
}

# Clear or create the log file with UTF-8 encoding
"" | Out-File -FilePath $LOG_FILE -Encoding utf8

function Write-LogMessage {
    param([string]$Message, [bool]$ToLog = $true)
    Write-Host $Message
    if ($ToLog) {
        $Message | Out-File -FilePath $LOG_FILE -Append -Encoding utf8
    }
}

function Take-Screenshot {
    param([string]$BspPath, [string]$TargetTga)
    
    Write-LogMessage "Taking screenshot for $BspPath..."
    
    # 1. Copy bsp
    Copy-Item $BspPath "$GAME_MAPS\$MAP_NAME.bsp" -Force
    
    # 2. Run hl2.exe
    $gameArgs = "-game", "sourcetest", "-novid", "-sw", "-w", "2560", "-h", "1440", "+sv_cheats 1", "+map $MAP_NAME", "+cl_mouselook 0", "+cl_drawhud 0", "+r_drawviewmodel 0", "+mat_fullbright 2", "+wait 1000", "+screenshot", "+quit"
    $proc = Start-Process -FilePath $GAME_EXE -ArgumentList $gameArgs -Wait -PassThru
    if ($proc.ExitCode -ne 0) {
        Write-LogMessage "CRITICAL ERROR: hl2.exe exited with code $($proc.ExitCode) for map $BspPath."
        exit 1
    }

    # 3. Move screenshot
    if (Test-Path "$GAME_SCREENSHOTS\${MAP_NAME}0000.tga") {
        Move-Item "$GAME_SCREENSHOTS\${MAP_NAME}0000.tga" $TargetTga -Force
    }
    else {
        Write-LogMessage "CRITICAL ERROR: Screenshot not found for $BspPath!"
        exit 1
    }
}

# Ensure directories exist
if (!(Test-Path $REF_DIR)) { New-Item -ItemType Directory -Path $REF_DIR | Out-Null }
if (!(Test-Path $CONTROL_DIR)) { New-Item -ItemType Directory -Path $CONTROL_DIR | Out-Null }
if (!(Test-Path $TEST_DIR)) { New-Item -ItemType Directory -Path $TEST_DIR | Out-Null }

# Refresh maps from source
$MAP_SRC = "bsp_unit_tests\$MAP_NAME.bsp"
if (!(Test-Path $MAP_SRC)) {
    Write-LogMessage "CRITICAL ERROR: Source map $MAP_SRC not found!"
    exit 1
}
Copy-Item $MAP_SRC "$REF_DIR\$MAP_NAME.bsp" -Force
Copy-Item $MAP_SRC "$CONTROL_DIR\$MAP_NAME.bsp" -Force
Copy-Item $MAP_SRC "$TEST_DIR\$MAP_NAME.bsp" -Force

$HasReference = Test-Path ".\vrad.exe"

# --- Phase 1: Reference Run (vrad.exe, CPU only) ---
$refTime = New-TimeSpan
if ($HasReference) {
    Write-LogMessage "--- Compiling reference map (Original vrad.exe) ---"
    $fullLogPath = Join-Path (Get-Location).Path $REF_LOG
    Write-LogMessage "VRAD Log: $fullLogPath"
    "" | Out-File -FilePath $REF_LOG -Encoding utf8
    $start = Get-Date
    & ".\vrad.exe" "$REF_DIR\$MAP_NAME" *>$null
    if ($LASTEXITCODE -ne 0) {
        Write-LogMessage "WARNING: vrad.exe (reference) failed with exit code $LASTEXITCODE. Skipping reference comparison."
        $HasReference = $false
    }
    else {
        $refTime = (Get-Date) - $start
    }
}
else {
    Write-LogMessage "vrad.exe not found, skipping reference pass."
}

# --- Phase 2: Control Run (vrad_rtx.exe CPU) ---
Write-LogMessage "--- Compiling control map (vrad_rtx.exe CPU only) ---"
$fullLogPath = Join-Path (Get-Location).Path $CONTROL_LOG
Write-LogMessage "VRAD Control Log: $fullLogPath"
"" | Out-File -FilePath $CONTROL_LOG -Encoding utf8
$start = Get-Date
& ".\vrad_rtx.exe" "$CONTROL_DIR\$MAP_NAME" *>$null
if ($LASTEXITCODE -ne 0) {
    Write-LogMessage "CRITICAL ERROR: vrad_rtx.exe (control, CPU only) failed with exit code $LASTEXITCODE."
    exit 1
}
$controlTime = (Get-Date) - $start

# --- Phase 4: Reference vs Control Comparison ---
if ($HasReference) {
    Write-LogMessage "`n--- Comparing Reference vs Control (CPU Parity Check) ---"
    
    # Check binary identity first
    fc.exe /b /LB1 "$REF_DIR\$MAP_NAME.bsp" "$CONTROL_DIR\$MAP_NAME.bsp" | Out-Null
    $isIdentical = ($LASTEXITCODE -eq 0)
    
    if ($isIdentical) {
        Write-LogMessage "RESULT: PASS (Reference and Control BSPs are bit-identical)"
    }
    else {
        Write-LogMessage "BSPs differ bitwise. Checking lightmaps..."
        $pythonDiff = python bsp_diff_lightmaps.py "$REF_DIR\$MAP_NAME.bsp" "$CONTROL_DIR\$MAP_NAME.bsp" --threshold 0.1 2>&1
        $pythonDiff | Write-Host
        $pythonDiff | Out-File -FilePath $LOG_FILE -Append -Encoding utf8
        
        if ($pythonDiff -like "*All lightmaps are identical.*") {
            Write-LogMessage "RESULT: PASS (BSP file bytes differ, but all lightmaps are identical. Skipping visual check.)"
        }
        else {
            Write-LogMessage "Lightmaps differ. Initiating visual comparison (Ref vs Control)..."
            Take-Screenshot "$REF_DIR\$MAP_NAME.bsp" "screenshot_ref.tga"
            Take-Screenshot "$CONTROL_DIR\$MAP_NAME.bsp" "screenshot_control.tga"
            
            $diffOutput = .\tgadiff.exe screenshot_ref.tga screenshot_control.tga screenshot_diff_ref_ctrl.tga 2>&1
            $diffMatch = $diffOutput | Select-String "Difference: ([\d\.]+)%"
            if ($diffMatch) {
                $percentDiff = [double]$diffMatch.Matches.Groups[1].Value
                Write-LogMessage "Visual Difference (Ref vs Control): $percentDiff%"
                if ($percentDiff -gt 0.5) {
                    Write-LogMessage "CRITICAL ERROR: Fundamental issue detected! CPU tests should be identical. Difference: $percentDiff% > 0.5% tolerance."
                    exit 1
                }
                else {
                    Write-LogMessage "CPU Visual tests passed within 0.5% margin of error."
                }
            }
            else {
                Write-LogMessage "CRITICAL ERROR: Could not parse tgadiff output for Ref vs Control."
                exit 1
            }
        }
    }
}

# --- Phase 3: Test Run (vrad_rtx.exe -cuda) ---
Write-LogMessage "--- Compiling test map (vrad_rtx.exe -cuda) ---"
$fullLogPath = Join-Path (Get-Location).Path $CUDA_LOG
Write-LogMessage "VRAD OptiX Log: $fullLogPath"
"" | Out-File -FilePath $CUDA_LOG -Encoding utf8
$start = Get-Date

# Use .NET Process class with event-based async output draining.
# CRITICAL: Do NOT use Task.Run() with PowerShell scriptblocks to drain
# stdout/stderr -- PS scriptblocks don't reliably run on .NET ThreadPool
# threads, causing the pipe buffer (4KB on Windows) to fill up and deadlock
# the child process when it tries to write (typically around bounce 12-13).
# Instead, use BeginOutputReadLine/BeginErrorReadLine which use native .NET
# async callbacks that are guaranteed to drain the pipes.
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = (Resolve-Path ".\vrad_rtx.exe").Path
$psi.Arguments = "-cuda $TEST_DIR\$MAP_NAME"
$psi.WorkingDirectory = (Get-Location).Path
$psi.UseShellExecute = $false
$psi.CreateNoWindow = $false
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true

try {
    $process = [System.Diagnostics.Process]::Start($psi)
}
catch {
    Write-LogMessage "CRITICAL ERROR: Failed to start vrad_rtx.exe -cuda: $($_.Exception.Message)"
    exit 1
}

if ($null -eq $process) {
    Write-LogMessage "CRITICAL ERROR: Failed to start vrad_rtx.exe -cuda. Process object is NULL."
    exit 1
}

# Use .NET event-based async draining (reliable, no deadlock)
$cudaOutput = [System.Text.StringBuilder]::new()
$cudaErrors = [System.Text.StringBuilder]::new()

$outEvent = Register-ObjectEvent -InputObject $process -EventName OutputDataReceived -Action {
    if ($null -ne $Event.SourceEventArgs.Data) {
        $Event.MessageData.AppendLine($Event.SourceEventArgs.Data)
    }
} -MessageData $cudaOutput

$errEvent = Register-ObjectEvent -InputObject $process -EventName ErrorDataReceived -Action {
    if ($null -ne $Event.SourceEventArgs.Data) {
        $Event.MessageData.AppendLine($Event.SourceEventArgs.Data)
    }
} -MessageData $cudaErrors

$process.BeginOutputReadLine()
$process.BeginErrorReadLine()

$timedOut = $false
$maxSeconds = ($controlTime.TotalSeconds * 2.5) + ($TimeoutExtensionMinutes * 60)
$hasWarnedByExceedingControl = $false

while (-not $process.HasExited) {
    $elapsed = (Get-Date) - $start
    if (($elapsed.TotalSeconds -gt $controlTime.TotalSeconds) -and (-not $hasWarnedByExceedingControl)) {
        $remaining = [math]::Round($maxSeconds - $elapsed.TotalSeconds)
        Write-LogMessage "WARNING: Test run has exceeded the control test time ($([math]::Round($controlTime.TotalSeconds))s). Will terminate in ${remaining}s."
        $hasWarnedByExceedingControl = $true
    }
    if ($elapsed.TotalSeconds -gt $maxSeconds) {
        $process.Kill()
        $timedOut = $true
        Write-LogMessage "CRITICAL ERROR: vrad_rtx -cuda hung or is significantly slower than control test! Use -TimeoutExtensionMinutes <minutes> to extend wait time."
        break
    }
    Start-Sleep -Seconds 1
}

if (-not $timedOut) {
    $process.WaitForExit()
}

# Clean up event subscriptions
Unregister-Event -SourceIdentifier $outEvent.Name
Unregister-Event -SourceIdentifier $errEvent.Name
Remove-Job -Job $outEvent -Force
Remove-Job -Job $errEvent -Force

# Save captured stderr if any
$errText = $cudaErrors.ToString()
if ($errText.Trim()) {
    $errText | Out-File -FilePath "$env:TEMP\vrad_rtx_cuda_err.txt" -Encoding utf8
}

# Final refresh to ensure exit code is captured
$exitCode = if ($process.HasExited) { $process.ExitCode } else { $null }

if (-not $timedOut -and ($null -eq $exitCode -or $exitCode -ne 0)) {
    $errMessage = if ($null -eq $exitCode) { "UNKNOWN (NULL)" } else { $exitCode }
    Write-LogMessage "CRITICAL ERROR: vrad_rtx.exe -cuda failed with exit code $errMessage."
    exit 1
}
$cudaTime = (Get-Date) - $start

# --- Final Timing Summary ---
Write-LogMessage "`n--- Timing Summary ---"
if ($HasReference) { Write-LogMessage "vrad.exe Time (Source SDK 2013 Unmodified):	$($refTime.TotalSeconds.ToString("F2"))s" }
Write-LogMessage "vrad_rtx.exe Time (CPU only):			$($controlTime.TotalSeconds.ToString("F2"))s"
if ($timedOut) {
    Write-LogMessage "vrad_rtx.exe -cuda Time (GPU accelerated):	Did not finish"
}
else {
    Write-LogMessage "vrad_rtx.exe -cuda Time (GPU accelerated):	$($cudaTime.TotalSeconds.ToString("F2"))s"
}

if (-not $timedOut) {
    Write-LogMessage "`n--- Comparing Control vs Test (GPU Parity Check) ---"
    fc.exe /b /LB1 "$CONTROL_DIR\$MAP_NAME.bsp" "$TEST_DIR\$MAP_NAME.bsp" | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-LogMessage "Control and Test BSPs differ bitwise. Investigating..."
        $pythonDiff = python bsp_diff_lightmaps.py "$CONTROL_DIR\$MAP_NAME.bsp" "$TEST_DIR\$MAP_NAME.bsp" --threshold 0.1 2>&1
        $pythonDiff | Write-Host
        $pythonDiff | Out-File -FilePath $LOG_FILE -Append -Encoding utf8
        
        if ($pythonDiff -like "*All lightmaps are identical.*") {
            Write-LogMessage "RESULT: PASS (BSP file bytes differ, but all lightmaps are identical)"
        }
        else {
            Write-LogMessage "Initiating visual comparison for GPU..."
            if (!(Test-Path "screenshot_control.tga")) {
                Take-Screenshot "$CONTROL_DIR\$MAP_NAME.bsp" "screenshot_control.tga"
            }
            Take-Screenshot "$TEST_DIR\$MAP_NAME.bsp" "screenshot_test.tga"
            
            $diffOutput = .\tgadiff.exe screenshot_control.tga screenshot_test.tga screenshot_diff_ctrl_test.tga 2>&1
            $diffMatch = $diffOutput | Select-String "Difference: ([\d\.]+)%"
            if ($diffMatch) {
                $percentDiff = [double]$diffMatch.Matches.Groups[1].Value
                Write-LogMessage "Visual Difference (Control vs Test): $percentDiff%"
                if ($percentDiff -ge 0.5) {
                    Write-LogMessage "RESULT: FAIL (Visual difference $percentDiff% >= 0.5%)"
                    exit 1
                }
                else {
                    Write-LogMessage "RESULT: PASS (Visual difference $percentDiff% < 0.5%)"
                }
            }
            else {
                Write-LogMessage "Warning: Could not parse tgadiff output for Control vs Test."
                exit 1
            }
        }
    }
    else {
        Write-LogMessage "RESULT: PASS (BSPs are bit-identical)"
    }
}

if ($timedOut) { exit 1 }
