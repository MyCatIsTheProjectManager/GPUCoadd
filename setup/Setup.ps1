# GPUCoadd setup and uninstall window (Windows PowerShell 5.1 and WinForms, nothing else needed).
# Installs into one folder without admin rights: the GPUCoadd files, a private Python 3.11 and the
# pinned packages from packages.json (each verified by SHA-256), then runs the GPU self-test.
#   Setup.ps1 [-Version V] [-InstallDir D]   install or update
#   Setup.ps1 -Uninstall                     remove (run from the installed folder)
#   Setup.ps1 -DryRun [-Snapshot DIR]        show every page with simulated work; installs nothing
param(
    [string]$Version = "",
    [string]$InstallDir = "",
    [switch]$Uninstall,
    [switch]$DryRun,
    [string]$Snapshot = "",
    [string]$LocalFiles = ""
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Windows.Forms, System.Drawing
Add-Type -Namespace Native -Name Dpi -MemberDefinition '[DllImport("user32.dll")] public static extern bool SetProcessDPIAware();'
[void][Native.Dpi]::SetProcessDPIAware()
[System.Windows.Forms.Application]::EnableVisualStyles()

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Manifest = Get-Content -Raw -Encoding UTF8 (Join-Path $Here "packages.json") | ConvertFrom-Json
if (-not $Version) {
    $v = Join-Path (Split-Path -Parent $Here) "VERSION"
    if (Test-Path $v) { $Version = (Get-Content $v -TotalCount 1).Trim() }
}
# No suggested install folder: the user chooses one (or passes -InstallDir)
if ($Uninstall -and -not $InstallDir) { $InstallDir = Split-Path -Parent $Here }
if ($InstallDir) { $InstallDir = [IO.Path]::GetFullPath($InstallDir) }
$WorkDir = Join-Path $env:LOCALAPPDATA "GPUCoadd"
$Files = @($Manifest.python) + @($Manifest.packages)
$DownloadBytes = [int64](($Files | Measure-Object -Property size -Sum).Sum)
$PackageBytes = [int64](($Manifest.packages | Measure-Object -Property size -Sum).Sum)
$InstalledBytes = [int64](1.62 * $PackageBytes + 150MB)  # measured on a real install (4.5 GB): 1.62x the download
$PeakBytes = $DownloadBytes + $InstalledBytes
$Biggest = $Manifest.packages | Sort-Object size -Descending | Select-Object -First 1
# Top-level names setup puts in the install folder, and its record of every file it installed there
$AppNames = @("worker", "GPUCoadd", "start_gpucoadd.pyw", "docs", "README.md", "LICENSE", "THIRD-PARTY-NOTICES.txt", "VERSION", "requirements-lock.txt")
if (Test-Path (Join-Path $Here "app.zip")) {
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $z = [IO.Compression.ZipFile]::OpenRead((Join-Path $Here "app.zip"))
    $AppNames = @($z.Entries | ForEach-Object { $_.FullName.Split("/")[0] } | Sort-Object -Unique)
    $z.Dispose()
}
$OwnNames = $AppNames + @(".venv", "python", "setup", "downloads", "Uninstall GPUCoadd.cmd")
$RecordName = "setup\installed-files.txt"

function Size([double]$b) {
    if ($b -ge 1GB) { return "{0:N1} GB" -f ($b / 1GB) }
    if ($b -ge 1MB) { return "{0:N0} MB" -f ($b / 1MB) }
    return "{0:N0} KB" -f ($b / 1KB)
}
function Minutes([double]$s) {
    if ($s -lt 60) { return "under a minute" }
    $m = [math]::Round($s / 60)
    if ($m -eq 1) { return "about 1 minute" }
    return "about $m minutes"
}
function DownloadMinutes([double]$mbit) { [math]::Max(1, [math]::Round($DownloadBytes * 8 / ($mbit * 1e6) / 60)) }

# ---------------------------------------------------------------- checks before installing
function Get-Checks([string]$dir) {
    $checks = @()
    $gpu = $null
    foreach ($fields in "name,driver_version,memory.total,compute_cap", "name,driver_version,memory.total") {
        try {  # drivers before about 2022 do not know compute_cap
            $smi = & nvidia-smi "--query-gpu=$fields" --format=csv,noheader,nounits 2>$null
            if ($LASTEXITCODE -eq 0 -and $smi) { $gpu = @(($smi | Select-Object -First 1).Split(",") | ForEach-Object { $_.Trim() }); break }
        } catch {}
    }
    if ($gpu) {
        $cc = 99.0  # unknown: the GPU test at the end decides
        if ($gpu.Count -gt 3) { $cc = [double]::Parse($gpu[3], [Globalization.CultureInfo]::InvariantCulture) }
        $drv = [int]($gpu[1].Split(".")[0])
        $name = "$($gpu[0]), $([math]::Round([double]$gpu[2] / 1024)) GB"
        if ($cc -lt 7.5) {
            $checks += @{ ok = $false; text = "$name is too old for this release (it needs a GTX 16xx, RTX 20xx or newer)." }
        } else {
            $checks += @{ ok = $true; text = "NVIDIA GPU: $name" }
        }
        if ($drv -lt 570) {
            $checks += @{ ok = $null; text = "NVIDIA driver $($gpu[1]) is older than 570. Updating it first is recommended; the GPU test at the end will tell." }
        } else {
            $checks += @{ ok = $true; text = "NVIDIA driver $($gpu[1])" }
        }
    } else {
        $names = (Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue | ForEach-Object { $_.Name }) -join ", "
        $checks += @{ ok = $false; text = "No NVIDIA GPU with a working driver found ($names). GPUCoadd needs an NVIDIA GPU; AMD and Intel are not supported." }
    }
    $checks += @{ ok = [Environment]::Is64BitOperatingSystem; text = "Windows 64-bit" }
    if (-not $dir.Trim()) {
        $checks += @{ ok = $false; text = "Install folder: not chosen yet. Click Change... or type a folder." }
        return , $checks
    }
    # Setup records what it installs and uninstalling removes only that, so other files may share
    # the folder; but nothing of theirs may carry a name GPUCoadd would overwrite.
    try {
        $full = [IO.Path]::GetFullPath($dir)
        $ours = Test-Path (Join-Path $full $RecordName)
        $clash = @()
        if ((Test-Path $full) -and -not $ours) { $clash = @($OwnNames | Where-Object { Test-Path (Join-Path $full $_) }) }
        if ([IO.Path]::GetPathRoot($full).TrimEnd('\') -eq $full.TrimEnd('\')) {
            $checks += @{ ok = $false; text = "Choose a folder, not a whole drive." }
        } elseif ($clash.Count) {
            $checks += @{ ok = $false; text = "$full already has $($clash -join ', '), which GPUCoadd would overwrite. Choose another folder." }
        } else {
            $probe = $full
            while (-not (Test-Path $probe)) { $probe = Split-Path -Parent $probe }
            $test = Join-Path $probe ("gpucoadd-write-test-" + [Guid]::NewGuid().ToString("N"))
            [IO.File]::WriteAllText($test, ""); Remove-Item $test
            $what = "a new folder"
            if ($ours) { $what = "an earlier GPUCoadd install, updated in place" }
            elseif ((Test-Path $full) -and (Get-ChildItem -Force $full | Select-Object -First 1)) { $what = "next to the files already there (uninstalling leaves those alone)" }
            elseif (Test-Path $full) { $what = "an empty folder" }
            $checks += @{ ok = $true; text = "Install folder: $what, writable" }
        }
    } catch {
        $checks += @{ ok = $false; text = "Setup cannot write to that folder (it may need admin rights, like Program Files). Choose a folder in your user profile or on another drive." }
    }
    try {
        $root = [IO.Path]::GetPathRoot([IO.Path]::GetFullPath($dir))
        $free = (New-Object IO.DriveInfo $root).AvailableFreeSpace
        if (Test-Path (Join-Path $dir ".venv\Scripts\python.exe")) {
            $need = $DownloadBytes
            $why = "needs $(Size $need) during setup for the downloads, deleted at the end"
        } else {
            $need = $PeakBytes
            $why = "needs $(Size $need) during setup: $(Size $DownloadBytes) of downloads, deleted at the end, plus $(Size $InstalledBytes) installed"
        }
        $checks += @{ ok = ($free -ge $need); text = "Free space on $($root.TrimEnd('\')) $(Size $free) ($why)" }
    } catch {
        $checks += @{ ok = $false; text = "The install folder is not on a usable drive." }
    }
    return , $checks
}

# Shared by the uninstall window and a stopped setup (which runs it in its own thread): what setup's
# record says it installed, and removing exactly that.
$Shared = @'
function Get-UninstallPlan([string]$dir) {
    # What setup's record lists (files, and folders it created whole) plus the __pycache__ folders
    # Python writes next to those files; everything else in the folder is someone else's and stays.
    $dir = $dir.TrimEnd("\")
    $files = @(); $owned = @()
    foreach ($line in Get-Content -Encoding UTF8 (Join-Path $dir $RecordName)) {
        if (-not $line.Trim() -or $line.StartsWith("#")) { continue }
        if ($line.EndsWith("\")) { $owned += Join-Path $dir $line.TrimEnd("\") } else { $files += Join-Path $dir $line }
    }
    $parents = @($files | ForEach-Object { Split-Path -Parent $_ } | Sort-Object -Unique)
    $caches = @($parents | ForEach-Object { Join-Path $_ "__pycache__" } | Where-Object { Test-Path -LiteralPath $_ })
    $set = New-Object 'Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
    $bytes = [int64]0
    foreach ($f in $files) { if (Test-Path -LiteralPath $f -PathType Leaf) { [void]$set.Add($f); $bytes += (Get-Item -Force -LiteralPath $f).Length } }
    foreach ($d in ($owned + $caches)) {
        if (Test-Path -LiteralPath $d) { foreach ($x in Get-ChildItem -Recurse -File -Force -LiteralPath $d) { [void]$set.Add($x.FullName); $bytes += $x.Length } }
    }
    $kept = @(Get-ChildItem -Recurse -File -Force -LiteralPath $dir | Where-Object { -not $set.Contains($_.FullName) } |
        ForEach-Object { $_.FullName.Substring($dir.Length + 1) })
    # folders GPUCoadd's files sit in, deepest first: removed only when nothing is left in them
    $folders = New-Object 'Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
    foreach ($p in $parents) { $q = $p; while ($q.Length -gt $dir.Length) { [void]$folders.Add($q); $q = Split-Path -Parent $q } }
    return @{ dir = $dir; files = $files; dirs = $owned + $caches; bytes = $bytes; count = $set.Count; kept = $kept
        folders = @($folders | Sort-Object Length -Descending) }
}

function Remove-Plan($plan, [bool]$keepDownloads, $tell) {
    # Deletes what Get-UninstallPlan found: the recorded files, the folders setup created whole (all but
    # downloads when keeping them), then GPUCoadd's folders that end up empty, and the install folder
    # only when nothing at all is left in it. $tell gets each path as it goes.
    $count = 0; $bytes = [int64]0
    foreach ($f in $plan.files) {
        if (Test-Path -LiteralPath $f -PathType Leaf) {
            $bytes += (Get-Item -Force -LiteralPath $f).Length; Remove-Item -LiteralPath $f -Force; $count++
            if ($tell) { & $tell $f }
        }
    }
    foreach ($d in $plan.dirs) {
        if ($keepDownloads -and $d -eq (Join-Path $plan.dir "downloads")) { continue }
        if (Test-Path -LiteralPath $d) {
            $items = @(Get-ChildItem -Recurse -File -Force -LiteralPath $d)
            $count += $items.Count; $bytes += [int64](($items | Measure-Object Length -Sum).Sum)
            Remove-Item -LiteralPath $d -Recurse -Force
            if ($tell) { & $tell "$d ($($items.Count) files)" }
        }
    }
    foreach ($d in $plan.folders) {
        if ((Test-Path -LiteralPath $d) -and -not (Get-ChildItem -Force -LiteralPath $d)) { Remove-Item -LiteralPath $d; if ($tell) { & $tell $d } }
    }
    $folderGone = $false
    if ((Test-Path -LiteralPath $plan.dir) -and -not (Get-ChildItem -Force -LiteralPath $plan.dir)) {
        Remove-Item -LiteralPath $plan.dir; $folderGone = $true; if ($tell) { & $tell $plan.dir }
    }
    return @{ count = $count; bytes = $bytes; folderGone = $folderGone }
}

function Get-StartShortcut { Join-Path ([Environment]::GetFolderPath("Programs")) "GPUCoadd.lnk" }

function Remove-StartShortcut([string]$dir) {
    # The Start menu entry, only when it opens this install: another install's entry stays.
    $lnk = Get-StartShortcut
    if (-not (Test-Path -LiteralPath $lnk)) { return $false }
    $opens = [string](New-Object -ComObject WScript.Shell).CreateShortcut($lnk).Arguments
    $mine = $dir.TrimEnd("\") + "\start_gpucoadd.pyw"  # text only: the drive may be gone
    if (-not $opens -or $opens.IndexOf($mine, [StringComparison]::OrdinalIgnoreCase) -lt 0) { return $false }
    Remove-Item -LiteralPath $lnk -Force
    return $true
}
'@
. ([scriptblock]::Create($Shared))

# ---------------------------------------------------------------- the work, in its own thread
$Work = {
    param($S, $M, $Dir, $Here, $DryRun, $LocalFiles, $Shared)
    $ErrorActionPreference = "Stop"
    function MB([double]$b) { if ($b -ge 1GB) { "{0:N1} GB" -f ($b / 1GB) } else { "{0:N1} MB" -f ($b / 1MB) } }
    function Log([string]$t) { $S.Log.Enqueue((Get-Date -Format "HH:mm:ss") + "  " + $t) }
    function Step([int]$k) { $S.States[$k] = "running"; $S.Step = $k; $S.StepStart = [DateTime]::Now; $S.Done = 0; $S.Total = 0 }
    function Finish([int]$k, [string]$note) { $S.States[$k] = "done"; $S.Notes[$k] = $note }
    function Check() { if ($S.Cancel) { throw "cancelled" } }
    function Run([string]$exe, [string[]]$argv) {
        # Stop ends the program at once: a half-finished install is always redone from scratch next time
        $p = New-Object Diagnostics.Process
        $p.StartInfo.FileName = $exe
        $p.StartInfo.Arguments = ($argv | ForEach-Object { if ($_ -match '[\s"]') { '"' + $_.Replace('"', '\"') + '"' } else { $_ } }) -join " "
        $p.StartInfo.UseShellExecute = $false
        $p.StartInfo.CreateNoWindow = $true
        $p.StartInfo.RedirectStandardOutput = $true
        $p.StartInfo.RedirectStandardError = $true
        [void]$p.Start()
        $err = $p.StandardError.ReadToEndAsync()
        $next = $p.StandardOutput.ReadLineAsync()
        while ($true) {
            if ($next.Wait(250)) {
                $line = $next.Result
                if ($null -eq $line) { break }
                if ($line.Trim()) { Log "    $line" }
                if ($S.OnLine) { & $S.OnLine $line }
                $next = $p.StandardOutput.ReadLineAsync()
            } elseif ($S.Cancel) {
                try { $p.Kill() } catch {}
                $p.WaitForExit()
                throw "cancelled"
            }
        }
        $p.WaitForExit()
        foreach ($l in ($err.Result -split "`r?`n")) { if ($l.Trim()) { Log "    $l"; $S.LastError = $l.Trim() } }
        return $p.ExitCode
    }
    function Reason($e) {
        # the innermost, plain cause of a failed request ("The remote name could not be resolved", ...)
        $x = $e
        while ($x.InnerException) { $x = $x.InnerException }
        return $x.Message.Trim().TrimEnd(".")
    }
    function Wait-Task($task, [int]$seconds, [string]$what) {
        # waits for a network task in short slices, so Stop works and a dead connection is noticed
        $waited = 0
        while (-not $task.Wait(500)) {
            if ($S.Cancel) { throw "cancelled" }
            $waited += 500
            if ($waited -ge $seconds * 1000) { throw "no $what arrived for $seconds seconds" }
        }
        return $task.Result
    }
    function Get-Download($f, [string]$dest, [ref]$doneBefore) {
        $part = "$dest.part"
        if (Test-Path $dest) {  # verified on an earlier run, before it got this name
            if ((Get-Item $dest).Length -eq $f.size) { Log "  $($f.file): already downloaded"; $S.Done = $doneBefore.Value + $f.size; return }
            Remove-Item $dest
        }
        if ($LocalFiles -and (Test-Path (Join-Path $LocalFiles $f.file))) {
            Copy-Item (Join-Path $LocalFiles $f.file) $part -Force
        }
        $line = $S.Detail
        $host_ = ([Uri]$f.url).Host
        $sha = [Security.Cryptography.SHA256]::Create()
        $have = [int64]0
        if (Test-Path $part) {  # kept from a stopped run: checked again from its first byte
            $have = (Get-Item $part).Length
            if ($have -gt $f.size) { Remove-Item $part; $have = 0 }
            else {
                $S.Detail = "$line (checking the $([math]::Round($have / 1MB)) MB downloaded earlier)"
                $in = [IO.File]::OpenRead($part); $buf = New-Object byte[] (4MB)
                while (($n = $in.Read($buf, 0, $buf.Length)) -gt 0) { [void]$sha.TransformBlock($buf, 0, $n, $null, 0); Check }
                $in.Close()
                if ($have -gt 0) { Log "  $($f.file): continuing after $([math]::Round($have / 1MB)) MB already downloaded" }
            }
        }
        $S.Done = $doneBefore.Value + $have
        $waits = @(5, 10, 20, 30, 60)  # seconds before each new try; about two minutes in all
        $attempt = 0
        while ($have -lt $f.size) {
            $attempt++
            $S.Detail = $line
            $resp = $null; $in = $null; $out = $null
            try {
                $req = New-Object Net.Http.HttpRequestMessage ([Net.Http.HttpMethod]::Get), $f.url
                if ($have -gt 0) { $req.Headers.Range = New-Object Net.Http.Headers.RangeHeaderValue ([int64]$have), $null }
                if ($attempt -gt 1) { Log "  trying $($f.file) again from $([math]::Round($have / 1MB)) MB" }
                $resp = Wait-Task ($S.Http.SendAsync($req, [Net.Http.HttpCompletionOption]::ResponseHeadersRead)) 60 "answer from $host_"
                if (-not $resp.IsSuccessStatusCode) { throw "$host_ answered HTTP $([int]$resp.StatusCode)" }
                if ($have -gt 0 -and [int]$resp.StatusCode -ne 206) {  # the server sent the whole file again
                    $sha.Dispose(); $sha = [Security.Cryptography.SHA256]::Create(); $have = 0
                    Remove-Item $part; $S.Done = $doneBefore.Value
                }
                $in = Wait-Task ($resp.Content.ReadAsStreamAsync()) 60 "data from $host_"
                $out = [IO.File]::Open($part, [IO.FileMode]::Append)
                $buf = New-Object byte[] (1MB)
                while (($n = Wait-Task ($in.ReadAsync($buf, 0, $buf.Length)) 30 "data from $host_") -gt 0) {
                    $out.Write($buf, 0, $n); [void]$sha.TransformBlock($buf, 0, $n, $null, 0)
                    $have += $n; $S.Done = $doneBefore.Value + $have
                    Check  # Stop also works while data keeps arriving
                }
                if ($have -lt $f.size) { throw "$host_ ended the transfer early" }
            } catch {
                $why = Reason $_.Exception
                if ($why -eq "cancelled" -or $S.Cancel) { throw "cancelled" }
                if ($attempt -gt $waits.Count) {
                    throw "The download from $host_ kept failing ($why). Check your internet connection and run setup again: what was downloaded so far is kept, and it continues from there."
                }
                Log "  $($f.file): connection problem after $([math]::Round($have / 1MB)) MB ($why); trying again in $($waits[$attempt - 1]) s"
                for ($left = $waits[$attempt - 1]; $left -gt 0; $left--) {
                    $S.Detail = "Connection to $host_ lost ($why). Trying again in $left s (try $($attempt + 1) of $($waits.Count + 1)). Nothing downloaded so far is lost."
                    Start-Sleep -Seconds 1
                    Check
                }
            } finally {
                if ($out) { $out.Close() }
                if ($in) { try { $in.Dispose() } catch {} }
                if ($resp) { try { $resp.Dispose() } catch {} }
            }
        }
        [void]$sha.TransformFinalBlock((New-Object byte[] 0), 0, 0)
        $hex = -join ($sha.Hash | ForEach-Object { $_.ToString("x2") })
        if ($have -ne $f.size -or $hex -ne $f.sha256) {
            Remove-Item $part -ErrorAction SilentlyContinue
            throw "$($f.file) did not match its expected SHA-256 checksum, so it was deleted (the download was damaged on the way). Run setup again to download it again."
        }
        Move-Item $part $dest -Force
        Log "  $($f.file): $([math]::Round($f.size / 1MB, 1)) MB, SHA-256 verified"
    }
    function Simulate([string]$label, [int64]$bytes, [double]$rate) {
        $done = 0
        while ($done -lt $bytes) { Start-Sleep -Milliseconds 100; Check; $done = [math]::Min([double]$bytes, $done + $rate / 10); $S.Done += $rate / 10 }
    }

    function Package-Summary($f, [string]$site) {
        # where pip put a package, from its own RECORD: "N files, X MB in <folder>"
        $norm = $f.name.ToLower() -replace '[-_.]+', '_'
        $info = Get-ChildItem -Directory -LiteralPath $site -Filter "*.dist-info" |
            Where-Object { (($_.Name -replace '-[^-]+\.dist-info$', '').ToLower() -replace '[-_.]+', '_') -eq $norm } | Select-Object -First 1
        if (-not $info) { return "location unknown" }
        $rows = @(Get-Content -LiteralPath (Join-Path $info.FullName "RECORD"))
        $bytes = [int64]0; $tops = @{}
        foreach ($r in $rows) {
            $parts = $r.Split(",")
            if ($parts.Count -ge 3 -and $parts[-1]) { $bytes += [int64]$parts[-1] }
            $top = $parts[0].Split("/")[0]
            if ($top -and $top -ne ".." -and $top -notlike "*.dist-info" -and $top -ne "__pycache__") { $tops[$top] = 1 }
        }
        $where = (@($tops.Keys | Sort-Object | ForEach-Object { Join-Path $site $_ }) + $info.FullName) -join ", "
        return "$($rows.Count) files, $(MB $bytes): $where"
    }

    $RecordName = "setup\installed-files.txt"
    . ([scriptblock]::Create($Shared))
    $workDir = Join-Path $env:LOCALAPPDATA "GPUCoadd"
    $S.WorkExisted = Test-Path $workDir
    $files = @($M.python) + @($M.packages)
    $downloads = Join-Path $Dir "downloads"
    $setupDir = Join-Path $Dir "setup"
    $marker = Join-Path $setupDir "installed.json"
    $record = Join-Path $setupDir "installed-files.txt"
    $appsKey = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\GPUCoadd"
    function Write-Record {
        # Everything setup put in the install folder. Uninstalling removes exactly this (plus the
        # __pycache__ folders Python adds next to these files) and leaves anything else alone.
        $lines = @("# GPUCoadd $($S.Version): what setup installed in this folder. Uninstalling removes exactly this.",
            "# A line ending in \ is a folder setup created; it goes with everything in it.")
        foreach ($d in ".venv", "python", "downloads", "setup") { if (Test-Path (Join-Path $Dir $d)) { $lines += "$d\" } }
        $lines += $S.AppFiles
        if (Test-Path (Join-Path $Dir "Uninstall GPUCoadd.cmd")) { $lines += "Uninstall GPUCoadd.cmd" }
        $lines | Set-Content -Encoding UTF8 $record
    }

    try {
        $keep = $false
        if (-not $DryRun -and (Test-Path $marker) -and (Test-Path (Join-Path $Dir ".venv\Scripts\python.exe"))) {
            $keep = ((Get-Content -Raw $marker | ConvertFrom-Json).requirements_sha256 -eq $M.requirements_sha256)
        }

        # 1. GPUCoadd's own files
        Step 0
        Log "Install folder: $Dir"
        if (-not $DryRun) {
            New-Item -ItemType Directory -Force $Dir, $setupDir | Out-Null
            if (Test-Path $record) {  # the previous version's own files, never anything else
                $parents = @()
                foreach ($rel in Get-Content $record) {
                    if ($rel.StartsWith("#") -or $rel.EndsWith("\")) { continue }
                    $p = Join-Path $Dir $rel
                    if (Test-Path $p -PathType Leaf) { Remove-Item $p; Log "  - $p (previous version)" }
                    $parents += Split-Path -Parent $p
                }
                # then their bytecode caches, and the folders they were in once nothing is left there
                $root = $Dir.TrimEnd("\"); $folders = @()
                foreach ($q in @($parents | Sort-Object -Unique)) {
                    $cache = Join-Path $q "__pycache__"
                    if (Test-Path -LiteralPath $cache) { Remove-Item -LiteralPath $cache -Recurse -Force }
                    while ($q.Length -gt $root.Length) { $folders += $q; $q = Split-Path -Parent $q }
                }
                foreach ($d in @($folders | Sort-Object -Unique | Sort-Object Length -Descending)) {
                    if ((Test-Path -LiteralPath $d) -and -not (Get-ChildItem -Force -LiteralPath $d)) {
                        Remove-Item -LiteralPath $d; Log "  - $d (empty, previous version)"
                    }
                }
            }
            Add-Type -AssemblyName System.IO.Compression.FileSystem
            $zip = [IO.Compression.ZipFile]::OpenRead((Join-Path $Here "app.zip"))
            $names = @()
            foreach ($e in $zip.Entries) {
                if (-not $e.Name) { continue }
                $target = Join-Path $Dir $e.FullName
                New-Item -ItemType Directory -Force (Split-Path -Parent $target) | Out-Null
                [IO.Compression.ZipFileExtensions]::ExtractToFile($e, $target, $true)
                $names += $e.FullName.Replace("/", "\")
                $S.AppFiles = $names
                Log "  + $target"
            }
            $zip.Dispose()
            Copy-Item (Join-Path $Here "Setup.ps1"), (Join-Path $Here "packages.json") $setupDir -Force
            Log "  + $setupDir\Setup.ps1, packages.json (used by Uninstall)"
            # the uninstaller runs from %TEMP%, so nothing holds the folder it cleans up
            @('@cd /d "%TEMP%"', '@mkdir "%TEMP%\GPUCoadd-uninstall" 2>nul',
              '@copy /y "%~dp0setup\Setup.ps1" "%TEMP%\GPUCoadd-uninstall\" >nul',
              '@copy /y "%~dp0setup\packages.json" "%TEMP%\GPUCoadd-uninstall\" >nul',
              '@start "" powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "%TEMP%\GPUCoadd-uninstall\Setup.ps1" -Uninstall -InstallDir "%~dp0."') |
                Set-Content -Encoding ASCII (Join-Path $Dir "Uninstall GPUCoadd.cmd")
            Log "  + $Dir\Uninstall GPUCoadd.cmd"
            Write-Record
            Log "  + $record (the record of everything setup installs)"
        } else { Start-Sleep -Milliseconds 600 }
        if ($S.AppFiles) { Finish 0 "$(@($S.AppFiles).Count) files" } else { Finish 0 "done" }

        # 2. downloads
        Step 1
        if ($keep) {
            Log "Python and the packages of this version are already installed: nothing to download"
            Finish 1 "already installed, kept"
        } else {
            $S.Total = [int64](($files | Measure-Object -Property size -Sum).Sum)
            if (-not $DryRun) { New-Item -ItemType Directory -Force $downloads | Out-Null; Write-Record }
            $before = [int64]0
            $k = 0
            foreach ($f in $files) {
                $k++
                Check
                $S.Detail = "File $k of $($files.Count): $($f.name) $($f.version) ($(MB $f.size)) from $($f.source)"
                if ($DryRun) { Simulate $f.file $f.size 150MB } else {
                    $dest = Join-Path $downloads $f.file
                    Log "Downloading $($f.url)"
                    Log "  to $dest"
                    Get-Download $f $dest ([ref]$before)
                }
                $before += $f.size; $S.Done = $before
            }
            Finish 1 "all $($files.Count) files verified"
        }

        # 3. install
        Step 2
        if ($keep) { Finish 2 "kept" } else {
            $S.Total = [int64](($M.packages | Measure-Object -Property size -Sum).Sum) + $M.python.size
            $site = Join-Path $Dir ".venv\Lib\site-packages"
            if (-not $DryRun) {
                foreach ($old in ".venv", "python") {
                    $p = Join-Path $Dir $old
                    if (Test-Path $p) { Remove-Item -Recurse -Force $p; Log "  - $p (an earlier or unfinished install)" }
                }
                $S.Detail = "Unpacking Python $($M.python.version)"
                if ((Run "$env:WINDIR\System32\tar.exe" @("-xzf", (Join-Path $downloads $M.python.file), "-C", $Dir)) -ne 0) { throw "Python could not be unpacked." }
                Write-Record
                $py = @(Get-ChildItem -Recurse -File (Join-Path $Dir "python"))
                Log "  + $Dir\python: Python $($M.python.version), $($py.Count) files, $(MB (($py | Measure-Object Length -Sum).Sum))"
                $S.Detail = "Creating GPUCoadd's Python environment"
                if ((Run (Join-Path $Dir "python\python.exe") @("-m", "venv", (Join-Path $Dir ".venv"))) -ne 0) { throw "The Python environment could not be created." }
                Write-Record
                Log "  + $Dir\.venv: GPUCoadd's Python environment; packages go to $site"
            } else { Start-Sleep -Milliseconds 800 }
            $S.Done = $M.python.size
            $python = Join-Path $Dir ".venv\Scripts\python.exe"
            $k = 0
            foreach ($f in ($M.packages | Sort-Object size)) {
                $k++
                Check
                $note = ""
                if ($f.size -gt 1GB) { $note = ", the biggest one: 1 to 4 minutes" }
                $S.PackageStart = [DateTime]::Now
                $S.Detail = "Package $k of $($M.packages.Count): $($f.name) $($f.version) ($(MB $f.size)$note)"
                if ($DryRun) { Simulate $f.file $f.size 400MB } else {
                    $code = Run $python @("-m", "pip", "install", "--no-deps", "--no-index", "--no-compile", "--disable-pip-version-check",
                        "--no-warn-script-location", "--quiet", (Join-Path $downloads $f.file))
                    if ($code -ne 0) { throw "$($f.name) could not be installed (pip exit code $code)." }
                    Log "  + $($f.name) $($f.version): $(Package-Summary $f $site)"
                    $S.Done += $f.size
                }
            }
            if (-not $DryRun) {
                $S.Detail = "Checking that every package has what it needs"
                if ((Run $python @("-m", "pip", "check", "--disable-pip-version-check")) -ne 0) { throw "The installed packages are incomplete (pip check failed)." }
                Log "Every package has what it needs (pip check)"
            }
            Finish 2 "Python $($M.python.version) and $($M.packages.Count) packages"
        }

        # 4. GPU self-test
        Step 3
        $S.Detail = "Aligning and stacking 6 synthetic lights"
        if ($DryRun) { Start-Sleep -Seconds 2; $S.SelfTest = "6 synthetic lights aligned and stacked; stars within 0.03 px of their true positions." } else {
            $S.OnLine = {
                param($line)
                try { $e = $line | ConvertFrom-Json } catch { return }
                if ($e.message) { $S.Detail = $e.message }
                if ($e.result) { $S.SelfTestEvent = $e }
            }
            [void](Run (Join-Path $Dir ".venv\Scripts\python.exe") @((Join-Path $Dir "worker\selftest.py")))
            $S.OnLine = $null
            $e = $S.SelfTestEvent
            # "slow": every check passed, but a GPU kernel could not be built and its stage runs on a
            # slower path. GPUCoadd works, so setup completes and says so.
            if (-not $e -or ($e.result -ne "ok" -and $e.result -ne "slow")) {
                $why = "The GPU self-test could not run: $($S.LastError)"
                if ($e) { $why = "$($e.summary) $($e.next_step)" }
                throw $why
            }
            $S.SelfTest = $e.summary
            if ($e.result -eq "slow") {
                $S.SelfTest = "$($e.summary) $($e.next_step)"
                Log "  ! $($e.summary) $($e.error)"
            }
            Log "  + $workDir\cubin_cache and kernel_cache: compiled GPU kernels, reused by GPUCoadd"
        }
        Finish 3 "passed"

        # 5. finish
        Step 4
        if (-not $DryRun) {
            $S.Detail = "Adding GPUCoadd to Windows' installed apps"
            $uninstall = Join-Path $Dir "Uninstall GPUCoadd.cmd"
            $size = [int64]((Get-ChildItem -Recurse -File -Force $Dir -ErrorAction SilentlyContinue | Where-Object { $_.FullName -notlike "$downloads*" } | Measure-Object Length -Sum).Sum)
            New-Item -Force $appsKey | Out-Null
            $props = @{ DisplayName = "GPUCoadd"; DisplayVersion = $S.Version; Publisher = "GPUCoadd"; InstallLocation = $Dir
                UninstallString = "cmd.exe /c `"$uninstall`""; NoModify = 1; NoRepair = 1; EstimatedSize = [int]($size / 1KB)
                DisplayIcon = (Join-Path $Dir "GPUCoadd\gpucoadd.ico") }
            foreach ($p in $props.Keys) { New-ItemProperty -Force -Path $appsKey -Name $p -Value $props[$p] | Out-Null }
            Log "  + Windows installed-apps entry ($($appsKey.Replace('HKCU:', 'HKEY_CURRENT_USER')))"
            # the window on its own, without a console: GPUCoadd's pythonw runs the launcher
            $S.Detail = "Adding GPUCoadd to the Start menu"
            $lnk = Get-StartShortcut
            $shortcut = (New-Object -ComObject WScript.Shell).CreateShortcut($lnk)
            $shortcut.TargetPath = Join-Path $Dir ".venv\Scripts\pythonw.exe"
            $shortcut.Arguments = '"' + (Join-Path $Dir "start_gpucoadd.pyw") + '"'
            $shortcut.WorkingDirectory = $Dir
            $shortcut.IconLocation = (Join-Path $Dir "GPUCoadd\gpucoadd.ico") + ",0"
            $shortcut.Description = "GPUCoadd: preview, frame and stack Seestar lights"
            $shortcut.Save()
            Log "  + Start menu entry: $lnk"
            if (Test-Path $downloads) {
                $dl = [int64]((Get-ChildItem $downloads -File | Measure-Object Length -Sum).Sum)
                $S.Detail = "Deleting the downloaded files ($(MB $dl)), no longer needed"
                Remove-Item -Recurse -Force $downloads
                Log "  - $downloads ($(MB $dl) of downloads, no longer needed)"
            }
            @{ version = $S.Version; requirements_sha256 = $M.requirements_sha256; installed = (Get-Date).ToString("s") } |
                ConvertTo-Json | Set-Content -Encoding UTF8 $marker
            Write-Record
            $S.InstalledSize = $size
            Log "Installed: $(MB $size) in $Dir"
        } else { Start-Sleep -Milliseconds 600; $S.InstalledSize = 4.9GB }
        Finish 4 "done"
        $S.Result = "ok"
    } catch {
        $msg = $_.Exception.Message
        if ($_.Exception.InnerException) { $msg = $_.Exception.InnerException.Message }
        $cancelled = ($msg -eq "cancelled") -or $S.Cancel
        if ($cancelled) { $S.States[$S.Step] = "stopped"; Log "Stopped by you" } else { $S.States[$S.Step] = "failed"; $S.Error = $msg; Log "Stopped: $msg" }
        if (-not $DryRun -and $S.AppFiles) { try { Write-Record } catch {} }  # so Uninstall can always undo it
        if ($cancelled -and $S.CancelMode) {
            # undo: everything the record lists, the installed-apps entry, and the test's cache folder if setup made it
            $S.Undoing = $true; $S.Total = 0
            $S.Detail = "Removing what setup added to $Dir"
            Log "Removing what setup added:"
            try {
                if ($DryRun) { Start-Sleep -Seconds 1; $S.Rollback = @{ count = 41; bytes = 220MB; folderGone = $true; keptDownloads = [int64]0 } }
                else {
                    $keepDl = ($S.CancelMode -eq "keep") -and (Test-Path $downloads)
                    $kept = [int64]0
                    if ($keepDl) { $kept = [int64]((Get-ChildItem -File $downloads | Measure-Object Length -Sum).Sum) }
                    $r = @{ count = 0; bytes = [int64]0; folderGone = $false }
                    if (Test-Path $record) { $r = Remove-Plan (Get-UninstallPlan $Dir) $keepDl { param($p) Log "  - $p" } }
                    elseif ((Test-Path $Dir) -and -not (Get-ChildItem -Force $Dir)) { Remove-Item $Dir; $r.folderGone = $true; Log "  - $Dir" }
                    if (Test-Path $appsKey) { Remove-Item -Force $appsKey; Log "  - Windows installed-apps entry" }
                    if (Remove-StartShortcut $Dir) { Log "  - Start menu entry" }
                    if (-not $S.WorkExisted -and (Test-Path $workDir)) {
                        Remove-Item -Recurse -Force $workDir; Log "  - $workDir (the GPU test's kernel cache)"
                    }
                    if ($keepDl) {  # a later setup run recognizes these as its own and continues from them
                        New-Item -ItemType Directory -Force $setupDir | Out-Null
                        @("# GPUCoadd $($S.Version): downloads kept from a stopped setup. Running setup again continues from them.",
                          "downloads\", "setup\") | Set-Content -Encoding UTF8 $record
                        Log "  kept: $downloads ($(MB $kept)) for the next setup run"
                    }
                    $r.keptDownloads = $kept
                    $S.Rollback = $r
                }
            } catch { $S.RollbackError = $_.Exception.Message; Log "Could not remove everything: $($S.RollbackError)" }
        }
        if ($cancelled) { $S.Result = "cancelled" } else { $S.Result = "failed" }
    }
}

# ---------------------------------------------------------------- the window
$Font = New-Object Drawing.Font("Segoe UI", 10)
$Bold = New-Object Drawing.Font("Segoe UI", 10, [Drawing.FontStyle]::Bold)
$Title = New-Object Drawing.Font("Segoe UI", 15)
$Green = [Drawing.Color]::FromArgb(16, 124, 16)
$Red = [Drawing.Color]::FromArgb(196, 43, 28)
$Amber = [Drawing.Color]::FromArgb(157, 93, 0)
$Grey = [Drawing.Color]::FromArgb(96, 96, 96)

$form = New-Object Windows.Forms.Form
$form.Text = "GPUCoadd $Version setup"
if ($Uninstall) { $form.Text = "Remove GPUCoadd" }
$form.Font = $Font
$form.ClientSize = New-Object Drawing.Size(760, 750)
$form.FormBorderStyle = "FixedDialog"
$form.MaximizeBox = $false
$form.StartPosition = "CenterScreen"
$form.BackColor = [Drawing.Color]::White

function Label([string]$text, $font = $Font, $color = $null) {
    $l = New-Object Windows.Forms.Label
    $l.Text = $text; $l.Font = $font; $l.AutoSize = $true; $l.MaximumSize = New-Object Drawing.Size(700, 0)
    $l.Margin = New-Object Windows.Forms.Padding(0, 0, 0, 6)
    if ($color) { $l.ForeColor = $color }
    return $l
}
function Page() {
    $p = New-Object Windows.Forms.FlowLayoutPanel
    $p.FlowDirection = "TopDown"; $p.WrapContents = $false; $p.AutoScroll = $true
    $p.Location = New-Object Drawing.Point(28, 20); $p.Size = New-Object Drawing.Size(712, 660)
    $p.Visible = $false
    $form.Controls.Add($p)
    return $p
}
$buttons = New-Object Windows.Forms.FlowLayoutPanel
$buttons.FlowDirection = "RightToLeft"; $buttons.Location = New-Object Drawing.Point(20, 694); $buttons.Size = New-Object Drawing.Size(720, 44)
$form.Controls.Add($buttons)
function Button([string]$text) {
    $b = New-Object Windows.Forms.Button
    $b.Text = $text; $b.AutoSize = $true; $b.MinimumSize = New-Object Drawing.Size(110, 34); $b.Font = $Font
    $buttons.Controls.Add($b)
    return $b
}
function Show-Page($page) { foreach ($c in $form.Controls) { if ($c -is [Windows.Forms.FlowLayoutPanel] -and $c -ne $buttons) { $c.Visible = $false } }; $page.Visible = $true }
function Snap([string]$name) {
    if (-not $Snapshot) { return }
    New-Item -ItemType Directory -Force $Snapshot | Out-Null
    $form.Refresh()
    $bmp = New-Object Drawing.Bitmap($form.Width, $form.Height)
    $form.DrawToBitmap($bmp, (New-Object Drawing.Rectangle(0, 0, $form.Width, $form.Height)))
    $bmp.Save((Join-Path $Snapshot "$name.png")); $bmp.Dispose()
    if ($bar) { Add-Content (Join-Path $Snapshot "bars.txt") "$name  step bar $($bar.Value)/1000  overall $($overallBar.Value)/1000" }
}

# ---------------------------------------------------------------- uninstall
if ($Uninstall) {
    $p = Page
    $p.Controls.Add((Label "Remove GPUCoadd" $Title))
    $plan = $null
    if (Test-Path (Join-Path $InstallDir $RecordName)) { $plan = Get-UninstallPlan $InstallDir }
    if ($plan) {
        $p.Controls.Add((Label "Setup kept a record of everything it installed. Removing GPUCoadd deletes exactly that, its Start menu entry and its entry in Windows' installed apps:"))
        $p.Controls.Add((Label "$($plan.count) files, $(Size $plan.bytes), in $InstallDir" $Bold))
        if ($plan.kept.Count) {
            $show = ($plan.kept | Select-Object -First 3) -join ", "
            if ($plan.kept.Count -gt 3) { $show += ", ..." }
            $p.Controls.Add((Label "Stays: $($plan.kept.Count) file(s) setup did not install ($show). The folder stays with them."))
        } else {
            $p.Controls.Add((Label "Nothing else is in that folder, so the folder goes too."))
        }
    }
    $work = [int64]0
    if (Test-Path $WorkDir) { $work = [int64]((Get-ChildItem -Recurse -File -Force $WorkDir -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum) }
    $also = New-Object Windows.Forms.CheckBox
    $also.Text = "Also delete GPUCoadd's work folder ($(Size $work))"
    $also.AutoSize = $true; $also.Enabled = ($work -gt 0)
    $p.Controls.Add($also)
    # a check box doesn't wrap its text, so the details go in a label under it
    $alsoDetail = Label "Saved heat maps, run reports, error reports and the GPU kernel cache, in $WorkDir" $Font $Grey
    $alsoDetail.Margin = New-Object Windows.Forms.Padding(18, 0, 0, 6); $alsoDetail.MaximumSize = New-Object Drawing.Size(680, 0)
    $p.Controls.Add($alsoDetail)
    $p.Controls.Add((Label "Your lights and stacked images are never touched." $Font $Grey))
    $result = Label "" $Bold
    $p.Controls.Add($result)
    $cancel = Button "Cancel"; $cancel.Add_Click({ $form.Close() })
    $remove = Button "Remove"
    if (-not $plan) {
        $remove.Enabled = $false; $also.Enabled = $false
        $result.ForeColor = $Red; $result.Text = "$InstallDir has no record of a GPUCoadd install, so nothing is removed."
    }
    $remove.Add_Click({
        $remove.Enabled = $false; $cancel.Enabled = $false; $form.Cursor = "WaitCursor"
        $result.ForeColor = [Drawing.Color]::Black; $result.Text = "Removing... this can take up to a minute."
        [Windows.Forms.Application]::DoEvents()
        try {
            Set-Location $env:TEMP
            Remove-Item -Force "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\GPUCoadd" -ErrorAction SilentlyContinue
            [void](Remove-StartShortcut $InstallDir)
            $r = Remove-Plan $plan $false { param($p) [Windows.Forms.Application]::DoEvents() }
            $gone = "GPUCoadd is removed: $($r.count) files, $(Size $r.bytes)."
            if (-not $r.folderGone) { $gone += " The folder stays with the files setup did not install." }
            if ($also.Checked) { Remove-Item -Recurse -Force $WorkDir }
            $result.ForeColor = $Green; $result.Text = $gone
        } catch {
            $result.ForeColor = $Red; $result.Text = "Some files could not be deleted: $($_.Exception.Message) Close GPUCoadd and Siril and try again."
            $remove.Enabled = $true
        }
        $form.Cursor = "Default"; $cancel.Text = "Close"; $cancel.Enabled = $true
    })
    Show-Page $p
    if ($Snapshot) { $form.Add_Shown({ Snap "uninstall"; $form.Close() }) }
    [void]$form.ShowDialog()
    return
}

# ---------------------------------------------------------------- page 1: before installing
$intro = Page
$intro.Controls.Add((Label "Install GPUCoadd $Version" $Title))
$intro.Controls.Add((Label "What setup will do" $Bold))
$what = @(
    "1.  Check your NVIDIA GPU, driver and free space. Nothing is installed if they don't fit.",
    "2.  Copy GPUCoadd's own files (under 2 MB) into the folder below.",
    "3.  Download $(Size $DownloadBytes): Python $($Manifest.python.version) and $($Manifest.packages.Count) packages it needs. Most of it ($(Size $Biggest.size)) is PyTorch with NVIDIA's CUDA libraries. Sources: pypi.org, download.pytorch.org, github.com. Every file's checksum is verified.",
    "4.  Install them into the same folder: about $(Size $InstalledBytes) on disk. No admin rights needed.",
    "5.  Test the GPU by stacking a few synthetic lights (under a minute).",
    "6.  Add GPUCoadd to the Start menu and to Windows' installed apps, and delete the downloaded files."
)
foreach ($t in $what) { $intro.Controls.Add((Label $t)) }
$intro.Controls.Add((Label "How long: the download takes about $(DownloadMinutes 100) minutes at 100 Mbit/s, $(DownloadMinutes 50) at 50 and $(DownloadMinutes 25) at 25. Installing and testing take about 2 more minutes." $Font))
$intro.Controls.Add((Label "Install folder" $Bold))
$dirRow = New-Object Windows.Forms.FlowLayoutPanel
$dirRow.AutoSize = $true; $dirRow.WrapContents = $false
$dirBox = New-Object Windows.Forms.TextBox
$dirBox.Text = $InstallDir; $dirBox.Width = 560
$browse = New-Object Windows.Forms.Button
$browse.Text = "Change..."; $browse.AutoSize = $true
$dirRow.Controls.Add($dirBox); $dirRow.Controls.Add($browse)
$intro.Controls.Add($dirRow)
$intro.Controls.Add((Label "GPUCoadd needs a folder of its own. Change... makes a GPUCoadd folder inside the place you pick." $Font $Grey))
$checkBox = New-Object Windows.Forms.FlowLayoutPanel
$checkBox.FlowDirection = "TopDown"; $checkBox.AutoSize = $true; $checkBox.WrapContents = $false
$intro.Controls.Add((Label "Checks" $Bold))
$intro.Controls.Add($checkBox)
$intro.Controls.Add((Label "GPUCoadd's work folder, $WorkDir, is created by the GPU test (its GPU kernel cache, a few MB) and later holds your heat maps and reports. Uninstalling asks whether to delete it too." $Font $Grey))

$cancelBtn = Button "Cancel"
$installBtn = Button "Install"
$ChecksOk = $false
function Refresh-Checks {
    $checkBox.Controls.Clear()
    $script:ChecksOk = $true
    foreach ($c in (Get-Checks $dirBox.Text)) {
        $mark = "OK"; $color = $Green
        if ($c.ok -eq $false) { $mark = "NO"; $color = $Red; $script:ChecksOk = $false }
        elseif ($null -eq $c.ok) { $mark = "!"; $color = $Amber }
        $checkBox.Controls.Add((Label "$mark   $($c.text)" $Font $color))
    }
    $installBtn.Enabled = $script:ChecksOk -or $DryRun
}
$browse.Add_Click({
    $d = New-Object Windows.Forms.FolderBrowserDialog
    $d.Description = "Choose where to install GPUCoadd (a GPUCoadd folder is created inside)"
    if ($d.ShowDialog() -eq "OK") { $dirBox.Text = Join-Path $d.SelectedPath "GPUCoadd"; Refresh-Checks }
})
$dirBox.Add_Leave({ Refresh-Checks })
$dirBox.Add_KeyDown({ if ($_.KeyCode -eq "Enter") { Refresh-Checks } })

# ---------------------------------------------------------------- page 2: installing
$run = Page
$run.Controls.Add((Label "Installing GPUCoadd $Version" $Title))
$StepNames = @(
    "Copy GPUCoadd's files",
    "Download Python and $($Manifest.packages.Count) packages ($(Size $DownloadBytes))",
    "Install them (about $(Size $InstalledBytes) on disk)",
    "Test the GPU",
    "Finish: add to installed apps, delete the downloads"
)
$stepLabels = @()
foreach ($n in $StepNames) { $l = Label "      $n" $Font $Grey; $stepLabels += $l; $run.Controls.Add($l) }
$run.Controls.Add((Label " "))
$detail = Label "" $Bold
$run.Controls.Add($detail)
$bar = New-Object Windows.Forms.ProgressBar
$bar.Width = 700; $bar.Height = 22; $bar.Maximum = 1000
$run.Controls.Add($bar)
$numbers = Label "" $Font $Grey
$run.Controls.Add($numbers)
$run.Controls.Add((Label "Overall" $Bold))
$overallBar = New-Object Windows.Forms.ProgressBar
$overallBar.Width = 700; $overallBar.Height = 14; $overallBar.Maximum = 1000
$run.Controls.Add($overallBar)
$overallText = Label "" $Font $Grey
$run.Controls.Add($overallText)
$logToggle = New-Object Windows.Forms.LinkLabel
$logToggle.Text = "Show every step (log)"; $logToggle.AutoSize = $true
$run.Controls.Add($logToggle)
$logBox = New-Object Windows.Forms.TextBox
$logBox.Multiline = $true; $logBox.ReadOnly = $true; $logBox.ScrollBars = "Vertical"; $logBox.Width = 700; $logBox.Height = 150
$logBox.Font = New-Object Drawing.Font("Consolas", 9); $logBox.Visible = $false
$run.Controls.Add($logBox)
function Show-LogEnd { $logBox.SelectionStart = $logBox.TextLength; $logBox.SelectionLength = 0; $logBox.ScrollToCaret() }
$logToggle.Add_LinkClicked({
    $logBox.Visible = -not $logBox.Visible
    if ($logBox.Visible) { $logToggle.Text = "Hide the log"; Show-LogEnd } else { $logToggle.Text = "Show every step (log)" }
})

# ---------------------------------------------------------------- page 3: done or stopped
$end = Page
$endTitle = Label "" $Title
$end.Controls.Add($endTitle)
$endBody = New-Object Windows.Forms.FlowLayoutPanel
$endBody.FlowDirection = "TopDown"; $endBody.AutoSize = $true; $endBody.WrapContents = $false
$end.Controls.Add($endBody)

function Copy-Row([string]$text, [string]$value) {
    $row = New-Object Windows.Forms.FlowLayoutPanel
    $row.AutoSize = $true; $row.WrapContents = $false
    $l = Label $text; $l.MaximumSize = New-Object Drawing.Size(560, 0)
    $b = New-Object Windows.Forms.Button
    $b.Text = "Copy path"; $b.AutoSize = $true
    $b.Add_Click({ [Windows.Forms.Clipboard]::SetText($value); $this.Text = "Copied" }.GetNewClosure())
    $row.Controls.Add($l); $row.Controls.Add($b)
    return $row
}

# ---------------------------------------------------------------- running the work
$S = [hashtable]::Synchronized(@{
    Step = 0; States = @("waiting", "waiting", "waiting", "waiting", "waiting"); Notes = @("", "", "", "", "")
    Done = [int64]0; Total = [int64]0; Detail = ""; Log = New-Object Collections.Concurrent.ConcurrentQueue[string]
    Cancel = $false; Result = $null; Error = ""; Version = $Version; StepStart = [DateTime]::Now; OnLine = $null
    SelfTest = ""; SelfTestEvent = $null; InstalledSize = [int64]0; CancelMode = $null; Undoing = $false; LastError = ""; PackageStart = [DateTime]::Now; Rollback = $null; RollbackError = ""
})
$Weights = @(0.01, 0.74, 0.18, 0.06, 0.01)
$Started = $null
$Samples = New-Object Collections.Generic.List[object]
$LogLines = New-Object Collections.Generic.List[string]
$timer = New-Object Windows.Forms.Timer
$timer.Interval = 250

function Rate() {
    $now = [DateTime]::Now
    $Samples.Add(@($now, [double]$S.Done))
    while ($Samples.Count -gt 2 -and ($now - $Samples[0][0]).TotalSeconds -gt 8) { $Samples.RemoveAt(0) }
    $dt = ($now - $Samples[0][0]).TotalSeconds
    if ($dt -lt 1) { return 0 }
    return [math]::Max(0.0, ($S.Done - $Samples[0][1]) / $dt)
}

$timer.Add_Tick({
    $line = $null
    $added = $false
    while ($S.Log.TryDequeue([ref]$line)) { $LogLines.Add($line); $logBox.AppendText($line + "`r`n"); $added = $true }
    if ($added -and $logBox.Visible) { Show-LogEnd }  # the newest line stays in view
    for ($k = 0; $k -lt 5; $k++) {
        $st = $S.States[$k]
        if ($st -eq "done") { $stepLabels[$k].Text = "OK   $($StepNames[$k])"; if ($S.Notes[$k] -and $S.Notes[$k] -ne "done") { $stepLabels[$k].Text += "  ($($S.Notes[$k]))" }; $stepLabels[$k].ForeColor = $Green; $stepLabels[$k].Font = $Font }
        elseif ($st -eq "running") { $stepLabels[$k].Text = ">    $($StepNames[$k])"; $stepLabels[$k].ForeColor = [Drawing.Color]::Black; $stepLabels[$k].Font = $Bold }
        elseif ($st -eq "failed") { $stepLabels[$k].Text = "X    $($StepNames[$k])"; $stepLabels[$k].ForeColor = $Red; $stepLabels[$k].Font = $Bold }
        elseif ($st -eq "stopped") { $stepLabels[$k].Text = "-    $($StepNames[$k])  (stopped)"; $stepLabels[$k].ForeColor = $Amber; $stepLabels[$k].Font = $Bold }
    }
    $detail.Text = $S.Detail
    $k = $S.Step
    $frac = 0.0
    if ($S.Total -gt 0) { $frac = [math]::Min(1.0, $S.Done / $S.Total) }
    $bar.Value = [int](1000 * $frac)
    $elapsed = ([DateTime]::Now - $Started).TotalSeconds
    $left = $null
    if ($k -eq 1 -and $S.Total -gt 0) {
        $r = Rate
        $txt = "$(Size $S.Done) of $(Size $S.Total)"
        if ($r -gt 0) {
            $secs = ($S.Total - $S.Done) / $r
            $txt += "   $("{0:N1}" -f ($r / 1MB)) MB/s   $(Minutes $secs) left for the download"
            $left = $secs + 90 + 30  # measured: about 90 s to install, 20 to 30 s to test
        }
        $numbers.Text = $txt
    } elseif ($k -eq 2 -and $S.Total -gt 0) {
        $numbers.Text = "$([math]::Round(100 * $frac))% installed   this package for $([math]::Round(([DateTime]::Now - $S.PackageStart).TotalSeconds)) s   no download in this step"
        $left = (1 - $frac) * 90 + 30
    } elseif ($k -eq 3) {
        $numbers.Text = "running for $([math]::Round(([DateTime]::Now - $S.StepStart).TotalSeconds)) s"
        $left = 30
    } else { $numbers.Text = "" }
    if ($S.Undoing) { $numbers.Text = "removing only what setup added"; $left = $null }
    $overall = 0.0
    for ($j = 0; $j -lt 5; $j++) { if ($S.States[$j] -eq "done") { $overall += $Weights[$j] } elseif ($j -eq $k) { $overall += $Weights[$j] * $frac } }
    $overallBar.Value = [int](1000 * [math]::Min(1.0, $overall))
    $ot = "$([math]::Round(100 * $overall))%   running for $(Minutes $elapsed)".Replace("under a minute", "less than a minute").Replace("about ", "")
    if ($null -ne $left) { $ot += "   $(Minutes $left) left in total" }
    $overallText.Text = $ot
    if ($S.Result) { $timer.Stop(); Show-End }
})

function Show-End {
    $endBody.Controls.Clear()
    $total = Minutes (([DateTime]::Now - $Started).TotalSeconds)
    $logFile = $null
    if (-not $DryRun) {  # with a finished install; otherwise in %TEMP%, so removing what setup added keeps it
        $logFile = Join-Path $env:TEMP "GPUCoadd-setup-log.txt"
        if ($S.Result -eq "ok") { $logFile = Join-Path $dirBox.Text "setup\setup-log.txt" }
        $LogLines | Set-Content -Encoding UTF8 $logFile
    }
    $buttons.Controls.Clear()
    $close = Button "Close"; $close.Add_Click({ $form.Close() })
    if ($logFile) { $openLog = Button "Open the log"; $openLog.Add_Click({ Start-Process notepad.exe $logFile }.GetNewClosure()) }
    if ($S.Result -eq "ok") {
        $overallBar.Value = 1000
        $endTitle.Text = "GPUCoadd $Version is installed"
        $endTitle.ForeColor = $Green
        $endBody.Controls.Add((Label "Folder: $($dirBox.Text)   ($(Size $S.InstalledSize) on disk, setup took $($total.Replace('about ', '')))."))
        $endBody.Controls.Add((Label "GPU test: $($S.SelfTest)"))
        $endBody.Controls.Add((Label " "))
        $endBody.Controls.Add((Label "Next: open GPUCoadd" $Bold))
        $si = Join-Path $dirBox.Text "GPUCoadd"
        $endBody.Controls.Add((Label "Start menu: GPUCoadd opens the window on its own."))
        $endBody.Controls.Add((Copy-Row "Siril 1.4: Preferences > Scripts, add $si to the script paths (and remove an earlier GPUCoadd\siril path). GPUCoadd is then under Scripts > Python Scripts > GPUCoadd." $si))
        $endBody.Controls.Add((Label " "))
        $endBody.Controls.Add((Label "To remove it later: Windows Settings > Apps > Installed apps > GPUCoadd, or 'Uninstall GPUCoadd' in its folder." $Font $Grey))
        $open = Button "Open the folder"; $open.Add_Click({ Start-Process explorer.exe $dirBox.Text })
        if ($DryRun) { $open.Enabled = $false }
    } elseif ($S.Result -eq "cancelled") {
        $endTitle.ForeColor = $Amber
        $r = $S.Rollback
        if ($S.RollbackError) {
            $endTitle.Text = "Setup was stopped, but not everything could be removed"
            $endBody.Controls.Add((Label $S.RollbackError))
            $endBody.Controls.Add((Label "Close GPUCoadd and Siril, then run 'Uninstall GPUCoadd' in $($dirBox.Text): it removes the rest of what setup added."))
        } elseif ($r) {
            $endTitle.Text = "Setup was stopped and its files were removed"
            $endBody.Controls.Add((Label "Removed: $($r.count) files ($(Size $r.bytes)) that setup had added to $($dirBox.Text)."))
            if ($r.folderGone) { $endBody.Controls.Add((Label "The folder itself is gone too, since nothing else was in it.")) }
            elseif ($r.keptDownloads -gt 0) { $endBody.Controls.Add((Label "The folder stays, with what was in it before and the download you kept.")) }
            else { $endBody.Controls.Add((Label "The folder stays, with what was in it before.")) }
            if ($r.keptDownloads -gt 0) {
                $endBody.Controls.Add((Label "Kept for next time: $(Size $r.keptDownloads) of the $(Size $DownloadBytes) download, in $(Join-Path $dirBox.Text 'downloads'). Running setup again continues from there. To remove it instead, delete that folder and the 'setup' folder next to it."))
            }
            $endBody.Controls.Add((Label "Nothing else on your PC was changed."))
        } else {
            $endTitle.Text = "Setup was stopped"
            $endBody.Controls.Add((Label "Nothing had been installed yet."))
        }
    } else {
        $endTitle.Text = "Setup could not finish"; $endTitle.ForeColor = $Red
        $endBody.Controls.Add((Label "Stopped at: $($StepNames[$S.Step])" $Bold))
        if ($S.Error) { $endBody.Controls.Add((Label $S.Error)) }
        $endBody.Controls.Add((Label " "))
        $kept = [int64]0
        $dl = Join-Path $dirBox.Text "downloads"
        if (Test-Path $dl) { $kept = [int64]((Get-ChildItem -File $dl | Measure-Object Length -Sum).Sum) }
        $script:KeptText = Label "Kept: $(Size $kept) of the $(Size $DownloadBytes) download, in $dl. Running setup again continues from there; nothing is downloaded twice."
        $script:KeptText.Visible = ($kept -gt 0)
        $endBody.Controls.Add($script:KeptText)
        $script:UndoText = Label "Or remove everything setup added so far: only what it added, nothing else."
        $endBody.Controls.Add($script:UndoText)
        if (-not $DryRun -and (Test-Path (Join-Path $dirBox.Text $RecordName))) {
            $undo = Button "Remove what setup added"
            $undo.Add_Click({
                $this.Enabled = $false; $form.Cursor = "WaitCursor"
                $script:UndoText.Text = "Removing..."; [Windows.Forms.Application]::DoEvents()
                try {
                    $r = Remove-Plan (Get-UninstallPlan $dirBox.Text) $false { param($p) [Windows.Forms.Application]::DoEvents() }
                    Remove-Item -Force "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\GPUCoadd" -ErrorAction SilentlyContinue
                    [void](Remove-StartShortcut $dirBox.Text)
                    $script:KeptText.Visible = $false
                    $script:UndoText.ForeColor = $Green
                    $script:UndoText.Text = "Removed: $($r.count) files ($(Size $r.bytes)) that setup had added. Nothing else on your PC was changed."
                } catch {
                    $script:UndoText.ForeColor = $Red
                    $script:UndoText.Text = "Not everything could be removed: $($_.Exception.Message) Close GPUCoadd and Siril and use 'Uninstall GPUCoadd' in the folder."
                }
                $form.Cursor = "Default"
            })
        } else { $script:UndoText.Visible = $false }
    }
    Show-Page $end
    Snap "3-end"
}

function Ask-Stop {
    # Stop: keep going, stop and remove everything setup added, or stop but keep the download for next time
    $d = New-Object Windows.Forms.Form
    $d.Text = "Stop setup?"; $d.Font = $Font; $d.FormBorderStyle = "FixedDialog"; $d.MaximizeBox = $false; $d.MinimizeBox = $false
    $d.StartPosition = "CenterParent"; $d.ClientSize = New-Object Drawing.Size(600, 250); $d.BackColor = [Drawing.Color]::White
    $body = New-Object Windows.Forms.FlowLayoutPanel
    $body.FlowDirection = "TopDown"; $body.WrapContents = $false; $body.Location = New-Object Drawing.Point(20, 16); $body.Size = New-Object Drawing.Size(560, 150)
    $d.Controls.Add($body)
    $l = Label "Stop installing GPUCoadd?" $Bold; $l.MaximumSize = New-Object Drawing.Size(560, 0); $body.Controls.Add($l)
    $l = Label "Setup then removes everything it added to $($dirBox.Text), and nothing else."; $l.MaximumSize = New-Object Drawing.Size(560, 0); $body.Controls.Add($l)
    if ($script:Updating) {
        $l = Label "This was updating an earlier GPUCoadd, so stopping removes GPUCoadd completely. Run setup again to install it." $Font $Amber
        $l.MaximumSize = New-Object Drawing.Size(560, 0); $body.Controls.Add($l)
    }
    # setup's own count: Windows does not update the listed size of a file still being written
    $have = [int64]0
    if ($S.Step -eq 1) { $have = [int64]$S.Done } elseif ($S.Step -gt 1 -and $S.Notes[1] -ne "already installed, kept") { $have = $DownloadBytes }
    $row = New-Object Windows.Forms.FlowLayoutPanel
    $row.FlowDirection = "RightToLeft"; $row.Location = New-Object Drawing.Point(12, 186); $row.Size = New-Object Drawing.Size(580, 50)
    $d.Controls.Add($row)
    $script:StopChoice = $null
    $choices = @(@("Keep installing", ""), @("Stop and remove everything", "all"))
    if ($have -gt 0) { $choices += , @("Stop, keep the $(Size $have) download", "keep") }
    foreach ($c in $choices) {
        $b = New-Object Windows.Forms.Button; $b.Text = $c[0]; $b.Tag = $c[1]; $b.AutoSize = $true; $b.MinimumSize = New-Object Drawing.Size(120, 34)
        $b.Add_Click({ $script:StopChoice = $this.Tag; $this.FindForm().Close() })
        $row.Controls.Add($b)
    }
    if ($have -gt 0) {
        $l = Label "Keeping the download means a later setup run continues from it." $Font $Grey; $l.MaximumSize = New-Object Drawing.Size(560, 0); $body.Controls.Add($l)
    }
    [void]$d.ShowDialog($form)
    if ($script:StopChoice) { return $script:StopChoice }
    return $null
}

$cancelBtn.Add_Click({
    if (-not $Started) { $form.Close(); return }
    $choice = Ask-Stop
    if ($choice) {
        $S.CancelMode = $choice; $S.Cancel = $true
        $cancelBtn.Enabled = $false; $cancelBtn.Text = "Stopping..."
    }
})
$form.Add_FormClosing({
    param($sender, $e)
    if ($Started -and -not $S.Result) {
        $e.Cancel = $true
        $cancelBtn.PerformClick()
    }
})
$installBtn.Add_Click({
    if (-not $dirBox.Text.Trim()) { return }
    $dir = [IO.Path]::GetFullPath($dirBox.Text.Trim())
    $dirBox.Text = $dir
    $installBtn.Visible = $false; $cancelBtn.Text = "Stop"
    $script:Updating = Test-Path (Join-Path $dir "setup\installed.json")  # a finished earlier install, not just a kept download
    $script:Started = [DateTime]::Now
    Show-Page $run
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Add-Type -AssemblyName System.Net.Http
    $handler = New-Object Net.Http.HttpClientHandler
    $S.Http = New-Object Net.Http.HttpClient($handler)
    $S.Http.Timeout = [TimeSpan]::FromMinutes(2)  # covers connecting; reading has its own 30 s stall limit
    $S.Http.DefaultRequestHeaders.UserAgent.ParseAdd("GPUCoadd-setup/$Version")
    $ps = [PowerShell]::Create()
    [void]$ps.AddScript($Work).AddArgument($S).AddArgument($Manifest).AddArgument($dir).AddArgument($Here).AddArgument([bool]$DryRun).AddArgument($LocalFiles).AddArgument($Shared)
    $script:Job = $ps.BeginInvoke()
    $timer.Start()
    if ($Snapshot) {
        $shots = New-Object Windows.Forms.Timer
        $shots.Interval = 1500
        $script:ShotCount = 0
        $shots.Add_Tick({ $script:ShotCount++; Snap "2-running-$($script:ShotCount)"; if ($S.Result -or $script:ShotCount -ge 6) { $this.Stop() } })
        $shots.Start()
    }
})

$form.Add_Shown({
    Refresh-Checks
    Show-Page $intro
    Snap "1-intro"
    if ($Snapshot -and $DryRun) { $installBtn.PerformClick() }
})
if ($Snapshot -and $DryRun) {
    $closer = New-Object Windows.Forms.Timer
    $closer.Interval = 500
    $closer.Add_Tick({ if ($S.Result) { $this.Stop(); Start-Sleep -Milliseconds 800; Snap "3-end"; $form.Close() } })
    $closer.Start()
}
Show-Page $intro
[void]$form.ShowDialog()
