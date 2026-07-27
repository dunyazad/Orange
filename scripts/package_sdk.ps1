# ---------------------------------------------------------------------------
# package_sdk.ps1 -- build Orange and stage it as a redistributable binary SDK.
#
#   ./scripts/package_sdk.ps1                          # Debug + Release -> dist/
#   ./scripts/package_sdk.ps1 -Configs Release         # Release only
#   ./scripts/package_sdk.ps1 -Shared -Configs Release # orange_core.dll build
#   ./scripts/package_sdk.ps1 -StaticCrt                # orange_c.dll: KERNEL32 only
#   ./scripts/package_sdk.ps1 -Out D:/SDK -Zip         # custom dir + .zip archives
#
# One prefix per configuration (dist/Orange-SDK-Release, -Debug): the plugin
# DLLs and SDL3.dll have config-independent file names, so mixing both into one
# prefix would have the second install overwrite the first's runtime.
# The static core itself is config-tagged (orange_cored.lib), so a single-prefix
# layout also works if you only care about the .lib -- see -Single.
# ---------------------------------------------------------------------------
[CmdletBinding()]
param(
    [string[]]$Configs   = @('Debug', 'Release'),
    [string]  $Generator = 'Visual Studio 17 2022',
    [string]  $Arch      = 'x64',
    [string]  $Out       = 'dist',
    [string]  $BuildDir  = 'build-sdk',
    [switch]  $Single,          # all configs into one prefix (dist/Orange-SDK)
    [switch]  $Shared,          # orange_core as a DLL instead of a static .lib
    [switch]  $StaticCrt,       # /MT: orange_c.dll then imports only KERNEL32
    [switch]  $Zip              # also produce <prefix>.zip
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    $build = Join-Path $root $BuildDir
    $outDir = if ([System.IO.Path]::IsPathRooted($Out)) { $Out } else { Join-Path $root $Out }

    Write-Host "== configure ($Generator $Arch) ==" -ForegroundColor Cyan
    $sharedFlag = if ($Shared)    { 'ON' } else { 'OFF' }
    $crtFlag    = if ($StaticCrt) { 'ON' } else { 'OFF' }
    cmake -S $root -B $build -G $Generator -A $Arch `
          -DORANGE_INSTALL=ON -DORANGE_BUILD_TESTS=OFF `
          -DORANGE_BUILD_SHARED=$sharedFlag -DORANGE_STATIC_CRT=$crtFlag
    if ($LASTEXITCODE -ne 0) { throw "configure failed ($LASTEXITCODE)" }

    foreach ($cfg in $Configs) {
        $prefix = if ($Single) { Join-Path $outDir 'Orange-SDK' }
                  else         { Join-Path $outDir "Orange-SDK-$cfg" }

        Write-Host "== build $cfg ==" -ForegroundColor Cyan
        cmake --build $build --config $cfg
        if ($LASTEXITCODE -ne 0) { throw "build $cfg failed ($LASTEXITCODE)" }

        Write-Host "== install $cfg -> $prefix ==" -ForegroundColor Cyan
        cmake --install $build --config $cfg --prefix $prefix
        if ($LASTEXITCODE -ne 0) { throw "install $cfg failed ($LASTEXITCODE)" }

        if ($Zip) {
            $zip = "$prefix.zip"
            if (Test-Path $zip) { Remove-Item $zip }
            Compress-Archive -Path "$prefix/*" -DestinationPath $zip
            Write-Host "   packed $zip"
        }
    }

    Write-Host ""
    Write-Host "SDK ready. Consume it with:" -ForegroundColor Green
    Write-Host "  cmake -S <yourproject> -B <build> -DCMAKE_PREFIX_PATH=$outDir/Orange-SDK-Release"
    Write-Host "  find_package(Orange CONFIG REQUIRED); target_link_libraries(app PRIVATE orange::core)"
}
finally {
    Pop-Location
}
