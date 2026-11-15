param([string]$out = "shot.png")
Add-Type -AssemblyName System.Drawing
$bmp = New-Object System.Drawing.Bitmap 1504, 1500
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.CopyFromScreen(3840, 0, 0, 0, (New-Object System.Drawing.Size 1504, 1500))
$bmp.Save("H:\software\ASMR翻译器\_shots\$out", [System.Drawing.Imaging.ImageFormat]::Png)
$g.Dispose(); $bmp.Dispose()
Write-Output "saved $out"
