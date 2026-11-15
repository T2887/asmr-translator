param([int]$x, [int]$y, [int]$samples = 80, [int]$intervalMs = 100)
Add-Type -AssemblyName System.Drawing
Add-Type @'
using System;
using System.Runtime.InteropServices;
public class M3 {
  [DllImport("user32.dll")] public static extern bool SetCursorPos(int x, int y);
  [DllImport("user32.dll")] public static extern void mouse_event(uint f, uint dx, uint dy, uint d, UIntPtr e);
}
'@
function AvgBrightness {
    $bmp = New-Object System.Drawing.Bitmap 1504, 1500
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.CopyFromScreen(3840, 0, 0, 0, (New-Object System.Drawing.Size 1504, 1500))
    $sum = 0.0; $cnt = 0
    for ($py = 0; $py -lt 1500; $py += 60) {
        for ($px = 0; $px -lt 1504; $px += 60) {
            $c = $bmp.GetPixel($px, $py); $sum += ($c.R + $c.G + $c.B) / 3.0; $cnt++
        }
    }
    $g.Dispose(); $bmp.Dispose()
    return [math]::Round($sum / $cnt, 1)
}
Write-Output ("baseline=" + (AvgBrightness))
[M3]::SetCursorPos($x, $y)
Start-Sleep -Milliseconds 150
[M3]::mouse_event(0x0002, 0, 0, 0, [UIntPtr]::Zero)
Start-Sleep -Milliseconds 50
[M3]::mouse_event(0x0004, 0, 0, 0, [UIntPtr]::Zero)
$min = 999.0; $minAt = -1
for ($i = 0; $i -lt $samples; $i++) {
    $v = AvgBrightness
    if ($v -lt $min) { $min = $v; $minAt = $i }
    if ($i % 10 -eq 0) { Write-Output ("t=" + ($i/10) + "s bright=" + $v) }
    Start-Sleep -Milliseconds $intervalMs
}
Write-Output ("MIN=" + $min + " at sample " + $minAt + " (" + ($minAt/10) + "s)")
