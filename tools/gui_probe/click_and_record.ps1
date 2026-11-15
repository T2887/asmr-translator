param([int]$x, [int]$y, [int]$frames = 20, [int]$intervalMs = 120, [string]$prefix = "f")
Add-Type -AssemblyName System.Drawing
Add-Type @'
using System;
using System.Runtime.InteropServices;
public class M2 {
  [DllImport("user32.dll")] public static extern bool SetCursorPos(int x, int y);
  [DllImport("user32.dll")] public static extern void mouse_event(uint f, uint dx, uint dy, uint d, UIntPtr e);
}
'@
[M2]::SetCursorPos($x, $y)
Start-Sleep -Milliseconds 100
[M2]::mouse_event(0x0002, 0, 0, 0, [UIntPtr]::Zero)
Start-Sleep -Milliseconds 50
[M2]::mouse_event(0x0004, 0, 0, 0, [UIntPtr]::Zero)
$dir = "H:\software\ASMR翻译器\_shots"
for ($i = 0; $i -lt $frames; $i++) {
    $bmp = New-Object System.Drawing.Bitmap 1504, 1500
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.CopyFromScreen(3840, 0, 0, 0, (New-Object System.Drawing.Size 1504, 1500))
    # 快速算平均亮度（每 40 像素采样一次，够判断黑屏）
    $sum = 0.0; $cnt = 0
    for ($py = 0; $py -lt 1500; $py += 40) {
        for ($px = 0; $px -lt 1504; $px += 40) {
            $c = $bmp.GetPixel($px, $py)
            $sum += ($c.R + $c.G + $c.B) / 3.0
            $cnt++
        }
    }
    $avg = [math]::Round($sum / $cnt, 1)
    $file = "$dir\{0}_{1:D2}.png" -f $prefix, $i
    $bmp.Save($file, [System.Drawing.Imaging.ImageFormat]::Png)
    $g.Dispose(); $bmp.Dispose()
    Write-Output ("frame {0} avg_brightness={1}" -f $i, $avg)
    Start-Sleep -Milliseconds $intervalMs
}
