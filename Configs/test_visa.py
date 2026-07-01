import struct
import os

print("=== DIAGNOZA VISA ===")
print("Arhitectura Python:", struct.calcsize("P") * 8, "biti")

locatii_posibile = [
    r"C:\Windows\System32\visa64.dll",
    r"C:\Windows\System32\visa32.dll",
    r"C:\Windows\SysWOW64\visa32.dll",
    r"C:\Program Files\IVI Foundation\VISA\Win64\Bin\visa64.dll",
    r"C:\Program Files\IVI Foundation\VISA\Win64\Bin\visa32.dll",
    r"C:\Program Files\IVI Foundation\VISA\Win64\agvisa\agbin\visa64.dll",
    r"C:\Program Files\IVI Foundation\VISA\Win64\agvisa\agbin\visa32.dll",
    r"C:\Program Files (x86)\IVI Foundation\VISA\WinNT\Bin\visa32.dll",
    r"C:\Program Files (x86)\IVI Foundation\VISA\WinNT\agvisa\agbin\visa32.dll"
]

print("\nCaut DLL-uri VISA in sistem...")
gasite = 0
for cale in locatii_posibile:
    if os.path.exists(cale):
        print("GASIT ->", cale)
        gasite += 1

if gasite == 0:
    print("EROARE: Nu am gasit niciun fisier DLL in locatiile standard!")
print("=====================")
