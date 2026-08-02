Option Explicit

Dim shell, fileSystem, projectRoot, scriptPath, command
Set shell = CreateObject("WScript.Shell")
Set fileSystem = CreateObject("Scripting.FileSystemObject")

projectRoot = fileSystem.GetParentFolderName(WScript.ScriptFullName)
scriptPath = projectRoot & "\run_desktop.ps1"
command = "powershell.exe -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File " _
    & Chr(34) & scriptPath & Chr(34)

shell.Run command, 0, False
