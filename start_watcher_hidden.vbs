Set WshShell = CreateObject("WScript.Shell")
' Get the directory where this script is located
ScriptDir = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
' Run the batch file hidden (0 = hidden window, False = don't wait for it to finish)
WshShell.Run Chr(34) & ScriptDir & "\start_auto_watcher.bat" & Chr(34), 0, False
