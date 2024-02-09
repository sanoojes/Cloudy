# Installing Cloudy on Windows

Please choose one of the methods that you prefer to use:

- [Install Python](https://www.python.org/downloads/)
- [Install Cmake](https://cmake.org/download/)
- Install FFplay (part of FFmpeg)

## **Installing FFplay from FFmpeg**

### **1.Using chocolatey:**

```bash
# installing Chocolatey via Powershell(optional)
Set-ExecutionPolicy Bypass -Scope Process -Force; iwr https://community.chocolatey.org/install.ps1 -UseBasicParsing | iex

# Install FFmpeg-Full (which includes FFplay) using Chocolatey
choco install ffmpeg-full 
```
### **2.Using scoop:**

```bash
# Instaling Scoop via Powershell(optional)
Set-ExecutionPolicy Bypass -Scope Process -Force; Invoke-RestMethod -Uri https://get.scoop.sh | Invoke-Expression

# Install FFmpeg-Full (which includes FFplay) using scoop
scoop install ffmpeg
```

### **3.Using winget**

```bash
# Installing Winget from Prowershell  

# Get latest winget download url
$URL = "https://api.github.com/repos/microsoft/winget-cli/releases/latest"
$URL = (Invoke-WebRequest -Uri $URL).Content | ConvertFrom-Json |
        Select-Object -ExpandProperty "assets" |
        Where-Object "browser_download_url" -Match '.msixbundle' |
        Select-Object -ExpandProperty "browser_download_url"

# Download winget
Invoke-WebRequest -Uri $URL -OutFile "Setup.msix" -UseBasicParsing

# install and remove residue
Add-AppxPackage -Path "Setup.msix"
Remove-Item "Setup.msix"

# Install FFmpeg-Full (which includes FFplay) using winget
winget install ffmpeg
```

## **Installing Dlib for Windows**

```bash
git clone https://github.com/Sachu-Settan/dlib
cd dlib

# Replace the filenames according to the available files in the directory
pip install ./dlib-<version>.whl
```

## **Installing Other Packages For the Project via pip**

### **From [requirements.txt](requirements.txt)**

```
pip install -r ./requirements.txt
```

### **Manual Installation**

```
pip install schedule openai speechrecognition python-dotenv aiohttp opencv-python asyncio pygame gtts edge_tts cmake face_recognition pyaudio
```
<br>

# Check Configs
[Check](./main.py#280)

# Contact For Help
**`If there is any issue with installation and running`**

**Contact me**: [**`Whatsapp`**](https://wa.me/+919744933034) or **`Discord: sachu_settan`**
