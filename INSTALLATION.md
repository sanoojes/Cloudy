# **Installing Cloudy on Windows**

**`Please choose one of the methods that you prefer to use:`**

## **Installing FFplay from FFmpeg**

### **1.Using chocolatey:**

```bash
# installing Chocolatey via Powershell(optional)
Set-ExecutionPolicy Bypass -Scope Process -Force; iwr https://community.chocolatey.org/install.ps1 -UseBasicParsing | iex

# Installing FFmpeg-Full using Chocolatey
choco install ffmpeg-full 
```
#### **2.Using scoop:**

```bash
# Instaling Scoop via Powershell(optional)
Set-ExecutionPolicy Bypass -Scope Process -Force; Invoke-RestMethod -Uri https://get.scoop.sh | Invoke-Expression

# Installing FFmpeg-Full using scoop
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

# Installing FFmpeg-Full using winget
winget install ffmpeg
```

## **Installing Dlib for Windows**

```bash
git clone https://github.com/Sachu-Settan/dlib
cd dlib

For py 3.7 => pip install .\dlib-19.19.0-cp37-cp37m-win_amd64.whl 
For py 3.8 => pip install .\dlib-19.19.0-cp38-cp38-win_amd64.whl   
For py 3.9 => pip install .\dlib-19.22.1-cp39-cp39-win_amd64.whl   
For py 3.10 => pip install .\dlib-19.22.99-cp310-cp310-win_amd64.whl
For py 3.11 => pip install .\dlib-19.24.1-cp311-cp311-win_amd64.whl 
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

# Contact For Help
**`If there is any issue with installation and running`**

**Contact me**: [**`Whatsapp`**](https://wa.me/+919744933034) or **`Discord: sachu_settan`**