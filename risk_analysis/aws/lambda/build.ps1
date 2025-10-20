Write-Host "🔨 Building Lambda package..." -ForegroundColor Yellow

# Criar diretório temporário
$tempDir = "temp_package"
if (Test-Path $tempDir) {
    Remove-Item -Recurse -Force $tempDir
}
New-Item -ItemType Directory -Path $tempDir | Out-Null

# Copiar arquivos Python
Copy-Item "*.py" $tempDir -ErrorAction SilentlyContinue

# Copiar requirements e instalar dependências
if (Test-Path "requirements.txt") {
    Copy-Item "requirements.txt" $tempDir
    Set-Location $tempDir
    pip install -r requirements.txt -t .
    Set-Location ..
}

# Criar zip
if (Test-Path "risk-pipeline.zip") {
    Remove-Item "risk-pipeline.zip"
}

Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::CreateFromDirectory($tempDir, "risk-pipeline.zip")

# Limpar
Remove-Item -Recurse -Force $tempDir

Write-Host "✅ Package built: risk-pipeline.zip" -ForegroundColor Green
