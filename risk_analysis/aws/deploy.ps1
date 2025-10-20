param(
    [string]$Region = "sa-east-1",
    [string]$FunctionName = "ProjetoA_Risco_Function"
)

Write-Host "🚀 Iniciando deploy para a função '$FunctionName' na região $Region" -ForegroundColor Green

# 1. Construir pacote Lambda
Write-Host "📦 Construindo pacote Lambda..." -ForegroundColor Yellow
Set-Location -Path "lambda"
.\build.ps1
Set-Location -Path ..

# 2. Implantar CloudFormation
Write-Host "☁️ Implantando stack CloudFormation..." -ForegroundColor Yellow
aws cloudformation deploy `
    --template-file cloudformation/risk-pipeline-stack.yml `
    --stack-name "risk-analytics-stack" `
    --capabilities CAPABILITY_IAM `
    --region $Region

# 3. Fazer upload do código Lambda
Write-Host "📤 Fazendo upload do código Lambda..." -ForegroundColor Yellow
aws lambda update-function-code `
    --function-name $FunctionName `
    --zip-file fileb://lambda/risk-pipeline.zip `
    --region $Region
