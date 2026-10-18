# Provisionamento e execução de dados têm identidades próprias no GitHub.
resource "aws_iam_role" "github_infra" {
  name               = "${var.project}-github-infra"
  assume_role_policy = aws_iam_role.github_deploy.assume_role_policy
}
# Alterações nestas políticas são aplicadas pelo perfil administrativo local.
resource "aws_iam_role_policy" "github_infra_iam" {
  role   = aws_iam_role.github_infra.id
  policy = jsonencode(local.github_infra_read_policy)
}
resource "aws_iam_role_policy" "github_infra_operations" {
  name   = "${var.project}-infra-operations"
  role   = aws_iam_role.github_infra.id
  policy = jsonencode(local.github_infra_operations_policy)
  lifecycle {
    precondition {
      condition     = length(jsonencode(local.github_infra_read_policy)) + length(jsonencode(local.github_infra_operations_policy)) <= 10240
      error_message = "As políticas inline da infraestrutura excedem o limite de 10.240 caracteres."
    }
  }
}
resource "aws_iam_role" "github_etl" {
  name                 = "${var.project}-github-etl"
  max_session_duration = 10800
  assume_role_policy   = aws_iam_role.github_deploy.assume_role_policy
}
resource "aws_iam_role_policy" "github_etl" {
  role = aws_iam_role.github_etl.id
  policy = jsonencode({ Version = "2012-10-17", Statement = [
    { Effect = "Allow", Action = ["s3:GetObject"], Resource = ["${aws_s3_bucket.data["ops"].arn}/config/deployment.json", "${aws_s3_bucket.data["ops"].arn}/config/emergency.json"] },
    { Effect = "Allow", Action = ["s3:ListBucket"], Resource = [aws_s3_bucket.data["ops"].arn], Condition = { StringEquals = { "s3:prefix" = "config/emergency.json" } } },
    { Effect = "Allow", Action = ["s3:ListBucket"], Resource = [aws_s3_bucket.data["ops"].arn], Condition = { StringLike = { "s3:prefix" = ["glue/*"] } } },
    { Effect = "Allow", Action = ["s3:GetObject", "s3:PutObject"], Resource = ["${aws_s3_bucket.data["ops"].arn}/glue/*"] },
    { Effect = "Allow", Action = ["glue:GetJob", "glue:StartJobRun", "glue:GetJobRun"], Resource = [aws_glue_job.etl.arn] },
    { Effect = "Allow", Action = ["ssm:SendCommand"], Resource = [
      aws_ssm_document.reload_data.arn, "arn:aws:ec2:${var.region}:${data.aws_caller_identity.current.account_id}:instance/${aws_instance.app.id}"
    ] },
    { Effect = "Allow", Action = ["ssm:GetCommandInvocation"], Resource = ["*"] }
  ] })
}
resource "aws_ssm_document" "reload_data" {
  name          = "${var.project}-reload-data"
  document_type = "Command"
  content = jsonencode({
    schemaVersion = "2.2", description = "Recarrega os dados publicados, sem receber comandos ou argumentos",
    mainSteps = [{ action = "aws:runShellScript", name = "reload", inputs = {
      timeoutSeconds = "1800",
      runCommand     = ["if [ -L /srv/acartomante/current ]; then python3 /opt/acartomante/manage.py restart; else echo 'Dados publicados; primeiro deploy da aplicação ainda pendente.'; fi"]
    } }]
  })
}
# Identificadores atualizados pelo apply; os workflows não dependem de IDs EC2 antigos.
locals {
  deployment_context = {
    region                   = var.region
    buckets                  = local.bucket_env
    ops_bucket               = aws_s3_bucket.data["ops"].id
    instance_id              = aws_instance.app.id
    ecr_repository           = aws_ecr_repository.api.repository_url
    ssm_deploy_document      = aws_ssm_document.deploy.name
    ssm_data_reload_document = aws_ssm_document.reload_data.name
    app_settings_parameter   = local.settings_app
    etl_settings_parameter   = local.settings_etl
    glue_job                 = aws_glue_job.etl.name
  }
}
resource "aws_s3_object" "deployment_context" {
  bucket       = aws_s3_bucket.data["ops"].id
  key          = "config/deployment.json"
  content      = jsonencode({ for key, value in local.deployment_context : key => { value = value } })
  content_type = "application/json"
}
