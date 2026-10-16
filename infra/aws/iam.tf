resource "aws_iam_role" "app" {
  name = "${var.project}-runtime"
  assume_role_policy = jsonencode({ Version = "2012-10-17", Statement = [{
    Effect = "Allow", Principal = { Service = "ec2.amazonaws.com" }, Action = "sts:AssumeRole"
  }] })
}
resource "aws_iam_instance_profile" "app" {
  name = var.project
  role = aws_iam_role.app.name
}
resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.app.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}
resource "aws_iam_role_policy" "app" {
  role = aws_iam_role.app.id
  policy = jsonencode({ Version = "2012-10-17", Statement = [
    { Effect = "Allow", Action = ["s3:ListBucket"], Resource = [for name in local.read_buckets : aws_s3_bucket.data[name].arn] },
    { Effect = "Allow", Action = ["s3:GetObject"], Resource = concat(
      [for name in local.read_buckets : "${aws_s3_bucket.data[name].arn}/*"],
      ["${aws_s3_bucket.data["ops"].arn}/releases/*", "${aws_s3_bucket.data["ops"].arn}/backups/*"]
    ) },
    { Effect = "Allow", Action = ["s3:PutObject"], Resource = ["${aws_s3_bucket.data["ops"].arn}/backups/*"] },
    { Effect = "Allow", Action = ["ssm:GetParameter"], Resource = ["arn:aws:ssm:${var.region}:${data.aws_caller_identity.current.account_id}:parameter${local.settings_app}"] },
    { Effect = "Allow", Action = ["bedrock:InvokeModel"], Resource = [
      "arn:aws:bedrock:${var.region}::foundation-model/${local.qwen}",
      "arn:aws:bedrock:${var.region}:${data.aws_caller_identity.current.account_id}:inference-profile/us.${local.scout}",
      "arn:aws:bedrock:us-*::foundation-model/${local.scout}"
    ] },
    { Effect = "Allow", Action = ["bedrock:GetFoundationModel"], Resource = [
      "arn:aws:bedrock:${var.region}::foundation-model/${local.qwen}"
    ] },
    { Effect = "Allow", Action = ["bedrock-mantle:CreateInference"], Resource = ["arn:aws:bedrock-mantle:${var.region}:${data.aws_caller_identity.current.account_id}:project/*"], Condition = { StringEquals = { "bedrock-mantle:Model" = "qwen.qwen3-32b" } } },
    { Effect = "Allow", Action = ["bedrock-mantle:GetModel"], Resource = ["arn:aws:bedrock-mantle:${var.region}:${data.aws_caller_identity.current.account_id}:project/*"] },
    { Effect = "Allow", Action = ["bedrock:GetInferenceProfile"], Resource = [
      "arn:aws:bedrock:${var.region}:${data.aws_caller_identity.current.account_id}:inference-profile/us.${local.scout}"
    ] },
    { Effect = "Allow", Action = ["ecr:GetAuthorizationToken"], Resource = ["*"] },
    { Effect = "Allow", Action = ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer", "ecr:BatchCheckLayerAvailability"], Resource = [aws_ecr_repository.api.arn] },
    { Effect = "Allow", Action = ["logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams"], Resource = ["${aws_cloudwatch_log_group.operations.arn}:*"] }
  ] })
}
resource "aws_iam_openid_connect_provider" "github" {
  count          = var.oidc_provider_arn == null ? 1 : 0
  url            = "https://token.actions.githubusercontent.com"
  client_id_list = ["sts.amazonaws.com"]
}
locals {
  github_oidc_arn = var.oidc_provider_arn == null ? aws_iam_openid_connect_provider.github[0].arn : var.oidc_provider_arn
}
resource "aws_iam_role" "github_deploy" {
  name = "${var.project}-github-deploy"
  assume_role_policy = jsonencode({ Version = "2012-10-17", Statement = [{
    Effect = "Allow", Principal = { Federated = local.github_oidc_arn }, Action = "sts:AssumeRoleWithWebIdentity",
    Condition = { StringEquals = {
      "token.actions.githubusercontent.com:aud" = "sts.amazonaws.com",
      "token.actions.githubusercontent.com:sub" = "repository_id:${var.github_repository_id}:ref:refs/heads/main"
    } }
  }] })
}
resource "aws_ssm_document" "deploy" {
  name          = "${var.project}-deploy"
  document_type = "Command"
  content = jsonencode({
    schemaVersion = "2.2", description = "Deploy fixado por revisão e digest, sem comandos arbitrários",
    parameters = {
      ReleaseKey = { type = "String", interpolationType = "ENV_VAR", allowedPattern = "^releases/[0-9a-f]{40}/bundle[.]tar[.]gz$" },
      Image      = { type = "String", interpolationType = "ENV_VAR", allowedPattern = "^${replace(aws_ecr_repository.api.repository_url, ".", "[.]")}@sha256:[0-9a-f]{64}$" }
    },
    mainSteps = [{ action = "aws:runShellScript", name = "deploy", inputs = {
      timeoutSeconds = "1800",
      runCommand     = ["cloud-init status --wait >/dev/null && python3 /opt/acartomante/manage.py deploy --release-key \"$SSM_ReleaseKey\" --image \"$SSM_Image\""]
    } }]
  })
}
resource "aws_iam_role_policy" "github_deploy" {
  role = aws_iam_role.github_deploy.id
  policy = jsonencode({ Version = "2012-10-17", Statement = [
    { Effect = "Allow", Action = ["ecr:GetAuthorizationToken"], Resource = ["*"] },
    { Effect = "Allow", Action = ["ecr:DescribeImages", "ecr:BatchGetImage", "ecr:BatchCheckLayerAvailability", "ecr:PutImage", "ecr:InitiateLayerUpload", "ecr:UploadLayerPart", "ecr:CompleteLayerUpload"], Resource = [aws_ecr_repository.api.arn] },
    { Effect = "Allow", Action = ["s3:PutObject", "s3:GetObject"], Resource = [
      "${aws_s3_bucket.data["ops"].arn}/releases/*"
    ] },
    { Effect = "Allow", Action = ["s3:GetObject"], Resource = ["${aws_s3_bucket.data["ops"].arn}/config/deployment.json"] },
    { Effect = "Allow", Action = ["ssm:SendCommand"], Resource = [
      aws_ssm_document.deploy.arn, "arn:aws:ec2:${var.region}:${data.aws_caller_identity.current.account_id}:instance/${aws_instance.app.id}"
    ] },
    { Effect = "Allow", Action = ["ssm:GetCommandInvocation"], Resource = ["*"] }
  ] })
}
resource "aws_iam_role" "etl" {
  name = "${var.project}-etl"
  assume_role_policy = jsonencode({ Version = "2012-10-17", Statement = [{
    Effect = "Allow", Principal = { Service = "glue.amazonaws.com" }, Action = "sts:AssumeRole"
  }] })
}
resource "aws_iam_role_policy" "etl" {
  role = aws_iam_role.etl.id
  policy = jsonencode({ Version = "2012-10-17", Statement = [
    { Effect = "Allow", Action = ["s3:ListBucket", "s3:GetBucketLocation"], Resource = [for b in aws_s3_bucket.data : b.arn] },
    { Effect = "Allow", Action = ["s3:GetObject", "s3:PutObject"], Resource = [for name in local.buckets : "${aws_s3_bucket.data[name].arn}/*" if name != "ops"] },
    { Effect = "Allow", Action = ["s3:GetObject"], Resource = ["${aws_s3_bucket.data["ops"].arn}/glue/*", "${aws_s3_bucket.data["ops"].arn}/config/*"] },
    { Effect = "Allow", Action = ["s3:GetObject", "s3:PutObject"], Resource = ["${aws_s3_bucket.data["ops"].arn}/glue-temp/*"] },
    { Effect = "Allow", Action = ["s3:DeleteObject"], Resource = ["${aws_s3_bucket.data["indexes"].arn}/production/etl.lock"] },
    { Effect = "Allow", Action = ["ssm:GetParameter"], Resource = ["arn:aws:ssm:${var.region}:${data.aws_caller_identity.current.account_id}:parameter${local.settings_etl}"] },
    { Effect = "Allow", Action = ["bedrock:InvokeModel", "bedrock:GetFoundationModel"], Resource = ["arn:aws:bedrock:${var.region}::foundation-model/${local.qwen}"] },
    { Effect = "Allow", Action = ["bedrock-mantle:CreateInference"], Resource = ["arn:aws:bedrock-mantle:${var.region}:${data.aws_caller_identity.current.account_id}:project/*"], Condition = { StringEquals = { "bedrock-mantle:Model" = "qwen.qwen3-32b" } } },
    { Effect = "Allow", Action = ["bedrock-mantle:GetModel"], Resource = ["arn:aws:bedrock-mantle:${var.region}:${data.aws_caller_identity.current.account_id}:project/*"] },
    { Effect = "Allow", Action = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"], Resource = ["arn:aws:logs:${var.region}:${data.aws_caller_identity.current.account_id}:log-group:/aws-glue/*"] }
  ] })
}
