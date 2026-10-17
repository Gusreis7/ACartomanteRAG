# A identidade de operações de dados também executa o MCQ manual.
# Escrita limitada aos resultados; corpus, índices e releases são somente leitura.
resource "aws_iam_role_policy" "github_mcq" {
  name   = "${var.project}-mcq"
  role   = aws_iam_role.github_etl.id
  policy = jsonencode({ Version = "2012-10-17", Statement = [
    { Effect = "Allow", Action = ["s3:ListBucket"], Resource = [for name in local.read_buckets : aws_s3_bucket.data[name].arn] },
    { Effect = "Allow", Action = ["s3:GetObject"], Resource = [for name in local.read_buckets : "${aws_s3_bucket.data[name].arn}/*"] },
    { Effect = "Allow", Action = ["s3:ListBucket"], Resource = [aws_s3_bucket.data["benchmarks"].arn], Condition = { StringLike = { "s3:prefix" = "mcq/current/*" } } },
    { Effect = "Allow", Action = ["s3:GetObject", "s3:PutObject"], Resource = ["${aws_s3_bucket.data["benchmarks"].arn}/mcq/current/*"] },
    { Effect = "Allow", Action = ["ssm:GetParameter"], Resource = ["arn:aws:ssm:${var.region}:${data.aws_caller_identity.current.account_id}:parameter${local.settings_app}"] },
    { Effect = "Allow", Action = ["bedrock-mantle:CreateInference"], Resource = ["arn:aws:bedrock-mantle:${var.region}:${data.aws_caller_identity.current.account_id}:project/*"], Condition = { StringEquals = { "bedrock-mantle:Model" = "qwen.qwen3-32b" } } },
    { Effect = "Allow", Action = ["bedrock-mantle:GetModel"], Resource = ["arn:aws:bedrock-mantle:${var.region}:${data.aws_caller_identity.current.account_id}:project/*"] },
    { Effect = "Allow", Action = ["bedrock:InvokeModel"], Resource = [
      "arn:aws:bedrock:${var.region}::foundation-model/${local.qwen}",
      "arn:aws:bedrock:${var.region}:${data.aws_caller_identity.current.account_id}:inference-profile/us.${local.scout}",
      "arn:aws:bedrock:us-*::foundation-model/${local.scout}"
    ] },
    { Effect = "Allow", Action = ["bedrock:GetFoundationModel"], Resource = ["arn:aws:bedrock:${var.region}::foundation-model/${local.qwen}"] },
    { Effect = "Allow", Action = ["bedrock:GetInferenceProfile"], Resource = ["arn:aws:bedrock:${var.region}:${data.aws_caller_identity.current.account_id}:inference-profile/us.${local.scout}"] }
  ] })
}
