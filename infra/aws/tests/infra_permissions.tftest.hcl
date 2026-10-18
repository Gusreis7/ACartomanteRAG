# Somente plano: os valores simulados ficam disponíveis nas asserções.
mock_provider "aws" {
  override_during = plan

  mock_data "aws_caller_identity" {
    defaults = { account_id = "123456789012" }
  }
  mock_data "aws_ssm_parameter" {
    defaults = { value = "ami-0123456789abcdef0" }
  }
  mock_resource "aws_vpc" {
    defaults = { arn = "arn:aws:ec2:us-east-1:123456789012:vpc/vpc-0123456789abcdef0" }
  }
  mock_resource "aws_instance" {
    defaults = { arn = "arn:aws:ec2:us-east-1:123456789012:instance/i-0123456789abcdef0" }
  }
  mock_resource "aws_ebs_volume" {
    defaults = { arn = "arn:aws:ec2:us-east-1:123456789012:volume/vol-0123456789abcdef0" }
  }
  mock_resource "aws_eip" {
    defaults = { public_ip = "192.0.2.10" }
  }
  mock_resource "aws_route53_zone" {
    defaults = { zone_id = "Z0123456789" }
  }
  mock_resource "aws_s3_bucket" {
    defaults = { arn = "arn:aws:s3:::acartomanterag-test" }
  }
  mock_resource "aws_ecr_repository" {
    defaults = {
      arn            = "arn:aws:ecr:us-east-1:123456789012:repository/acartomanterag"
      repository_url = "123456789012.dkr.ecr.us-east-1.amazonaws.com/acartomanterag"
    }
  }
  mock_resource "aws_glue_job" {
    defaults = { arn = "arn:aws:glue:us-east-1:123456789012:job/acartomanterag-etl" }
  }
  mock_resource "aws_ssm_document" {
    defaults = { arn = "arn:aws:ssm:us-east-1:123456789012:document/acartomanterag-deploy" }
  }
  # Em plan, todos os tipos usados pelas políticas precisam de mock_resource.
  mock_resource "aws_iam_instance_profile" {
    defaults = { arn = "arn:aws:iam::123456789012:instance-profile/acartomanterag" }
  }
  mock_resource "aws_iam_openid_connect_provider" {
    defaults = { arn = "arn:aws:iam::123456789012:oidc-provider/token.actions.githubusercontent.com" }
  }
}

variables {
  acme_email           = "ops@example.test"
  budget_email         = "ops@example.test"
  github_repository_id = "12345678"
  domain               = "acartomanterag.example.test"
}

run "project_permissions" {
  command = plan

  override_resource {
    override_during = plan
    target          = aws_iam_role.app
    values          = { arn = "arn:aws:iam::123456789012:role/acartomanterag-runtime" }
  }
  override_resource {
    override_during = plan
    target          = aws_iam_role.etl
    values          = { arn = "arn:aws:iam::123456789012:role/acartomanterag-etl" }
  }
  override_resource {
    override_during = plan
    target          = aws_iam_role.github_deploy
    values          = { arn = "arn:aws:iam::123456789012:role/acartomanterag-github-deploy" }
  }
  override_resource {
    override_during = plan
    target          = aws_iam_role.github_infra
    values          = { arn = "arn:aws:iam::123456789012:role/acartomanterag-github-infra" }
  }
  override_resource {
    override_during = plan
    target          = aws_iam_role.github_etl
    values          = { arn = "arn:aws:iam::123456789012:role/acartomanterag-github-etl" }
  }

  assert {
    condition = alltrue([
      for statement in concat(local.github_infra_read_policy.Statement, local.github_infra_operations_policy.Statement) :
      alltrue([for action in statement.Action : !strcontains(action, "*")]) if statement.Effect == "Allow"
    ])
    error_message = "Allow não deve ter ações wildcard, iam:* ou políticas administrativas."
  }
  assert {
    condition = alltrue([
      for statement in concat(local.github_infra_read_policy.Statement, local.github_infra_operations_policy.Statement) :
      alltrue([for action in statement.Action : startswith(action, "ec2:Describe") || action == "logs:DescribeLogGroups"])
      if statement.Effect == "Allow" && contains(statement.Resource, "*")
    ])
    error_message = "Resource=* só é aceito nas consultas regionais sem suporte a ARN."
  }
  assert {
    condition = alltrue([
      for statement in concat(local.github_infra_read_policy.Statement, local.github_infra_operations_policy.Statement) :
      alltrue([for action in statement.Action : !startswith(action, "iam:") || startswith(action, "iam:Get") || startswith(action, "iam:List") || action == "iam:PassRole"])
      if statement.Effect == "Allow"
    ])
    error_message = "A CI não pode criar ou modificar identidades IAM."
  }
  assert {
    condition = alltrue([
      for statement in local.github_infra_operations_policy.Statement :
      statement.Resource == [aws_iam_role.etl.arn] && statement.Condition.StringEquals["iam:PassedToService"] == "glue.amazonaws.com"
      if contains(statement.Action, "iam:PassRole")
    ])
    error_message = "PassRole deve ficar restrito à role ETL e ao serviço Glue."
  }
  assert {
    condition = alltrue([
      for statement in local.github_infra_operations_policy.Statement :
      statement.Resource == ["${local.infra_state_object_arn}.tflock"]
      if statement.Effect == "Allow" && contains(statement.Action, "s3:DeleteObject")
    ])
    error_message = "A CI pode apagar o lock, nunca o state, corpus ou backups."
  }
  assert {
    condition = alltrue([
      for statement in local.github_infra_operations_policy.Statement :
      statement.Condition["ForAllValues:StringEquals"]["route53:ChangeResourceRecordSetsNormalizedRecordNames"] == [var.domain] &&
      statement.Condition["ForAllValues:StringEquals"]["route53:ChangeResourceRecordSetsRecordTypes"] == ["A"]
      if contains(statement.Action, "route53:ChangeResourceRecordSets")
    ])
    error_message = "Alterações DNS devem ficar no A da aplicação."
  }
  assert {
    condition = contains(
      [for statement in local.github_infra_operations_policy.Statement : statement.Effect if statement.Sid == "ProtectIamConfiguration"],
      "Deny"
    )
    error_message = "A negação explícita das alterações IAM deve permanecer."
  }
}
