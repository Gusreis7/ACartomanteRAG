output "buckets" { value = local.bucket_env }
output "ops_bucket" { value = aws_s3_bucket.data["ops"].id }
output "instance_id" { value = aws_instance.app.id }
output "ami_id" { value = aws_instance.app.ami }
output "public_ip" { value = aws_eip.app.public_ip }
output "nameservers" { value = var.hosted_zone_id == null ? aws_route53_zone.app[0].name_servers : [] }
output "url" { value = "https://${var.domain}" }
output "data_volume_id" { value = aws_ebs_volume.data.id }
output "github_role_arn" { value = aws_iam_role.github_deploy.arn }
output "ecr_repository" { value = aws_ecr_repository.api.repository_url }
output "ssm_deploy_document" { value = aws_ssm_document.deploy.name }
output "app_settings_parameter" { value = local.settings_app }
output "etl_settings_parameter" { value = local.settings_etl }
output "glue_job" { value = aws_glue_job.etl.name }
output "github_oidc_subject" { value = "repository_id:${var.github_repository_id}:ref:refs/heads/main" }
output "region" { value = var.region }
