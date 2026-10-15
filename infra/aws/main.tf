data "aws_caller_identity" "current" {}
data "aws_ssm_parameter" "ami" {
  name = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
}
locals {
  prefix       = "${var.project}-${data.aws_caller_identity.current.account_id}-${var.region}"
  buckets      = toset(["raw", "bronze", "silver", "indexes", "enrichment", "benchmarks", "ops"])
  read_buckets = ["bronze", "silver", "indexes", "enrichment"]
  settings_app = "/${var.project}/production/app"
  settings_etl = "/${var.project}/production/etl"
  private_ip   = "10.42.1.10"
  qwen         = "qwen.qwen3-32b-v1:0"
  scout        = "meta.llama4-scout-17b-instruct-v1:0"
  bucket_env = {
    RAW_BUCKET               = aws_s3_bucket.data["raw"].id
    BRONZE_BUCKET            = aws_s3_bucket.data["bronze"].id
    SILVER_BUCKET            = aws_s3_bucket.data["silver"].id
    EMBEDDING_INDEXES_BUCKET = aws_s3_bucket.data["indexes"].id
    ENRICHMENT_BUCKET        = aws_s3_bucket.data["enrichment"].id
    BENCHMARKS_BUCKET        = aws_s3_bucket.data["benchmarks"].id
  }
}
resource "aws_s3_bucket" "data" {
  for_each      = local.buckets
  bucket        = "${local.prefix}-${each.key}"
  force_destroy = false
  lifecycle { prevent_destroy = true }
}
resource "aws_s3_bucket_public_access_block" "data" {
  for_each                = local.buckets
  bucket                  = aws_s3_bucket.data[each.key].id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
resource "aws_s3_bucket_versioning" "data" {
  for_each = local.buckets
  bucket   = aws_s3_bucket.data[each.key].id
  versioning_configuration { status = "Enabled" }
}
resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  for_each = local.buckets
  bucket   = aws_s3_bucket.data[each.key].id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "AES256" }
  }
}
resource "aws_s3_bucket_policy" "tls" {
  for_each = local.buckets
  bucket   = aws_s3_bucket.data[each.key].id
  policy = jsonencode({
    Version = "2012-10-17", Statement = [{
      Sid       = "DenyInsecureTransport", Effect = "Deny", Principal = "*", Action = "s3:*",
      Resource  = [aws_s3_bucket.data[each.key].arn, "${aws_s3_bucket.data[each.key].arn}/*"],
      Condition = { Bool = { "aws:SecureTransport" = "false" } }
    }]
  })
}
resource "aws_s3_bucket_lifecycle_configuration" "ops" {
  bucket = aws_s3_bucket.data["ops"].id
  rule {
    id     = "backups-30-days"
    status = "Enabled"
    filter { prefix = "backups/" }
    expiration { days = 30 }
    noncurrent_version_expiration { noncurrent_days = 30 }
  }
}
resource "aws_s3_object" "buckets" {
  bucket       = aws_s3_bucket.data["ops"].id
  key          = "config/buckets.json"
  content      = jsonencode(local.bucket_env)
  content_type = "application/json"
}
resource "aws_ecr_repository" "api" {
  name                 = var.project
  image_tag_mutability = "IMMUTABLE"
  image_scanning_configuration { scan_on_push = true }
  lifecycle { prevent_destroy = true }
}
resource "aws_ecr_lifecycle_policy" "api" {
  repository = aws_ecr_repository.api.name
  policy = jsonencode({ rules = [{
    rulePriority = 1, description = "Últimas dez imagens", selection = {
      tagStatus = "any", countType = "imageCountMoreThan", countNumber = 10
    }, action   = { type = "expire" }
  }] })
}
resource "aws_vpc" "app" {
  cidr_block           = "10.42.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
}
resource "aws_internet_gateway" "app" { vpc_id = aws_vpc.app.id }
resource "aws_subnet" "app" {
  vpc_id            = aws_vpc.app.id
  cidr_block        = "10.42.1.0/24"
  availability_zone = var.availability_zone
}
resource "aws_route_table" "app" {
  vpc_id = aws_vpc.app.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.app.id
  }
}
resource "aws_route_table_association" "app" {
  subnet_id      = aws_subnet.app.id
  route_table_id = aws_route_table.app.id
}
resource "aws_security_group" "app" {
  name   = "${var.project}-https"
  vpc_id = aws_vpc.app.id
  ingress {
    description = "Único serviço público"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  dynamic "egress" {
    for_each = [443, 6333]
    content {
      from_port   = egress.value
      to_port     = egress.value
      protocol    = "tcp"
      cidr_blocks = ["0.0.0.0/0"]
    }
  }
  dynamic "egress" {
    for_each = ["tcp", "udp"]
    content {
      from_port   = 53
      to_port     = 53
      protocol    = egress.value
      cidr_blocks = ["10.42.0.2/32"]
    }
  }
}
resource "aws_ebs_volume" "data" {
  availability_zone = var.availability_zone
  size              = var.data_volume_gb
  type              = "gp3"
  encrypted         = true
  tags              = { Name = "${var.project}-data" }
  lifecycle { prevent_destroy = true }
}
resource "aws_instance" "app" {
  ami                         = coalesce(var.ami_id, nonsensitive(data.aws_ssm_parameter.ami.value))
  instance_type               = var.instance_type
  subnet_id                   = aws_subnet.app.id
  private_ip                  = local.private_ip
  associate_public_ip_address = true
  vpc_security_group_ids      = [aws_security_group.app.id]
  iam_instance_profile        = aws_iam_instance_profile.app.name
  metadata_options {
    http_tokens                 = "required"
    http_put_response_hop_limit = 2
  }
  root_block_device {
    volume_type = "gp3"
    volume_size = 40
    encrypted   = true
  }
  user_data_base64 = base64gzip(templatefile("${path.module}/cloud-init.sh.tftpl", {
    volume_id = replace(aws_ebs_volume.data.id, "-", "")
    manager   = file("${path.module}/manage.py")
    host_config = jsonencode(merge(local.bucket_env, {
      AWS_REGION       = var.region, APP_DOMAIN = var.domain, ACME_EMAIL = var.acme_email,
      EC2_BIND_IP      = local.private_ip, SETTINGS_PARAMETER = local.settings_app,
      OPS_BUCKET       = aws_s3_bucket.data["ops"].id, ECR_REPOSITORY = aws_ecr_repository.api.repository_url,
      DATA_RELEASE_KEY = "production/current.json"
    }))
  }))
  user_data_replace_on_change = true
  tags                        = { Name = var.project }
  depends_on                  = [aws_route_table_association.app, aws_iam_role_policy.app]
}
resource "aws_volume_attachment" "data" {
  device_name                    = "/dev/sdf"
  volume_id                      = aws_ebs_volume.data.id
  instance_id                    = aws_instance.app.id
  stop_instance_before_detaching = true
}
resource "aws_eip" "app" { domain = "vpc" }
resource "aws_eip_association" "app" {
  allocation_id = aws_eip.app.id
  instance_id   = aws_instance.app.id
}
resource "aws_route53_zone" "app" {
  count = var.hosted_zone_id == null ? 1 : 0
  name  = var.domain
  lifecycle { prevent_destroy = true }
}
resource "aws_route53_record" "app" {
  zone_id = var.hosted_zone_id == null ? aws_route53_zone.app[0].zone_id : var.hosted_zone_id
  name    = var.domain
  type    = "A"
  ttl     = 300
  records = [aws_eip.app.public_ip]
}
resource "aws_cloudwatch_log_group" "operations" {
  name              = "/${var.project}/operations"
  retention_in_days = 14
}
resource "aws_budgets_budget" "monthly" {
  name         = "${var.project}-monthly"
  budget_type  = "COST"
  limit_amount = tostring(var.monthly_budget_usd)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.budget_email]
  }
}
