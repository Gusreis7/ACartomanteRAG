resource "aws_glue_job" "etl" {
  name              = "${var.project}-etl"
  role_arn          = aws_iam_role.etl.arn
  glue_version      = "5.1"
  worker_type       = "G.1X"
  number_of_workers = 2
  max_retries       = 0
  timeout           = 600
  execution_property { max_concurrent_runs = 1 }
  command {
    name            = "glueetl"
    python_version  = "3"
    script_location = "s3://${aws_s3_bucket.data["ops"].id}/${var.glue_bundle_prefix}/job.py"
  }
  default_arguments = {
    "--job-language"                     = "python"
    "--enable-metrics"                   = "true"
    "--enable-continuous-cloudwatch-log" = "true"
    "--job-bookmark-option"              = "job-bookmark-disable"
    "--TempDir"                          = "s3://${aws_s3_bucket.data["ops"].id}/glue-temp/"
    "--additional-python-modules"        = "s3://${aws_s3_bucket.data["ops"].id}/${var.glue_bundle_prefix}/dependencies.gluewheels.zip"
    "--python-modules-installer-option"  = "--no-index"
    "--assets-bucket"                    = aws_s3_bucket.data["ops"].id
    "--bundle-prefix"                    = var.glue_bundle_prefix
    "--settings-parameter"               = local.settings_etl
  }
}
