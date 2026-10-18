# A CI atualiza recursos existentes. IAM, rede e substituições usam o perfil local.
locals {
  infra_account          = data.aws_caller_identity.current.account_id
  infra_state_bucket_arn = "arn:aws:s3:::${var.project}-tfstate-${local.infra_account}-${var.region}"
  infra_state_object_arn = "${local.infra_state_bucket_arn}/${var.project}/production.tfstate"
  infra_zone_id          = var.hosted_zone_id == null ? aws_route53_zone.app[0].zone_id : var.hosted_zone_id
  infra_zone_arn         = "arn:aws:route53:::hostedzone/${local.infra_zone_id}"
  infra_role_arns = [
    aws_iam_role.app.arn, aws_iam_role.etl.arn, aws_iam_role.github_deploy.arn,
    aws_iam_role.github_infra.arn, aws_iam_role.github_etl.arn
  ]
  infra_document_arns  = [aws_ssm_document.deploy.arn, aws_ssm_document.reload_data.arn]
  infra_config_objects = [for key in ["buckets.json", "deployment.json"] : "${aws_s3_bucket.data["ops"].arn}/config/${key}"]
  infra_volume_arns = [
    aws_ebs_volume.data.arn,
    "arn:aws:ec2:${var.region}:${local.infra_account}:volume/${aws_instance.app.root_block_device[0].volume_id}"
  ]
  infra_logs_arn = "arn:aws:logs:${var.region}:${local.infra_account}:log-group:/${var.project}/operations"

  github_infra_read_policy = {
    Version = "2012-10-17"
    Statement = [
      {
        Sid = "ReadEc2RegionalMetadata", Effect = "Allow",
        Action = [
          "ec2:DescribeAddresses", "ec2:DescribeAddressesAttribute",
          "ec2:DescribeIamInstanceProfileAssociations",
          "ec2:DescribeInstanceCreditSpecifications", "ec2:DescribeInstances",
          "ec2:DescribeInstanceTypes", "ec2:DescribeInternetGateways",
          "ec2:DescribeNetworkAcls",
          "ec2:DescribeNetworkInterfaces", "ec2:DescribeRouteTables",
          "ec2:DescribeSecurityGroups", "ec2:DescribeSubnets", "ec2:DescribeTags",
          "ec2:DescribeVolumes", "ec2:DescribeVolumesModifications",
          "ec2:DescribeVpcs"
        ],
        Resource = ["*"], Condition = { StringEquals = { "aws:RequestedRegion" = var.region } }
      },
      {
        Sid    = "ReadVpcAttributes", Effect = "Allow",
        Action = ["ec2:DescribeVpcAttribute"], Resource = [aws_vpc.app.arn]
      },
      {
        Sid    = "ReadInstanceAttributes", Effect = "Allow",
        Action = ["ec2:DescribeInstanceAttribute"], Resource = [aws_instance.app.arn]
      },
      {
        Sid = "ReadProjectBuckets", Effect = "Allow",
        Action = [
          "s3:ListBucket", "s3:GetBucketLocation", "s3:GetBucketAcl",
          "s3:GetBucketCORS", "s3:GetBucketWebsite", "s3:GetBucketVersioning",
          "s3:GetAccelerateConfiguration", "s3:GetBucketRequestPayment",
          "s3:GetBucketLogging", "s3:GetBucketNotification", "s3:GetLifecycleConfiguration",
          "s3:GetReplicationConfiguration", "s3:GetEncryptionConfiguration",
          "s3:GetBucketObjectLockConfiguration", "s3:GetBucketPolicy",
          "s3:GetBucketPublicAccessBlock", "s3:GetBucketTagging"
        ],
        Resource = [for bucket in aws_s3_bucket.data : bucket.arn]
      },
      {
        Sid      = "ReadProjectRoles", Effect = "Allow",
        Action   = ["iam:GetRole", "iam:GetRolePolicy", "iam:ListRolePolicies", "iam:ListAttachedRolePolicies", "iam:ListRoleTags"],
        Resource = local.infra_role_arns
      },
      {
        Sid    = "ReadInstanceProfile", Effect = "Allow",
        Action = ["iam:GetInstanceProfile"], Resource = [aws_iam_instance_profile.app.arn]
      },
      {
        Sid    = "ReadGithubOidc", Effect = "Allow",
        Action = ["iam:GetOpenIDConnectProvider"], Resource = [local.github_oidc_arn]
      },
      {
        Sid    = "ReadGlueJob", Effect = "Allow",
        Action = ["glue:GetJob", "glue:GetJobRuns", "glue:GetTags"], Resource = [aws_glue_job.etl.arn]
      },
      {
        Sid      = "ReadSsmDocuments", Effect = "Allow",
        Action   = ["ssm:DescribeDocument", "ssm:GetDocument", "ssm:DescribeDocumentPermission", "ssm:ListTagsForResource"],
        Resource = local.infra_document_arns
      },
      {
        Sid      = "ReadPinnedAmiParameter", Effect = "Allow",
        Action   = ["ssm:GetParameter"],
        Resource = ["arn:aws:ssm:${var.region}::parameter/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"]
      },
      {
        Sid      = "ReadEcrConfiguration", Effect = "Allow",
        Action   = ["ecr:DescribeRepositories", "ecr:GetLifecyclePolicy", "ecr:ListTagsForResource"],
        Resource = [aws_ecr_repository.api.arn]
      },
      {
        Sid      = "ReadDnsZone", Effect = "Allow",
        Action   = ["route53:GetHostedZone", "route53:ListResourceRecordSets", "route53:ListTagsForResource"],
        Resource = [local.infra_zone_arn]
      },
      {
        Sid    = "ReadDnsChangeStatus", Effect = "Allow",
        Action = ["route53:GetChange"], Resource = ["arn:aws:route53:::change/*"]
      },
      {
        Sid       = "ReadLogGroupMetadata", Effect = "Allow",
        Action    = ["logs:DescribeLogGroups"], Resource = ["*"],
        Condition = { StringEquals = { "aws:RequestedRegion" = var.region } }
      },
      {
        Sid    = "ReadLogGroupTags", Effect = "Allow",
        Action = ["logs:ListTagsForResource"], Resource = [local.infra_logs_arn]
      },
      {
        Sid      = "ReadBudget", Effect = "Allow",
        Action   = ["budgets:ViewBudget", "budgets:ListTagsForResource"],
        Resource = ["arn:aws:budgets::${local.infra_account}:budget/${var.project}-monthly"]
      }
    ]
  }

  github_infra_operations_policy = {
    Version = "2012-10-17"
    Statement = [
      {
        Sid = "ProtectIamConfiguration", Effect = "Deny",
        Action = [
          "iam:Add*", "iam:Attach*", "iam:Create*", "iam:Delete*", "iam:Detach*",
          "iam:Put*", "iam:Remove*", "iam:Set*", "iam:Tag*", "iam:Untag*", "iam:Update*"
        ], Resource = ["*"]
      },
      {
        Sid    = "UseTerraformBackend", Effect = "Allow",
        Action = ["s3:ListBucket", "s3:GetBucketLocation"], Resource = [local.infra_state_bucket_arn]
      },
      {
        Sid      = "WriteStateAndLock", Effect = "Allow",
        Action   = ["s3:GetObject", "s3:PutObject"],
        Resource = [local.infra_state_object_arn, "${local.infra_state_object_arn}.tflock"]
      },
      {
        Sid    = "ReleaseStateLock", Effect = "Allow",
        Action = ["s3:DeleteObject"], Resource = ["${local.infra_state_object_arn}.tflock"]
      },
      {
        Sid      = "MaintainDeploymentContext", Effect = "Allow",
        Action   = ["s3:GetObject", "s3:PutObject", "s3:GetObjectTagging", "s3:PutObjectTagging", "s3:DeleteObjectTagging"],
        Resource = local.infra_config_objects
      },
      {
        Sid      = "EmergencyFlag", Effect = "Allow",
        Action   = ["s3:GetObject", "s3:PutObject"],
        Resource = ["${aws_s3_bucket.data["ops"].arn}/config/emergency.json"]
      },
      {
        Sid    = "StopOrResumeInstance", Effect = "Allow",
        Action = ["ec2:StartInstances", "ec2:StopInstances"], Resource = [aws_instance.app.arn]
      },
      {
        Sid       = "TuneInstanceType", Effect = "Allow",
        Action    = ["ec2:ModifyInstanceAttribute"], Resource = [aws_instance.app.arn],
        Condition = { StringEquals = { "ec2:Attribute" = "instanceType" } }
      },
      {
        Sid    = "TuneProjectVolumes", Effect = "Allow",
        Action = ["ec2:ModifyVolume"], Resource = local.infra_volume_arns
      },
      {
        Sid    = "TagComputeResources", Effect = "Allow",
        Action = ["ec2:CreateTags", "ec2:DeleteTags"], Resource = concat([aws_instance.app.arn], local.infra_volume_arns)
      },
      {
        Sid      = "UpdateOrStopGlueJob", Effect = "Allow",
        Action   = ["glue:UpdateJob", "glue:BatchStopJobRun", "glue:TagResource", "glue:UntagResource"],
        Resource = [aws_glue_job.etl.arn]
      },
      {
        Sid       = "PassOnlyGlueExecutionRole", Effect = "Allow",
        Action    = ["iam:PassRole"], Resource = [aws_iam_role.etl.arn],
        Condition = { StringEquals = { "iam:PassedToService" = "glue.amazonaws.com" } }
      },
      {
        Sid      = "UpdateExistingSsmDocuments", Effect = "Allow",
        Action   = ["ssm:UpdateDocument", "ssm:UpdateDocumentDefaultVersion", "ssm:AddTagsToResource", "ssm:RemoveTagsFromResource"],
        Resource = local.infra_document_arns
      },
      {
        Sid      = "UpdateRepositoryConfiguration", Effect = "Allow",
        Action   = ["ecr:PutImageScanningConfiguration", "ecr:PutImageTagMutability", "ecr:PutLifecyclePolicy", "ecr:TagResource", "ecr:UntagResource"],
        Resource = [aws_ecr_repository.api.arn]
      },
      {
        Sid    = "MaintainApplicationARecord", Effect = "Allow",
        Action = ["route53:ChangeResourceRecordSets"], Resource = [local.infra_zone_arn],
        Condition = { "ForAllValues:StringEquals" = {
          "route53:ChangeResourceRecordSetsNormalizedRecordNames" = [trimsuffix(lower(var.domain), ".")],
          "route53:ChangeResourceRecordSetsRecordTypes"           = ["A"],
          "route53:ChangeResourceRecordSetsActions"               = ["CREATE", "UPSERT", "DELETE"]
        } }
      },
      {
        Sid      = "MaintainOperationsLogGroup", Effect = "Allow",
        Action   = ["logs:PutRetentionPolicy", "logs:DeleteRetentionPolicy", "logs:TagResource", "logs:UntagResource"],
        Resource = [local.infra_logs_arn, "${local.infra_logs_arn}:*"]
      }
    ]
  }
}
