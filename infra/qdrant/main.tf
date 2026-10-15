terraform {
  required_version = ">= 1.11.0, < 2.0.0"
  backend "s3" {}
  required_providers {
    qdrant-cloud = {
      source  = "qdrant/qdrant-cloud"
      version = "= 1.28.0"
    }
  }
}
variable "account_id" { type = string }
variable "cloud_api_key" {
  type      = string
  sensitive = true
  ephemeral = true
}
provider "qdrant-cloud" {
  account_id = var.account_id
  api_key    = var.cloud_api_key
}
# cluster.generated.tf é gerado do cluster real pelo script de importação.
# Não escolho um pacote pago ou uma configuração fictícia para o free tier.
