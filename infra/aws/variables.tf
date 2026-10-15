variable "region" {
  type    = string
  default = "us-east-1"
}
variable "project" {
  type    = string
  default = "acartomanterag"
  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{2,25}$", var.project))
    error_message = "Use um nome curto com letras minúsculas, números e hífen."
  }
}
variable "domain" {
  type    = string
  default = "acartomanterag.gustavodreis.dev.br"
}
variable "acme_email" { type = string }
variable "github_repository_id" {
  type = string
  validation {
    condition     = can(regex("^[0-9]+$", var.github_repository_id))
    error_message = "Informe o ID numérico retornado pelo script configure_github.py."
  }
}
variable "oidc_provider_arn" {
  type        = string
  default     = null
  description = "Se a conta já tem o provider GitHub, informe seu ARN para reutilizá-lo."
}
variable "hosted_zone_id" {
  type        = string
  default     = null
  description = "Zona Route 53 existente; vazio cria uma zona somente para o subdomínio."
}
variable "instance_type" {
  type    = string
  default = "m7i-flex.large"
}
variable "ami_id" {
  type        = string
  default     = null
  description = "Fixe o AMI retornado no primeiro plan para repetir a imagem do SO."
}
variable "availability_zone" {
  type    = string
  default = "us-east-1a"
}
variable "data_volume_gb" {
  type    = number
  default = 50
}
variable "monthly_budget_usd" {
  type    = number
  default = 75
}
variable "budget_email" { type = string }
variable "glue_bundle_prefix" {
  type        = string
  default     = "glue/bootstrap"
  description = "Fallback do job. O workflow ETL seleciona script, fontes e wheels da mesma revisão em cada execução."
}
