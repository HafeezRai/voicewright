{{/* Common name/label helpers. Kept to the conventional Helm set so the chart
     behaves like any other and `helm lint`'s naming checks are meaningful. */}}

{{- define "cloudpso.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "cloudpso.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{- define "cloudpso.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/* Labels shared by every resource. */}}
{{- define "cloudpso.labels" -}}
helm.sh/chart: {{ include "cloudpso.chart" . }}
{{ include "cloudpso.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: {{ .Values.global.partOf }}
{{- end }}

{{/* Selector labels -- must be stable across upgrades, so only name+instance. */}}
{{- define "cloudpso.selectorLabels" -}}
app.kubernetes.io/name: {{ include "cloudpso.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Per-component labels. Usage: {{ include "cloudpso.componentLabels" (dict "ctx" $ "component" "backend-core") }}
*/}}
{{- define "cloudpso.componentLabels" -}}
{{ include "cloudpso.labels" .ctx }}
app.kubernetes.io/component: {{ .component }}
{{- end }}

{{- define "cloudpso.imagePullSecrets" -}}
{{- with .Values.image.pullSecrets }}
imagePullSecrets:
{{- toYaml . | nindent 2 }}
{{- end }}
{{- end }}

{{/*
The Secret name holding runtime credentials: the operator's existing Secret, or
the one this chart creates when secrets.create is true.
*/}}
{{- define "cloudpso.secretName" -}}
{{- if .Values.secrets.create }}{{ include "cloudpso.fullname" . }}-secrets
{{- else }}{{ .Values.secrets.existingSecret }}{{- end }}
{{- end }}
