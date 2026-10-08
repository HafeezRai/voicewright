{{/* Common name/label helpers. Kept to the conventional Helm set so the chart
     behaves like any other and `helm lint`'s naming checks are meaningful. */}}

{{- define "voicewright.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "voicewright.fullname" -}}
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

{{- define "voicewright.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/* Labels shared by every resource. */}}
{{- define "voicewright.labels" -}}
helm.sh/chart: {{ include "voicewright.chart" . }}
{{ include "voicewright.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: {{ .Values.global.partOf }}
{{- end }}

{{/* Selector labels -- must be stable across upgrades, so only name+instance. */}}
{{- define "voicewright.selectorLabels" -}}
app.kubernetes.io/name: {{ include "voicewright.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Per-component labels. Usage: {{ include "voicewright.componentLabels" (dict "ctx" $ "component" "backend-core") }}
*/}}
{{- define "voicewright.componentLabels" -}}
{{ include "voicewright.labels" .ctx }}
app.kubernetes.io/component: {{ .component }}
{{- end }}

{{- define "voicewright.imagePullSecrets" -}}
{{- with .Values.image.pullSecrets }}
imagePullSecrets:
{{- toYaml . | nindent 2 }}
{{- end }}
{{- end }}

{{/*
The Secret name holding runtime credentials: the operator's existing Secret, or
the one this chart creates when secrets.create is true.
*/}}
{{- define "voicewright.secretName" -}}
{{- if .Values.secrets.create }}{{ include "voicewright.fullname" . }}-secrets
{{- else }}{{ .Values.secrets.existingSecret }}{{- end }}
{{- end }}
