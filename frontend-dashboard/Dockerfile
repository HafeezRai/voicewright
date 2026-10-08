# syntax=docker/dockerfile:1
# Voicewright Voice Agent — frontend-dashboard (Phase 1 skeleton).
# Multi-stage: build the Vite bundle, serve the static output with nginx.

# --- stage 1: build ---------------------------------------------------------
FROM node:22-alpine AS build

WORKDIR /app

# Dependency layer: cached until the lockfile/manifest change.
COPY package.json package-lock.json* ./
RUN npm ci || npm install

# Build layer.
COPY tsconfig.json tsconfig.node.json vite.config.ts tailwind.config.js postcss.config.js index.html ./
COPY src ./src

ARG VITE_BACKEND_CORE_URL=http://localhost:8000
ARG VITE_AI_ENGINE_URL=http://localhost:8001
ENV VITE_BACKEND_CORE_URL=${VITE_BACKEND_CORE_URL}
ENV VITE_AI_ENGINE_URL=${VITE_AI_ENGINE_URL}
RUN npm run build

# --- stage 2: serve ---------------------------------------------------------
FROM nginx:1.27-alpine AS runtime

# Served on 5173 to match the compose port mapping and the Vite dev server.
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 5173

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget -qO- http://127.0.0.1:5173/ >/dev/null 2>&1 || exit 1

CMD ["nginx", "-g", "daemon off;"]
