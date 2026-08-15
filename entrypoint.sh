#!/usr/bin/env bash
set -e

ENV_NAME=robodiff
ENV_YAML=/workspace/diffusion_policy/conda_environment.yaml

if [ ! -d "/opt/conda/envs/${ENV_NAME}" ]; then
  if [ -f "$ENV_YAML" ]; then
    echo "[entrypoint] '${ENV_NAME}' 환경이 없어서 처음 생성합니다. (${ENV_YAML} 기준, 시간 좀 걸려요)"
    mamba env create -f "$ENV_YAML"
  else
    echo "[entrypoint] 경고: ${ENV_YAML} 을 찾을 수 없습니다."
    echo "[entrypoint] 호스트에서 diffusion_policy 저장소를 clone 했는지, docker-compose.yml의 볼륨 마운트 경로가 맞는지 확인하세요."
  fi
else
  echo "[entrypoint] '${ENV_NAME}' 환경이 이미 존재합니다. 재생성을 건너뜁니다."
fi

source /opt/conda/etc/profile.d/conda.sh
conda activate ${ENV_NAME} 2>/dev/null || echo "[entrypoint] 아직 ${ENV_NAME} 환경을 활성화할 수 없습니다 (생성 실패 여부 확인 필요)."

exec "$@"
