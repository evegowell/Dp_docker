# Dp_docker

diffusion_policy와 같은 폴더안에 도커파일들 있도록 설정

```
mkdir dp-docker
cd ~/dp-docker

git clone https://github.com/real-stanford/diffusion_policy.git

docker compose build

docker compose run --rm diffusion_policy
```
