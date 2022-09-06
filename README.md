# python 3.6 setup

```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt install python3.6 python3.6-dev
```

# setup pipenv

```
export PIPENV_VENV_IN_PROJECT=true
python3.6 -m pipenv --python 3.6
python3.6 -m pipenv install tensorflow==1.12.0
python3.6 -m pipenv shell
```

# docker

## build

```
docker build -t gan-tf-gpu .
```
## gpu
```
docker run --gpus all --rm --oom-kill-disable=true \
-e USE_GPU=1 -e TF_FORCE_GPU_ALLOW_GROWTH=true -e TZ=Asia/Tokyo \
-e LOCAL_UID=$(id -u) -e LOCAL_UID=$(id -g) -e LOCAL_UNAME=$(whoami) \
-v $(pwd)/src:/app/src -v $(pwd)/output:/app/output \
-it gan-tf-gpu python -m pipenv run python /app/src/ch4.py
```

## cpu
```
docker run --rm --oom-kill-disable=true \
-e TZ=Asia/Tokyo \
-e LOCAL_UID=$(id -u) -e LOCAL_UID=$(id -g) -e LOCAL_UNAME=$(whoami) \
-v $(pwd)/src:/app/src -v $(pwd)/output:/app/output \
-it gan-tf-gpu python -m pipenv run python /app/src/ch4.py
```