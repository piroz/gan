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