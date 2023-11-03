FROM pytorch/pytorch:latest

apt install -q nnn neovim dstat htop  \
netbase  # To enable /etc/protocols which is required by git-annex.

conda env create -n mytransformers -c nvidia -c pytorch -c conda-forge  \
  python==3.10 \
  conda-forge::datasets \
  conda-forge::tokenizers>=0.13.3 \
  conda-forge::transformers \
  einops \
  httpcore \
  ipykernel \
  ipywidgets \
  jaxtyping \
  lightning \
  numpy \
  plotly \
  protobuf \
  pybids \
  pynvml \
  pynvml \
  pyparsing \
  pytorch-cuda >=11.7\
  pytorch::pytorch \
  pytorch::torchaudio \
  pytorch::torchvision \
  scikit-learn \
  sentencepiece \
  setuptools \
  simple-parsing \
  sniffio \
  tqdm \
  wandb \
  wheel
