# improved_qtransformer

### Installation
```bash
conda create -n qtransformer python==3.9
conda activate qtransformer
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia

git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .

pip install datasets
pip install evaluate
pip install accelerate -U
pip install apache_beam
pip install -U scikit-learn
pip install wandb
pip install deepspeed
```