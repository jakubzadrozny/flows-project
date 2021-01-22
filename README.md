# A Comparison Project for Generative Models
This projects compares the generative power and representation quality of some of the most popular generative models to date.

## Models
This repository contains 4 model implementations:

* VAE, based on [https://github.com/AntixK/PyTorch-VAE](https://github.com/AntixK/PyTorch-VAE)
* GAN, based on [this PyTorch tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
* RealNVP, based on [https://github.com/fmu2/realNVP](https://github.com/fmu2/realNVP)
* Glow, based on [https://github.com/y0ast/Glow-PyTorch](https://github.com/y0ast/Glow-PyTorch)

## Training
Training is done with the help of [PyTorch Lightning](https://www.pytorchlightning.ai).

First install requirements by `pip install -r requirements.txt`.

You can start training by invoking the runner script:
```
python runner.py {vae, gan, realnvp, glow} [--FLAGS]
```
Use `python runner.py -h` for help with the flags.

## Experiments
Experiments are all placed in the `experiments.ipynb` notebook.
They require PyTorch Lightning checkpoints with model weights to run. You can generate your own checkpoints by training a model (see the **Training** section) or download mine from:

* VAE (40MB download): [https://drive.google.com/file/d/1i1QHCoGB1HH8IbGkrCuOXeDBDWEigSqs/view?usp=sharing] (https://drive.google.com/file/d/1i1QHCoGB1HH8IbGkrCuOXeDBDWEigSqs/view?usp=sharing)
* GAN (70MB download): [https://drive.google.com/file/d/1CI1YJOYvn2cSTdFUd3I9HfleeMXQnTLM/view?usp=sharing] (https://drive.google.com/file/d/1CI1YJOYvn2cSTdFUd3I9HfleeMXQnTLM/view?usp=sharing)
* RealNVP (1GB download): (coming soon)

You need to uncompress the files after download. You also need to provide paths to appropriate checkpoints in the notebook header.
