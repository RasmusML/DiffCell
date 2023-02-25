# Notes

## Papers
* [Lafarge](http://proceedings.mlr.press/v102/lafarge19a/lafarge19a.pdf)
* [Diffusion models article](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
* [Diffusion models paper](https://arxiv.org/pdf/2006.11239.pdf)
* [Latent diffusion models](https://arxiv.org/pdf/2112.10752.pdf)
* [Variational Sparse Coding](https://openreview.net/pdf?id=SkeJ6iR9Km)

## Direction
Train a diffusion model conditioned on treatment (treatment := (compound, concentration))
- Maybe log-transform concentration 

Use a classifier to predict treatment
- regression problem on concentration level
- classification problem on compound
- Consider adding VAE
