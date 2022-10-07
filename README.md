# MS&amp;E 408: Independent Research on GFNs

## Overview

- **Objective**: Compare performance of GFNs in building committer object to
  standard MCMC-based approaches.
- [References](https://www.zotero.org/groups/4797034/gfns/library)
- [Baseline code](https://github.com/rotskoff-group/learning-committor)

## Key Concepts

- **Variational Monte Carlo**
- **Kolomogorov equation**
- **Langevin equation**: $$d\mathbf{X}_t=-\nabula V(\mathbf{X}_t)dt
  +\sqrt{2\beta^{-1}}d\mathbf{W}_t$$
- **Metastability**
- **Muller-Brown potential**
- **Boltzmann distribution**
- **Umbrella sampling**: uses windowing functions to enhance sampling in otherwise
  rarely sampled regions
  - Windowing function is actively defined using the NN's current estimate
- **Replica exchange**: allows for the exchange between umbrella windows to
  accelerate sampling further
