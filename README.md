# Multi-Mixture Model (MMM) Architecture
## Table of Contents
- [Overview](#overview)
- [Introduction](#introduction)
- [Methodology](#methodology)
  - [Encoder and Decoder](#encoder-and-decoder)
  - [RecurrentNetwork](#recurrentnetwork)
  ...
- [Use Examples](#use-examples)

## Overview

The Multi-Mixture Model (MMM) architecture is a unified framework for general-purpose data modeling that integrates multiple probabilistic and neural components. It comprises distinct modules – an Encoder, a Decoder, a RecurrentNetwork, a GaussianMixture, a HiddenMarkov model, and a TimeSeriesTransformer – which can be combined flexibly. A hybrid module, the VariationalRecurrentMarkovGaussianTransformer, integrates variational inference with recurrent dynamics, discrete Markovian state transitions, and Gaussian mixture outputs, enhanced by transformer-based attention. Meta-architecture managers (MMTransformer and MMModel) orchestrate the assembly and interaction of these components. The MMM class encapsulates the overall model, providing specialized training strategies such as KL-annealing, gradient clipping, and variance clamping. This modular design enables flexible construction of purely Gaussian, purely Hidden Markov, or hybrid variational models within a coherent framework. This paper describes each component and their integration, highlighting the architecture’s capacity for capturing complex, multi-modal and temporal data structures.

## Introduction

Modeling complex data often requires capturing multiple modalities, temporal dependencies, and uncertainty in a unified way. Traditional approaches such as Gaussian Mixture Models (GMMs), Hidden Markov Models (HMMs), and Variational Autoencoders (VAEs) each address different aspects of this challenge. The MMM architecture combines these paradigms into a single flexible framework. It introduces a modular design where each component – for encoding inputs, decoding outputs, capturing sequence dynamics, and modeling probabilistic mixtures – can be included or interchanged. This enables a single implementation to support a wide range of latent-variable models, from simple mixture density networks to sophisticated variational sequence models. The MMM framework thus aims to provide a general-purpose solution for learning from multi-modal, sequential data, leveraging modern neural architectures like transformers alongside classical probabilistic models.

## Methodology

### Architectural Overview

The MMM architecture is organized into modular components that collectively define the network’s structure. Data flows through an Encoder into latent representations, optionally through a RecurrentNetwork or attention mechanisms, and then through a Decoder that outputs parameters for a probabilistic model of the data. Core probabilistic components include the GaussianMixture and HiddenMarkov modules, which specify how the output distribution is formed. The TimeSeriesTransformer module can process sequences to capture long-range dependencies. Together, these pieces can form a hybrid model exemplified by the VariationalRecurrentMarkovGaussianTransformer (VRMGT). The MMTransformer and MMModel classes act as meta-managers that configure and link these modules, while the top-level MMM class unifies the training process. Each component is described in detail below.

### Encoder and Decoder

The Encoder is a neural network that maps raw input data into a latent feature space. In variational settings, the Encoder may output parameters of a latent distribution (for example, mean and log-variance of a Gaussian), enabling sampling via the reparameterization trick. In non-variational settings, the Encoder produces deterministic features. The design of the Encoder can vary (e.g., multilayer perceptron, convolutional network, etc.), but its role is always to extract informative latent representations from inputs.

The Decoder is a complementary network that maps latent representations back to the data space or to parameters of an output distribution. In practice, the Decoder often outputs the sufficient parameters of a probabilistic model for the data (such as mixture weights, means, and variances in a GaussianMixture). For example, in a variational autoencoder (VAE) setting, the Decoder uses sampled latent variables to generate a reconstruction of the input. When combined with mixture or sequence models, the Decoder may take as input both latent variables and recurrent or transformer states to produce final output distribution parameters. In this architecture, the Encoder and Decoder work together to capture complex, potentially nonlinear transformations between observed data and latent spaces.

### RecurrentNetwork

The RecurrentNetwork module provides explicit modeling of temporal or sequential dynamics. It typically consists of one or more recurrent layers (such as LSTM or GRU units) that process a sequence of inputs or latent states. In the MMM framework, the RecurrentNetwork can be used in two ways:

1. As part of the Encoder/Decoder pipeline, where it processes a sequence of encoded inputs to capture temporal context.
2. As part of the latent model, where it updates a hidden state that represents temporal evolution of latent variables.

By incorporating recurrence, the model can capture short- and medium-range dependencies in time series data. When training, the RecurrentNetwork’s hidden state is updated step-by-step through the sequence, allowing the network to remember past information. Because recurrent nets are prone to exploding gradients on long sequences, gradient clipping is often applied within the MMM training regime to maintain stability.

### GaussianMixture

The GaussianMixture component models the output distribution as a weighted sum of Gaussian probability densities. Specifically, this module produces a set of means, variances (or covariances), and mixture weights for a predefined number of Gaussian components. The mixture model allows the overall distribution to be multi-modal and heteroscedastic, capturing complex uncertainty in the data. In practice, the Decoder outputs feed through parameterizing layers (for means and variances) and a softmax (for mixture weights). The GaussianMixture then defines the log-likelihood of observations given these parameters.

During training, the loss for this component is typically the negative log-likelihood of data under the mixture model. The MMM framework also includes variance clamping – a mechanism that constrains variance values to avoid numerical underflow or collapse (for example, by enforcing a minimum variance). This prevents pathological solutions where a component’s variance approaches zero.

### HiddenMarkov

The HiddenMarkov module introduces discrete latent states with Markovian transitions between time steps. In this setup, each time step is associated with a latent state drawn from a finite set, and the transition probabilities between states are governed by a Markov transition matrix. Each discrete state can correspond to a different regime or mode of the data.

In practice, the HiddenMarkov component may output the probability distribution over states at each time, and the mixture of emissions (e.g., the parameters of a GaussianMixture) can be conditioned on the current state. The HiddenMarkov model can be trained either in a fully differentiable manner (using methods such as the forward-backward algorithm integrated into the network) or by treating state paths as latent and optimizing via variational inference. Its inclusion allows the network to capture temporal structure where different segments of the sequence exhibit distinct characteristics. This is especially valuable for capturing piecewise-stationary behavior in time series.

### TimeSeriesTransformer

The TimeSeriesTransformer module applies attention mechanisms to sequential data. Unlike recurrent networks, transformers use self-attention to directly model interactions between all positions in the sequence, enabling long-range dependencies to be captured more effectively. In the MMM architecture, the TimeSeriesTransformer can serve as an alternative or complement to recurrent processing.

It takes a sequence of inputs or latent representations (for example, the output of the Encoder over time, or a sequence of latent samples) and computes context-aware representations via multi-head self-attention layers. These enriched representations can then feed into subsequent modules (such as the Decoder or the hybrid VRMGT module). By incorporating positional encoding or temporal embeddings, the transformer can handle sequences of varying length and irregular sampling. In summary, the TimeSeriesTransformer enhances the model’s ability to learn global temporal structure, complementing the local state transitions captured by the RecurrentNetwork and HiddenMarkov components.

### Hybrid VariationalRecurrentMarkovGaussianTransformer (VRMGT)

The VariationalRecurrentMarkovGaussianTransformer is a composite hybrid model that synthesizes the previous components into a unified latent-variable framework. It performs variational inference over a sequence of latent variables that evolve according to both continuous (via the RecurrentNetwork) and discrete (via HiddenMarkov) dynamics, and produces outputs modeled by a Gaussian mixture.

At each time step, the model may use the Encoder to infer the parameters of a latent (continuous) Gaussian variable and a latent discrete state. The discrete state transitions follow an HMM, while the continuous latent evolves through a recurrent update. The TimeSeriesTransformer can optionally provide global context to each step via attention. The Decoder then takes these latents and produces the parameters of a GaussianMixture distribution.

Training this model involves maximizing a variational lower bound: the reconstruction log-likelihood of data under the Gaussian mixture, minus KL terms between the inferred latent distributions and their priors. The MMM class’s KL-annealing schedule gradually increases the weight of the KL terms, preventing early over-regularization and encouraging richer latent representations. Overall, the VRMGT can be seen as a generalization of a variational recurrent neural network (VRNN) that also incorporates discrete state switching (Markov) and attention-based context (Transformer).

### Meta-Architecture Managers: MMTransformer and MMModel

The classes MMTransformer and MMModel serve as high-level managers for configuring and assembling the MMM components.

* **MMTransformer**: Integrates multiple sub-models or experts within a transformer-like structure, potentially using a mixture-of-experts scheme. Each attention head or output head may correspond to a different latent state or mixture component.
* **MMModel**: Supervises the overall model configuration by selecting active components and how they are connected. It provides a single interface to instantiate complex model variants.

Through configuration, MMModel can create a standard VAE, a pure GMM, a pure HMM, or the full hybrid VRMGT. These meta-classes enforce a consistent data flow and simplify experiments with different architectural combinations.

### The MMM Class and Training Mechanisms

The MMM (Multi-Mixture Model) class encapsulates the complete model and its training routine. It integrates all chosen components and implements loss computation and optimization.

#### Training Strategies

* **KL Annealing**: Gradually increases the KL divergence term in the loss to avoid latent collapse.
* **Gradient Clipping**: Clips gradient norms to maintain numerical stability during training.
* **Variance Clamping**: Enforces bounds on predicted variances to prevent numerical instability in GaussianMixture outputs.

The MMM class also handles optimizer configuration, batching, learning-rate schedules, and custom regularization.

## Modularity and Flexibility

Modularity is central to the MMM architecture. Each component is a self-contained module with a clear interface. Components can be enabled or disabled via configuration, allowing users to construct:

* A model with only Encoder, Decoder, and GMM output (mixture density network).
* A model with HiddenMarkov and Transformer modules (capturing switching dynamics and attention).

Meta-managers ensure interoperability, and the architecture supports classical to modern deep variational models with shared code. It is also easily extensible.

## Conclusion

The MMM architecture provides a comprehensive and flexible framework for modeling complex data. By combining neural encoders and decoders with probabilistic components like Gaussian mixtures and Markov chains, and enhancing sequence modeling with recurrent and transformer layers, it addresses a wide range of modeling needs. The meta-architecture managers enable seamless module assembly, and the MMM class supports robust training via KL annealing, gradient clipping, and variance clamping. Modular design enables use in isolation or combination, supporting mixture-based, sequential, or hybrid models. MMM is a versatile tool for capturing multi-modal outputs, latent structure, and temporal dependencies.

## Use Examples

The `MMM` (Mixture Model Manager) class is a powerful unified wrapper that manages multiple model types (`GMM`, `HMM`, and the deep hybrid `VariationalRecurrentMarkovGaussianTransformer` model). Here's a complete breakdown of **how to use and train all models** via the `MMM` class, including all the options and variations you can apply. First make sure you have torch installed.

---
```python
pip install torch
```
## ✅ 1. **Train and Add a GMM**

### Usage:

```python
mmm = MMM()
model_id = mmm.fit_and_add(data, model_type='gmm', n_components=5)
```

### Options:

* `data`: torch tensor or numpy array of shape `[samples, features]`
* `n_components`: number of mixture components (GMM)

---

## ✅ 2. **Train and Add an HMM**

### Usage:

```python
model_id = mmm.fit_and_add(data, model_type='hmm', n_components=4, covariance_type='diag')
```

### Options:

* `n_components`: number of hidden states
* `covariance_type`: `"full"` or `"diag"` or others supported by your `MMModel`

---

## ✅ 3. **Train and Add a Deep MMM (VRMGT)**

### Usage:

```python
model_id = mmm.fit_and_add(
    data=data,
    model_type='mmm',
    input_dim=128,
    hidden_dim=256,
    z_dim=32,
    rnn_hidden=64,
    num_states=4,
    n_mix=5,
    trans_d_model=64,
    trans_nhead=4,
    trans_layers=2,
    output_dim=128,
    epochs=200,
    kl_anneal_epochs=50,
    lr=1e-4,
    clip_norm=5.0
)
```

### Key Arguments:

* `input_dim`: input feature dimension
* `hidden_dim`: MLP/encoder hidden size
* `z_dim`: latent variable dimension
* `rnn_hidden`: RNN hidden dimension
* `num_states`: number of discrete latent states (HMM-style)
* `n_mix`: number of Gaussian components per state
* `trans_d_model`: transformer model dimension
* `trans_nhead`: number of attention heads
* `trans_layers`: transformer layers
* `output_dim`: same as input\_dim unless doing feature conversion
* `epochs`: training iterations
* `kl_anneal_epochs`: warm-up period for KLD loss
* `clip_norm`: gradient clipping threshold
* `lr`: learning rate

---

## ✅ 4. **Evaluate / Identify with Models**

You can compare how well each model fits a new sequence:

### Example: Loss comparison for classification

```python
losses = {
    model_id: F.mse_loss(mmm.models[model_id](input_sequence)['reconstruction'], input_sequence).item()
    for model_id in mmm.models
}
predicted = min(losses, key=losses.get)
```

Or use HMM score/log-likelihood:

```python
score = mmm.score(model_id, input_sequence)
log_likelihoods = mmm.get_log_likelihoods(model_id, input_sequence)
```

---

## ✅ 5. **Export / Import Models**

### Save state dict:

```python
mmm.export_model(model_id='speakerA', filepath='speakerA.pth')
```

### Load from file or state dict:

```python
mmm.import_model('speakerA', 'speakerA.pth')
```

---

## ✅ 6. **Extract Model Parameters**

For GMM/HMM models:

```python
means = mmm.get_means(model_id='speakerA')
variances = mmm.get_variances('speakerA')
weights = mmm.get_weights('speakerA')
```

---

## ✅ 7. **Persistence (Full Object Save/Load)**

### Save full manager:

```python
mmm.save('mmm_full.pth')
```

### Load manager later:

```python
mmm = MMM.load('mmm_full.pth')
```

---

## ✅ 8. **Advanced Data Selection (for MMModel)**

For GMM/HMM that track multiple data IDs:

```python
loglik = mmm.get_log_likelihoods(model_id='speakerA', X=test_data, data_ids=['id1','id2'])
```

---

## ✅ Summary Table

| Action                  | model\_type | Required Arguments (for fit\_and\_add)                                                                                                                                 | Example Function                         |
| ----------------------- | ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| Train simple model      | 'gmm'       | `data`, `n_components`                                                                                                                                                 | `fit_and_add(...)`                       |
| Train sequential model  | 'hmm'       | `data`, `n_components`, `covariance_type` (optional, defaults to `'diag'`)                                                                                             | `fit_and_add(...)`                       |
| Train hybrid deep model | 'mmm'       | `data`, plus all the following:• `input_dim`• `hidden_dim`• `z_dim`• `rnn_hidden`• `num_states`• `n_mix`• `trans_d_model`• `trans_nhead`• `trans_layers`• `output_dim` | `fit_and_add(...)`                       |
| Evaluate likelihood     | any         | `X`, `model_id`, `data_ids` (optional)                                                                                                                                 | `score(...)`, `get_log_likelihoods(...)` |
| Save/Load model state   | any         | `model_id`, `filepath` or `state_dict`                                                                                                                                 | `export_model`, `import_model`           |
| Full object persistence | N/A         | File `path`                                                                                                                                                            | `MMM.save/load(...)`                     |

### ✅ Required Arguments for 'mmm' (Hybrid Deep Model)

| Argument        | Description                                            |
| --------------- | ------------------------------------------------------ |
| `data`          | Input tensor of shape `[batch, seq_len, input_dim]`    |
| `input_dim`     | Dimensionality of each input feature vector            |
| `hidden_dim`    | Hidden size for the encoder MLP                        |
| `z_dim`         | Latent dimension for variational sampling              |
| `rnn_hidden`    | Hidden size for the RNN used inside the model          |
| `num_states`    | Number of discrete latent states (like HMM states)     |
| `n_mix`         | Number of Gaussian components per state                |
| `trans_d_model` | Transformer model hidden size                          |
| `trans_nhead`   | Number of attention heads                              |
| `trans_layers`  | Number of Transformer encoder layers                   |
| `output_dim`    | Output feature dimension (usually same as `input_dim`) |

### 🔧 Optional Training Args for 'mmm'

| Optional Arg       | Description                                              | Default          |
| ------------------ | -------------------------------------------------------- | ---------------- |
| `epochs`           | Number of training epochs                                | 100              |
| `lr`               | Learning rate for optimizer                              | 1e-4             |
| `clip_norm`        | Max norm for gradient clipping                           | 5.0              |
| `kl_anneal_epochs` | Epochs over which KL divergence weight ramps from 0 to 1 | 0 (no annealing) |
| `tgt`              | Optional target for conditional models (if needed)       | None             |
