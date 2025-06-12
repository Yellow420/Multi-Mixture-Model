# By: Chance Brownfield
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import string
import math

# --- Building Blocks ---

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        return torch.sigmoid(self.fc_out(h))


class RecurrentNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_states):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.state_emissions = nn.Linear(hidden_dim, num_states)
        self.transition_matrix = nn.Parameter(torch.randn(num_states, num_states))

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        emissions = F.log_softmax(self.state_emissions(rnn_out), dim=-1)
        transitions = F.log_softmax(self.transition_matrix, dim=-1)
        return emissions, transitions


class GaussianMixture(nn.Module):
    def __init__(self, n_components, n_features):
        super().__init__()
        self.n_components = n_components
        self.n_features = n_features
        self.logits = nn.Parameter(torch.zeros(n_components))
        self.means = nn.Parameter(torch.randn(n_components, n_features))
        self.log_vars = nn.Parameter(torch.zeros(n_components, n_features))

    def get_weights(self):
        return F.softmax(self.logits, dim=0)

    def get_means(self):
        return self.means

    def get_variances(self):
        return torch.exp(self.log_vars)

    def log_prob(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=self.means.dtype, device=self.means.device)
        else:
            X = X.to(self.means.device).type(self.means.dtype)
        N, D = X.shape
        diff = X.unsqueeze(1) - self.means.unsqueeze(0)
        inv_vars = torch.exp(-self.log_vars)
        exp_term = -0.5 * torch.sum(diff * diff * inv_vars.unsqueeze(0), dim=2)
        log_norm = -0.5 * (torch.sum(self.log_vars, dim=1) + D * math.log(2 * math.pi))
        comp_log_prob = exp_term + log_norm.unsqueeze(0)
        log_weights = F.log_softmax(self.logits, dim=0)
        weighted = comp_log_prob + log_weights.unsqueeze(0)
        return torch.logsumexp(weighted, dim=1)

    def get_log_likelihoods(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=self.means.dtype, device=self.means.device)
        else:
            X = X.to(self.means.device).type(self.means.dtype)
        with torch.no_grad():
            ll = self.log_prob(X)
        return ll.cpu().numpy()

    def score(self, X):
        ll = self.get_log_likelihoods(X)
        return float(ll.mean())


class HiddenMarkov(nn.Module):
    def __init__(self, n_states, n_mix, n_features):
        super().__init__()
        self.n_states = n_states
        self.n_mix = n_mix
        self.n_features = n_features
        self.pi_logits = nn.Parameter(torch.zeros(n_states))
        self.trans_logits = nn.Parameter(torch.zeros(n_states, n_states))
        self.weight_logits = nn.Parameter(torch.zeros(n_states, n_mix))
        self.means = nn.Parameter(torch.randn(n_states, n_mix, n_features))
        self.log_vars = nn.Parameter(torch.zeros(n_states, n_mix, n_features))

    def get_initial_prob(self):
        return F.softmax(self.pi_logits, dim=0)

    def get_transition_matrix(self):
        return F.softmax(self.trans_logits, dim=1)

    def get_weights(self):
        return F.softmax(self.weight_logits, dim=1)

    def get_means(self):
        return self.means

    def get_variances(self):
        return torch.exp(self.log_vars)

    def log_prob(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=self.means.dtype, device=self.means.device)
        else:
            X = X.to(self.means.device).type(self.means.dtype)
        T, D = X.shape
        diff = X.unsqueeze(1).unsqueeze(2) - self.means.unsqueeze(0)
        inv_vars = torch.exp(-self.log_vars)
        exp_term = -0.5 * torch.sum(diff * diff * inv_vars.unsqueeze(0), dim=3)
        log_norm = -0.5 * (torch.sum(self.log_vars, dim=2) + D * math.log(2 * math.pi))
        comp_log_prob = exp_term + log_norm.unsqueeze(0)
        log_mix_weights = F.log_softmax(self.weight_logits, dim=1)
        weighted = comp_log_prob + log_mix_weights.unsqueeze(0)
        emission_log_prob = torch.logsumexp(weighted, dim=2)
        log_pi = F.log_softmax(self.pi_logits, dim=0)
        log_A = F.log_softmax(self.trans_logits, dim=1)
        log_alpha = torch.zeros(T, self.n_states, dtype=X.dtype, device=X.device)
        log_alpha[0] = log_pi + emission_log_prob[0]
        for t in range(1, T):
            prev = log_alpha[t-1].unsqueeze(1)
            log_alpha[t] = emission_log_prob[t] + torch.logsumexp(prev + log_A, dim=1)
        return torch.logsumexp(log_alpha[-1], dim=0)

    def get_log_likelihoods(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=self.means.dtype, device=self.means.device)
        else:
            X = X.to(self.means.device).type(self.means.dtype)
        with torch.no_grad():
            if X.dim() == 3:
                return [self.log_prob(seq).item() for seq in X]
            else:
                return [self.log_prob(X).item()]

    def score(self, X):
        lls = self.get_log_likelihoods(X)
        return float(sum(lls) / len(lls))


class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, output_dim):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers
        )
        self.output_proj = nn.Linear(d_model, output_dim)

    def forward(self, src, tgt):
        src_emb = self.input_proj(src)
        tgt_emb = self.input_proj(tgt)
        out = self.transformer(src_emb, tgt_emb)
        return self.output_proj(out)


class VariationalRecurrentMarkovGaussianTransformer(nn.Module):
    """
    Variational Encoder + RNN-HMM + Hidden GMM + Transformer hybrid.
    """
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 z_dim,
                 rnn_hidden,
                 num_states,
                 n_mix,
                 trans_d_model,
                 trans_nhead,
                 trans_layers,
                 output_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, z_dim)
        self.decoder = Decoder(z_dim, hidden_dim, input_dim)
        self.rn = RecurrentNetwork(z_dim, rnn_hidden, num_states)
        self.hm = HiddenMarkov(num_states, n_mix, z_dim)
        self.transformer = TimeSeriesTransformer(
            input_dim=z_dim,
            d_model=trans_d_model,
            nhead=trans_nhead,
            num_layers=trans_layers,
            output_dim=output_dim
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, tgt=None):
        if x.dim() == 3:
            T, B, _ = x.size()
            zs, mus, logvars = [], [], []
            for t in range(T):
                mu_t, logvar_t = self.encoder(x[t])
                z_t = self.reparameterize(mu_t, logvar_t)
                zs.append(z_t)
                mus.append(mu_t)
                logvars.append(logvar_t)
            zs = torch.stack(zs)
            mus = torch.stack(mus)
            logvars = torch.stack(logvars)
        else:
            mu, logvar = self.encoder(x)
            zs = self.reparameterize(mu, logvar)
            mus, logvars = mu, logvar

        recon = self.decoder(zs.view(-1, zs.size(-1))).view_as(x)
        emissions, transitions = self.rn(zs.permute(1,0,2))
        flat_z = zs.view(-1, zs.size(-1))
        seq_ll = self.hm.log_prob(flat_z)
        hgmm_ll = seq_ll.view(1, 1, 1).expand_as(emissions)
        trans_out = self.transformer(zs, tgt) if tgt is not None else None

        return {
            'reconstruction': recon,
            'mu': mus,
            'logvar': logvars,
            'emissions': emissions,
            'transitions': transitions,
            'hgmm_log_likelihood': hgmm_ll,
            'transformer_out': trans_out
        }

    def loss(self, x, outputs):
        recon, mu, logvar = outputs['reconstruction'], outputs['mu'], outputs['logvar']
        recon_loss = F.mse_loss(recon, x, reduction='sum')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        hgmm_nll = -torch.sum(outputs['hgmm_log_likelihood'])
        return recon_loss + kld + hgmm_nll


class MMTransformer(nn.Module):
    """Multi-Mixture Transformrer."""
    def __init__(self, n_components, n_features, model_type='gmm', n_mix=1):
        super().__init__()
        self.model_type = model_type.lower()
        self.n_features = n_features
        self.gmms = []
        self.hgmm_models = {}
        self.active_hmm = None
        if self.model_type == 'gmm':
            self.gmm = GaussianMixture(n_components, n_features)
        elif self.model_type == 'hgmm':
            self.hm = HiddenMarkov(n_components, n_mix, n_features)
        else:
            raise ValueError("model_type must be 'gmm' or 'hgmm'")

    def _prepare_tensor(self, X):
        return torch.tensor(X, dtype=torch.float32) if not isinstance(X, torch.Tensor) else X.float()

    def fit(self, X, init_params=None, lr=1e-2, epochs=100, verbose=False, data_id=None):
        if init_params is not None:
            self.import_model(init_params)

        X_tensor = self._prepare_tensor(X).to(next(self.parameters()).device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()
            if self.model_type == 'gmm':
                loss = -torch.mean(self.gmm.log_prob(X_tensor))
            else:
                if X_tensor.dim() == 3:
                    loss = -sum(self.hm.log_prob(seq) for seq in X_tensor) / X_tensor.size(0)
                else:
                    loss = -self.hm.log_prob(X_tensor)
            loss.backward()
            optimizer.step()
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        if self.model_type == 'gmm':
            if data_id is None:
                data_id = len(self.gmms)
            while isinstance(data_id, int) and data_id < len(self.gmms) and self.gmms[data_id] is not None:
                data_id += 1
            if data_id == len(self.gmms):
                self.gmms.append(self.gmm)
            else:
                self.gmms[data_id] = self.gmm
        else:
            if data_id is None:
                while True:
                    data_id = ''.join(random.choices(string.ascii_lowercase, k=6))
                    if data_id not in self.hm_models:
                        break
            self.hm_models[data_id] = self.hm
            self.active_hmm = data_id

        return data_id

    def unfit(self, data_id):
        if isinstance(data_id, int):
            if 0 <= data_id < len(self.gmms):
                del self.gmms[data_id]
            else:
                raise ValueError(f"GMM with id {data_id} does not exist.")
        elif isinstance(data_id, str):
            if data_id in self.hm_models:
                del self.hm_models[data_id]
                if self.active_hmm == data_id:
                    self.active_hmm = None
            else:
                raise ValueError(f"HMM model with name '{data_id}' does not exist.")
        else:
            raise TypeError("data_id must be an int (GMM) or str (HMM)")

    def check_data(self):
        data = {i: 'gmm' for i in range(len(self.gmms))}
        data.update({name: 'hmm' for name in self.hm_models.keys()})
        return data

    def score(self, X):
        with torch.no_grad():
            X_tensor = self._prepare_tensor(X).to(next(self.parameters()).device)
            if self.model_type == 'gmm':
                return float(self.gmm.log_prob(X_tensor).mean().cpu().item())
            else:
                if X_tensor.dim() == 3:
                    return float(sum(self.hm.log_prob(seq).item() for seq in X_tensor) / X_tensor.size(0))
                else:
                    return float(self.hm.log_prob(X_tensor).cpu().item())

    def get_log_likelihoods(self, X):
        with torch.no_grad():
            X_tensor = self._prepare_tensor(X).to(next(self.parameters()).device)
            if self.model_type == 'gmm':
                return self.gmm.log_prob(X_tensor).cpu().numpy()
            else:
                if X_tensor.dim() == 3:
                    return [self.hm.log_prob(seq).item() for seq in X_tensor]
                else:
                    return [self.hm.log_prob(X_tensor).item()]

    def get_means(self):
        return (self.gmm if self.model_type == 'gmm' else self.hgmm).get_means().cpu().detach().numpy()

    def get_variances(self):
        return (self.gmm if self.model_type == 'gmm' else self.hgmm).get_variances().cpu().detach().numpy()

    def get_weights(self):
        return (self.gmm if self.model_type == 'gmm' else self.hgmm).get_weights().cpu().detach().numpy()

    def export_model(self, filepath=None):
        state = self.state_dict()
        if filepath:
            torch.save(state, filepath)
        return state

    def import_model(self, source):
        if isinstance(source, str):
            state = torch.load(source)
        elif isinstance(source, dict):
            state = source
        else:
            raise ValueError("Unsupported source for import_model")
        self.load_state_dict(state)


class MMModel(nn.Module):
    """Multi-Mixture Model."""
    def __init__(self):
        super().__init__()
        self.gmms = []  # List of GaussianMixture models
        self.hm_models = {}  # Dict of HM models keyed by string IDs
        self.active_hmm = None  # Optional: active HGMM for scoring/fitting

    def _generate_unique_id(self):
        while True:
            candidate = ''.join(random.choices(string.ascii_lowercase, k=6))
            if candidate not in self.hm_models:
                return candidate

    def fit(self, data=None, model_type='gmm', n_components=1, n_features=1, n_mix=1,
            data_id=None, init_params=None, lr=1e-2, epochs=100):
        """
        Fit or absorb a model:
        - If `data` is a tensor/array, fit a new model.
        - If `data` is a pre-trained model, absorb it directly.
        - `data_id` determines storage; if None, generate a unique one.
        """
        if model_type == 'gmm':
            if data_id is None:
                data_id = len(self.gmms)
                while data_id < len(self.gmms) and self.gmms[data_id] is not None:
                    data_id += 1
            if isinstance(data, GaussianMixture):
                # Absorb pretrained model
                if data_id < len(self.gmms):
                    self.gmms[data_id] = data
                else:
                    while len(self.gmms) < data_id:
                        self.gmms.append(None)
                    self.gmms.append(data)
            else:
                # Train new model
                model = MMTransformer(n_components, n_features, model_type='gmm')
                model.fit(data, init_params=init_params, lr=lr, epochs=epochs)
                if data_id < len(self.gmms):
                    self.gmms[data_id] = model.gmm
                else:
                    while len(self.gmms) < data_id:
                        self.gmms.append(None)
                    self.gmms.append(model.gmm)
        elif model_type == 'hmm':
            if data_id is None:
                data_id = self._generate_unique_id()
            if isinstance(data, HiddenMarkov):
                self.hm_models[data_id] = data
            else:
                model = MMTransformer(n_components, n_features, model_type='hmm', n_mix=n_mix)
                model.fit(data, init_params=init_params, lr=lr, epochs=epochs)
                self.hm_models[data_id] = model.hgmm
        else:
            raise ValueError("model_type must be 'gmm' or 'hmm'")
        return data_id

    def export_model(self, data_id):
        """
        Export the model associated with the data_id.
        Returns a GaussianMixture or HiddenMarkov instance.
        """
        if isinstance(data_id, int):
            if 0 <= data_id < len(self.gmms):
                return self.gmms[data_id]
            else:
                raise ValueError(f"GMM with id {data_id} does not exist.")
        elif isinstance(data_id, str):
            if data_id in self.hm_models:
                return self.hm_models[data_id]
            else:
                raise ValueError(f"HMM model with name '{data_id}' does not exist.")
        else:
            raise TypeError("data_id must be an int (GMM) or str (HMM)")

    def unfit(self, data_id):
        """
        Remove a model from the internal storage (GMM or HMM).
        """
        if isinstance(data_id, int):
            if 0 <= data_id < len(self.gmms):
                del self.gmms[data_id]
            else:
                raise ValueError(f"GMM with id {data_id} does not exist.")
        elif isinstance(data_id, str):
            if data_id in self.hgmm_models:
                del self.hm_models[data_id]
                if self.active_hmm == data_id:
                    self.active_hmm = None
            else:
                raise ValueError(f"HMM model with name '{data_id}' does not exist.")
        else:
            raise TypeError("data_id must be an int (GMM) or str (HMM)")

    def check_data(self):
        """
        Returns a dict mapping each stored data's ID to its type.

        - Integer keys → 'gmm'
        - String keys   → 'hmm'
        """
        data = {i: 'gmm' for i in range(len(self.gmms)) if self.gmms[i] is not None}
        data.update({name: 'hmm' for name in self.hm_models.keys()})
        return data

    def _all_ids(self):
        return list(self.check_data().keys())

    def _normalize_ids(self, data_ids):
        if data_ids is None:
            return self._all_ids()
        if isinstance(data_ids, (int, str)):
            return [data_ids]
        return list(data_ids)

    def _get_submodel(self, data_id):
        if isinstance(data_id, int):
            return self.gmms[data_id]
        return self.hm_models[data_id]

    def get_means(self, data_ids=None):
        """
        If data_ids is None, returns a dict {id: means} for all components;
        if a single id, returns just that component’s means (numpy array);
        if a list/tuple, returns a dict.
        """
        ids = self._normalize_ids(data_ids)
        out = {d: self._get_submodel(d).get_means() for d in ids}
        # unwrap singletons
        if isinstance(data_ids, (int, str)):
            return out[ids[0]]
        return out

    def get_variances(self, data_ids=None):
        ids = self._normalize_ids(data_ids)
        out = {d: self._get_submodel(d).get_variances() for d in ids}
        if isinstance(data_ids, (int, str)):
            return out[ids[0]]
        return out

    def get_weights(self, data_ids=None):
        ids = self._normalize_ids(data_ids)
        out = {d: self._get_submodel(d).get_weights() for d in ids}
        if isinstance(data_ids, (int, str)):
            return out[ids[0]]
        return out

    def score(self, X, data_ids=None):
        """
        Average log-likelihood(s) of X under each specified component.
        """
        ids = self._normalize_ids(data_ids)
        out = {d: self._get_submodel(d).score(X) for d in ids}
        if isinstance(data_ids, (int, str)):
            return out[ids[0]]
        return out

    def get_log_likelihoods(self, X, data_ids=None):
        """
        Per-sample log-likelihood(s) of X under each specified component.
        """
        ids = self._normalize_ids(data_ids)
        out = {d: self._get_submodel(d).get_log_likelihoods(X) for d in ids}
        if isinstance(data_ids, (int, str)):
            return out[ids[0]]
        return out

class MMM(nn.Module):
    """
    Manager for multiple models: GMM, HMM, and VariationalRecurrentMarkovGaussianTransformer.
    This version uses MSE for reconstruction, gradient clipping, variance clamping, numerical safeguards, and optional annealing.
    """
    def __init__(self):
        super().__init__()
        self.models = nn.ModuleDict()

    def _generate_unique_id(self, prefix='model'):
        while True:
            candidate = f"{prefix}_{''.join(random.choices(string.ascii_lowercase, k=6))}"
            if candidate not in self.models:
                return candidate

    def add_model(self, model: nn.Module, model_id: str = None):
        if model_id is None:
            model_id = self._generate_unique_id(model.__class__.__name__)
        if model_id in self.models:
            raise KeyError(f"Model with id '{model_id}' already exists.")
        self.models[model_id] = model
        return model_id

    def fit_and_add(self,
                    data,
                    model_type: str = 'gmm',
                    model_id: str = None,
                    kl_anneal_epochs: int = 0,
                    clip_norm: float = 5.0,
                    weight_decay: float = 1e-5,
                    **kwargs):
        model_type = model_type.lower()
        if model_type in ('gmm','hmm'):
            mm = MMModel()
            mm.fit(data, model_type=model_type, **kwargs)
            model = mm

        elif model_type == 'mmm':
            # build hybrid model
            model = VariationalRecurrentMarkovGaussianTransformer(
                kwargs.pop('input_dim'),
                kwargs.pop('hidden_dim'),
                kwargs.pop('z_dim'),
                kwargs.pop('rnn_hidden'),
                kwargs.pop('num_states'),
                kwargs.pop('n_mix'),
                kwargs.pop('trans_d_model'),
                kwargs.pop('trans_nhead'),
                kwargs.pop('trans_layers'),
                kwargs.pop('output_dim')
            )
            optim = torch.optim.Adam(model.parameters(), lr=kwargs.get('lr',1e-4), weight_decay=weight_decay)
            epochs = kwargs.get('epochs',100)
            x = data.float().to(next(model.parameters()).device)

            for epoch in range(epochs):
                model.train()
                optim.zero_grad()
                out = model(x, kwargs.get('tgt', None))

                # Reconstruction loss: MSE
                recon = out['reconstruction']
                recon_loss = F.mse_loss(recon, x, reduction='sum')

                # Clamp encoder logvars for KLD
                mu, logvar = out['mu'], out['logvar']
                logvar_clamped = torch.clamp(logvar, min=-10.0, max=10.0)
                kld = -0.5 * torch.sum(1 + logvar_clamped - mu.pow(2) - logvar_clamped.exp())

                # HMM NLL with clamped likelihoods
                hgmm_ll = out['hgmm_log_likelihood']
                # avoid extreme values
                hgmm_ll = torch.clamp(hgmm_ll, min=-1e6, max=1e6)
                hgmm_nll = -torch.sum(hgmm_ll)

                # numeric safe
                kld = torch.nan_to_num(kld, nan=0.0, posinf=1e8, neginf=-1e8)
                hgmm_nll = torch.nan_to_num(hgmm_nll, nan=0.0, posinf=1e8, neginf=-1e8)

                # Annealing weight
                anneal_w = min(1.0, epoch / kl_anneal_epochs) if kl_anneal_epochs > 0 else 1.0
                loss = recon_loss + anneal_w * (kld + hgmm_nll)

                # Backprop & gradient clipping
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optim.step()

                # Optional logging
                if epoch % max(1, epochs // 5) == 0:
                    print(f"Epoch {epoch}: recon={recon_loss.item():.1f}, kld={kld.item():.1f}, "
                          f"hmll={hgmm_nll.item():.1f}, anneal_w={anneal_w:.2f}")
        else:
            raise ValueError("model_type must be 'gmm','hmm', or 'mmm'")

        assigned_id = self.add_model(model, model_id)
        return assigned_id

    def export_model(self, model_id: str, filepath: str = None):
        if model_id not in self.models:
            raise KeyError(f"Model '{model_id}' not found.")
        model = self.models[model_id]
        state = model.state_dict()
        if filepath:
            torch.save(state, filepath)
        return state

    def import_model(self, model_id: str, source):
        if model_id not in self.models:
            raise KeyError(f"Model '{model_id}' not found.")
        model = self.models[model_id]
        if isinstance(source, str):
            state = torch.load(source)
        elif isinstance(source, dict):
            state = source
        else:
            raise ValueError("source must be filepath or state dict")
        model.load_state_dict(state)

    def _select_data(self, mm, fn, data_ids=None, *args, **kwargs):
        all_keys = list(mm.check_data().keys())
        if data_ids is None:
            ids = all_keys
        elif isinstance(data_ids, (list, tuple)):
            ids = data_ids
        else:
            ids = [data_ids]
        out = {d: fn(mm, d, *args, **kwargs) for d in ids}
        if not isinstance(data_ids, (list, tuple)) and data_ids is not None:
            return out[data_ids]
        return out

    def get_means(self, model_id: str, data_ids=None):
        mm = self.get_mmm(model_id)
        return self._select_data(
            mm,
            lambda m, d: m._get_submodel(d).get_means(),
            data_ids
        )

    def get_variances(self, model_id: str, data_ids=None):
        mm = self.get_mmm(model_id)
        return self._select_data(
            mm,
            lambda m, d: m._get_submodel(d).get_variances(),
            data_ids
        )

    def get_weights(self, model_id: str, data_ids=None):
        mm = self.get_mmm(model_id)
        return self._select_data(
            mm,
            lambda m, d: m._get_submodel(d).get_weights(),
            data_ids
        )

    def get_log_likelihoods(self, model_id: str, X, data_ids=None):
        mm = self.get_mmm(model_id)

        def fn(m, d):
            sub = m._get_submodel(d)
            return sub.get_log_likelihoods(X)

        return self._select_data(mm, fn, data_ids)

    def score(self, model_id: str, X, data_ids=None):
        mm = self.get_mmm(model_id)

        def fn(m, d):
            sub = m._get_submodel(d)
            return sub.score(X)

        return self._select_data(mm, fn, data_ids)

    def get_mmm(self, model_id: str):
        if model_id not in self.models:
            raise KeyError(f"Model '{model_id}' not found.")
        return self.models[model_id]

    def save(self, path: str):
        torch.save(self, path)

    @classmethod
    def load(cls, path: str):
        return torch.load(path, weights_only=False)