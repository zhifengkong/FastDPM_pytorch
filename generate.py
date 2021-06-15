import os
import argparse
import time
from tqdm import tqdm

import numpy as np
np.random.seed(0)

import torch
import torch.nn as nn
torch.manual_seed(0)

import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from model import Model
from config import diffusion_config


def _map_gpu(gpu):
    if gpu == 'cuda':
        return lambda x: x.cuda()
    else:
        return lambda x: x.to(torch.device('cuda:'+gpu))


def rescale(X, batch=True):
    if not batch:
        return (X - X.min()) / (X.max() - X.min())
    else:
        for i in range(X.shape[0]):
            X[i] = rescale(X[i], batch=False)
        return X


def std_normal(size):
    return map_gpu(torch.normal(0, 1, size=size))


def print_size(net):
    """
    Print the number of parameters of a network
    """
    if net is not None and isinstance(net, torch.nn.Module):
        module_parameters = filter(lambda p: p.requires_grad, net.parameters())
        params = sum([np.prod(p.size()) for p in module_parameters])
        print("{} Parameters: {:.6f}M".format(
            net.__class__.__name__, params / 1e6), flush=True)


def calc_diffusion_hyperparams(T, beta_0, beta_T):
    """
    Compute diffusion process hyperparameters

    Parameters:
    T (int):                    number of diffusion steps
    beta_0 and beta_T (float):  beta schedule start/end value, 
                                where any beta_t in the middle is linearly interpolated
    
    Returns:
    a dictionary of diffusion hyperparameters including:
        T (int), Beta/Alpha/Alpha_bar/Sigma (torch.tensor on cpu, shape=(T, ))
    """

    Beta = torch.linspace(beta_0, beta_T, T)
    Alpha = 1 - Beta
    Alpha_bar = Alpha + 0
    Beta_tilde = Beta + 0
    for t in range(1, T):
        Alpha_bar[t] *= Alpha_bar[t-1]
        Beta_tilde[t] *= (1-Alpha_bar[t-1]) / (1-Alpha_bar[t])
    Sigma = torch.sqrt(Beta_tilde)
    
    _dh = {}
    _dh["T"], _dh["Beta"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"] = T, Beta, Alpha, Alpha_bar, Sigma
    diffusion_hyperparams = _dh
    return diffusion_hyperparams


def bisearch(f, domain, target, eps=1e-8):
    """
    find smallest x such that f(x) > target

    Parameters:
    f (function):               function
    domain (tuple):             x in (left, right)
    target (float):             target value
    
    Returns:
    x (float)
    """
    # 
    sign = -1 if target < 0 else 1
    left, right = domain
    for _ in range(1000):
        x = (left + right) / 2 
        if f(x) < target:
            right = x
        elif f(x) > (1 + sign * eps) * target:
            left = x
        else:
            break
    return x


def get_VAR_noise(S, schedule='linear'):
    """
    Compute VAR noise levels

    Parameters:
    S (int):            approximante diffusion process length
    schedule (str):     linear or quadratic
    
    Returns:
    np array of noise levels, size = (S, )
    """
    target = np.prod(1 - np.linspace(diffusion_config["beta_0"], diffusion_config["beta_T"], diffusion_config["T"]))

    if schedule == 'linear':
        g = lambda x: np.linspace(diffusion_config["beta_0"], x, S)
        domain = (diffusion_config["beta_0"], 0.99)
    elif schedule == 'quadratic':
        g = lambda x: np.array([diffusion_config["beta_0"] * (1+i*x) ** 2 for i in range(S)])
        domain = (0.0, 0.95 / np.sqrt(diffusion_config["beta_0"]) / S)
    else:
        raise NotImplementedError

    f = lambda x: np.prod(1 - g(x))
    largest_var = bisearch(f, domain, target, eps=1e-4)
    return g(largest_var)


def get_STEP_step(S, schedule='linear'):
    """
    Compute STEP steps

    Parameters:
    S (int):            approximante diffusion process length
    schedule (str):     linear or quadratic
    
    Returns:
    np array of steps, size = (S, )
    """
    if schedule == 'linear':
        c = (diffusion_config["T"] - 1.0) / (S - 1.0)
        list_tau = [np.floor(i * c) for i in range(S)]
    elif schedule == 'quadratic':
        list_tau = np.linspace(0, np.sqrt(diffusion_config["T"] * 0.8), S) ** 2
    else:
        raise NotImplementedError

    return [int(s) for s in list_tau]


def _log_gamma(x):
    # Gamma(x+1) ~= sqrt(2\pi x) * (x/e)^x  (1 + 1 / 12x)
    y = x - 1
    return np.log(2 * np.pi * y) / 2 + y * (np.log(y) - 1) + np.log(1 + 1 / (12 * y))


def _log_cont_noise(t, beta_0, beta_T, T):
    # We want log_cont_noise(t, beta_0, beta_T, T) ~= np.log(Alpha_bar[-1].numpy())
    delta_beta = (beta_T - beta_0) / (T - 1)
    _c = (1.0 - beta_0) / delta_beta
    t_1 = t + 1
    return t_1 * np.log(delta_beta) + _log_gamma(_c + 1) - _log_gamma(_c - t_1 + 1)


# Standard DDPM generation
def STD_sampling(net, size, diffusion_hyperparams):
    """
    Perform the complete sampling step according to DDPM

    Parameters:
    net (torch network):            the model
    size (tuple):                   size of tensor to be generated, 
                                    usually is (number of audios to generate, channels=1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors

    Returns:
    the generated images in torch.tensor, shape=size
    """

    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, Beta = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Beta"]
    assert len(Alpha_bar) == T
    assert len(size) == 4

    Sigma = _dh["Sigma"]

    x = std_normal(size)
    with torch.no_grad():
        for t in range(T-1, -1, -1):
            diffusion_steps = t * map_gpu(torch.ones(size[0]))
            epsilon_theta = net(x, diffusion_steps)
            x = (x - (1-Alpha[t])/torch.sqrt(1-Alpha_bar[t]) * epsilon_theta) / torch.sqrt(Alpha[t])
            if t > 0:
                x = x + Sigma[t] * std_normal(size)
    return x


# STEP
def STEP_sampling(net, size, diffusion_hyperparams, user_defined_steps, kappa):
    """
    Perform the complete sampling step according to https://arxiv.org/pdf/2010.02502.pdf
    official repo: https://github.com/ermongroup/ddim

    Parameters:
    net (torch network):            the model
    size (tuple):                   size of tensor to be generated, 
                                    usually is (number of audios to generate, channels=1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors
    user_defined_steps (int list):  User defined steps (sorted)     
    kappa (float):                  factor multipled over sigma, between 0 and 1
    
    Returns:
    the generated images in torch.tensor, shape=size
    """
    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, _ = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]
    assert len(Alpha_bar) == T
    assert len(size) == 4
    assert 0.0 <= kappa <= 1.0

    T_user = len(user_defined_steps)
    user_defined_steps = sorted(list(user_defined_steps), reverse=True)

    x = std_normal(size)
    with torch.no_grad():
        for i, tau in enumerate(user_defined_steps):
            diffusion_steps = tau * map_gpu(torch.ones(size[0]))
            epsilon_theta = net(x, diffusion_steps)
            if i == T_user - 1:  # the next step is to generate x_0
                assert tau == 0
                alpha_next = torch.tensor(1.0) 
                sigma = torch.tensor(0.0) 
            else:
                alpha_next = Alpha_bar[user_defined_steps[i+1]]
                sigma = kappa * torch.sqrt((1-alpha_next) / (1-Alpha_bar[tau]) * (1 - Alpha_bar[tau] / alpha_next))
            x *= torch.sqrt(alpha_next / Alpha_bar[tau])
            c = torch.sqrt(1 - alpha_next - sigma ** 2) - torch.sqrt(1 - Alpha_bar[tau]) * torch.sqrt(alpha_next / Alpha_bar[tau])
            x += c * epsilon_theta + sigma * std_normal(size)
    return x


# VAR
def _precompute_VAR_steps(diffusion_hyperparams, user_defined_eta):
    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, Beta = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Beta"]
    assert len(Alpha_bar) == T

    # compute diffusion hyperparameters for user defined noise
    T_user = len(user_defined_eta)
    Beta_tilde = map_gpu(torch.from_numpy(user_defined_eta)).to(torch.float32)
    Gamma_bar = 1 - Beta_tilde
    for t in range(1, T_user):
        Gamma_bar[t] *= Gamma_bar[t-1]

    assert Gamma_bar[0] <= Alpha_bar[0] and Gamma_bar[-1] >= Alpha_bar[-1]
    
    continuous_steps = []
    with torch.no_grad():
        for t in range(T_user-1, -1, -1):
            t_adapted = None
            for i in range(T - 1):
                if Alpha_bar[i] >= Gamma_bar[t] > Alpha_bar[i+1]:
                    t_adapted = bisearch(f=lambda _t: _log_cont_noise(_t, Beta[0].cpu().numpy(), Beta[-1].cpu().numpy(), T), 
                                            domain=(i-0.01, i+1.01), 
                                            target=np.log(Gamma_bar[t].cpu().numpy()))
                    break
            if t_adapted is None:
                t_adapted = T - 1
            continuous_steps.append(t_adapted)  # must be decreasing
    return continuous_steps


def VAR_sampling(net, size, diffusion_hyperparams, user_defined_eta, kappa, continuous_steps):
    """
    Perform the complete sampling step according to user defined variances

    Parameters:
    net (torch network):            the model
    size (tuple):                   size of tensor to be generated, 
                                    usually is (number of audios to generate, channels=1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors
    user_defined_eta (np.array):    User defined noise       
    kappa (float):                  factor multipled over sigma, between 0 and 1
    continuous_steps (list):        continuous steps computed from user_defined_eta

    Returns:
    the generated images in torch.tensor, shape=size
    """

    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, Beta = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Beta"]
    assert len(Alpha_bar) == T
    assert len(size) == 4
    assert 0.0 <= kappa <= 1.0

    # compute diffusion hyperparameters for user defined noise
    T_user = len(user_defined_eta)
    Beta_tilde = map_gpu(torch.from_numpy(user_defined_eta)).to(torch.float32)
    Gamma_bar = 1 - Beta_tilde
    for t in range(1, T_user):
        Gamma_bar[t] *= Gamma_bar[t-1]

    assert Gamma_bar[0] <= Alpha_bar[0] and Gamma_bar[-1] >= Alpha_bar[-1]
    
    # print('begin sampling, total number of reverse steps = %s' % T_user)

    x = std_normal(size)
    with torch.no_grad():
        for i, tau in enumerate(continuous_steps):
            diffusion_steps = tau * map_gpu(torch.ones(size[0]))
            epsilon_theta = net(x, diffusion_steps)
            if i == T_user - 1:  # the next step is to generate x_0
                assert abs(tau) < 0.1
                alpha_next = torch.tensor(1.0) 
                sigma = torch.tensor(0.0) 
            else:
                alpha_next = Gamma_bar[T_user-1-i - 1]
                sigma = kappa * torch.sqrt((1-alpha_next) / (1-Gamma_bar[T_user-1-i]) * (1 - Gamma_bar[T_user-1-i] / alpha_next))
            x *= torch.sqrt(alpha_next / Gamma_bar[T_user-1-i])
            c = torch.sqrt(1 - alpha_next - sigma ** 2) - torch.sqrt(1 - Gamma_bar[T_user-1-i]) * torch.sqrt(alpha_next / Gamma_bar[T_user-1-i])
            x += c * epsilon_theta + sigma * std_normal(size)

    return x


def generate(output_name, model_path, model_config, 
             diffusion_config, approxdiff, generation_param, 
             n_generate, batchsize, n_exist):
    """
    Parameters:
    output_name (str):              save generated images to this folder
    model_path (str):               checkpoint file
    model_config (dic):             dic of model config
    diffusion_config (dic):         dic of diffusion config
    generation_param (dic):         parameter: user defined variance or user defined steps
    approxdiff (str):          diffusion style: STD, STEP, VAR
    n_generate (int):               number of generated samples
    batchsize (int):                batch size of training
    n_exist (int):                  existing number of samples

    Returns:
    Generated images (tensor):      (B, C, H, W) where C = 3
    """
    if batchsize > n_generate:
        batchsize = n_generate
    assert n_generate % batchsize == 0

    if 'generated' not in os.listdir():
        os.mkdir('generated')
    if output_name not in os.listdir('generated'):
        os.mkdir(os.path.join('generated', output_name))

    # map diffusion hyperparameters to gpu
    diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_config)
    for key in diffusion_hyperparams:
        if key is not "T":
            diffusion_hyperparams[key] = map_gpu(diffusion_hyperparams[key])

    # predefine model
    net = Model(**model_config)
    print_size(net)

    # load checkpoint
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        net.load_state_dict(checkpoint)
        net = map_gpu(net)
        net.eval()
        print('checkpoint successfully loaded')
    except:
        raise Exception('No valid model found')

    # sampling
    C, H, W = model_config["in_channels"], model_config["resolution"], model_config["resolution"]
    for i in tqdm(range(n_exist // batchsize, n_generate // batchsize)):
        if approxdiff == 'STD':
            Xi = STD_sampling(net, (batchsize, C, H, W), diffusion_hyperparams)
        elif approxdiff == 'STEP':
            user_defined_steps = generation_param["user_defined_steps"]
            Xi = STEP_sampling(net, (batchsize, C, H, W), 
                               diffusion_hyperparams,
                               user_defined_steps,
                               kappa=generation_param["kappa"])
        elif approxdiff == 'VAR':
            user_defined_eta = generation_param["user_defined_eta"]
            continuous_steps = _precompute_VAR_steps(diffusion_hyperparams, user_defined_eta)
            Xi = VAR_sampling(net, (batchsize, C, H, W),
                              diffusion_hyperparams,
                              user_defined_eta,
                              kappa=generation_param["kappa"],
                              continuous_steps=continuous_steps)
        
        # save image
        for j, x in enumerate(rescale(Xi)):
            index = i * batchsize + j 
            save_image(x, fp=os.path.join('generated', output_name, '{}.jpg'.format(index)))
        save_image(make_grid(rescale(Xi)[:64]), fp=os.path.join('generated', '{}.jpg'.format(output_name)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # dataset and model
    parser.add_argument('-name', '--name', type=str, choices=["cifar10", "lsun_bedroom", "lsun_church", "lsun_cat", "celeba64"],
                        help='Name of experiment')
    parser.add_argument('-ema', '--ema', action='store_true', help='Whether use ema')

    # fast generation parameters
    parser.add_argument('-approxdiff', '--approxdiff', type=str, choices=['STD', 'STEP', 'VAR'], help='approximate diffusion process')
    parser.add_argument('-kappa', '--kappa', type=float, default=1.0, help='factor to be multiplied to sigma')
    parser.add_argument('-S', '--S', type=int, default=50, help='number of steps')
    parser.add_argument('-schedule', '--schedule', type=str, choices=['linear', 'quadratic'], help='noise level schedules')

    # generation util
    parser.add_argument('-n', '--n_generate', type=int, help='Number of samples to generate')
    parser.add_argument('-bs', '--batchsize', type=int, default=256, help='Batchsize of generation')
    parser.add_argument('-gpu', '--gpu', type=str, default='cuda', choices=['cuda']+[str(i) for i in range(16)], help='gpu device')

    args = parser.parse_args()

    global map_gpu
    map_gpu = _map_gpu(args.gpu)

    from config import model_config_map
    model_config = model_config_map[args.name]

    
    kappa = args.kappa
    if args.approxdiff == 'STD':
        variance_schedule = '1000'
        generation_param = {"kappa": kappa}

    elif args.approxdiff == 'VAR':  # user defined variance
        user_defined_eta = get_VAR_noise(args.S, args.schedule)
        generation_param = {"kappa": kappa, 
                            "user_defined_eta": user_defined_eta}
        variance_schedule = '{}{}'.format(args.S, args.schedule)

    elif args.approxdiff == 'STEP':  # user defined step
        user_defined_steps = get_STEP_step(args.S, args.schedule)
        generation_param = {"kappa": kappa, 
                            "user_defined_steps": user_defined_steps}
        variance_schedule = '{}{}'.format(args.S, args.schedule)

    else:
        raise NotImplementedError

    output_name = '{}{}_{}{}_kappa{}'.format('ema_' if args.ema else '',
                                             args.name, 
                                             args.approxdiff,
                                             variance_schedule,
                                             kappa)
    
    n_exist = 0
    if 'generated' in os.listdir() and output_name in os.listdir('generated'):
        if len(os.listdir(os.path.join('generated', output_name))) == args.n_generate:
            print('{} already finished'.format(output_name))
            n_exist = args.n_generate
        else:
            n_exist = len(os.listdir(os.path.join('generated', output_name)))

    if n_exist < args.n_generate:
        if n_exist > 0:
            print('{} already generated, resuming'.format(n_exist))
        else:
            print('start generating')
        model_path = os.path.join('checkpoints', 
                                '{}diffusion_{}_model'.format('ema_' if args.ema else '', args.name), 
                                'model.ckpt')
        generate(output_name, model_path, model_config, 
                diffusion_config, args.approxdiff, generation_param, 
                args.n_generate, args.batchsize, n_exist)
