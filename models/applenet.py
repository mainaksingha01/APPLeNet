import imp
import os.path as osp
from collections import OrderedDict
import math
import pandas as pd
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.nn.modules.loss import _Loss

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()
import os


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class AdaIN_trans(nn.Module):
		def __init__(self):
				super().__init__()

		def mu(self, x):
				""" Takes a (n,c,h,w) tensor as input and returns the average across
				it's spatial dimensions as (h,w) tensor [See eq. 5 of paper]"""
				return torch.sum(x,(1))/(x.shape[1])

		def sigma(self, x):
				""" Takes a (n,c,h,w) tensor as input and returns the standard deviation
				across it's spatial dimensions as (h,w) tensor [See eq. 6 of paper] Note
				the permutations are required for broadcasting"""
				return torch.sqrt((torch.sum((x.permute([1,0,2])-self.mu(x)).permute([1,0,2])**2,(1))+0.000000023)/(x.shape[1]))

		def forward(self, x, y):
				""" Takes a content embeding x and a style embeding y and changes
				transforms the mean and standard deviation of the content embedding to
				that of the style. [See eq. 8 of paper] Note the permutations are
				required for broadcasting"""
				return (self.sigma(y)*((x.permute([1,0,2])-self.mu(x))/self.sigma(x)) + self.mu(y)).permute([1,0,2])

# for ViT :
class multi_scale(nn.Module):
	def __init__(self):
		super(multi_scale,self).__init__()
		self.linear = nn.ModuleList(nn.Linear(768,512) for _ in range (12))
		self.adain=AdaIN_trans()
		self.gap=nn.AdaptiveAvgPool2d((1,1))
	def forward(self,data):
		data_prompt = []
		for i in range(len(data)):
			x_mu=self.adain.mu(data[i])
			x_mu = x_mu.to(torch.float32)
			x=self.linear[i](x_mu)
			data_prompt.append(x)
		data_prompt=torch.stack(data_prompt,1)  
		return data_prompt 

class projector(nn.Module):
    def __init__(self):
        super(projector,self).__init__()
        self.adain=AdaIN_trans()
    def forward(self,im_features):
        im_prompt = []
        x_mu=self.adain.mu(im_features)
        im_prompt.append(x_mu)
        im_prompt=torch.stack(im_prompt,1) 
        return im_prompt
        

class InjectionBlock(nn.Module):
    def __init__(self, vis, ctx):
        super(InjectionBlock, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(vis, vis // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(vis // 16, vis, bias=False),
            nn.Sigmoid()
        )

        self.linearlayer1 = nn.Sequential(nn.Linear((vis*2),vis))
        self.linearlayer2 = nn.Sequential(nn.Linear((vis*2),ctx))
        self.gap=nn.AdaptiveAvgPool2d((1,512))

    def forward(self, vis):  
        vis_f = self.gap(vis)
        attn1 = self.attention(vis_f.type(torch.float))
        mulattn1 = torch.mul(attn1, vis_f)
        resattn1 = torch.cat((mulattn1, vis_f),2)
        linear1 = self.linearlayer1(resattn1)

        attn2 = self.attention(linear1.type(torch.float))
        mulattn2 = torch.mul(attn2, vis_f)
        resattn2 = torch.cat((mulattn2, vis_f),2)
        linear2 = self.linearlayer2(resattn2)
        
        output = linear2.to(torch.float16)
        return output


class TextEncoder(nn.Module):
	def __init__(self, clip_model):
		super().__init__()
		self.transformer = clip_model.transformer
		self.positional_embedding = clip_model.positional_embedding
		self.ln_final = clip_model.ln_final
		self.text_projection = clip_model.text_projection
		self.dtype = clip_model.dtype

	def forward(self, prompts, tokenized_prompts):
		x = prompts + self.positional_embedding.type(self.dtype)
		x = x.to(torch.float16)
		x = x.permute(1, 0, 2)
		x,_ = self.transformer(x)
		x = x.permute(1, 0, 2)
		x = self.ln_final(x).type(self.dtype)
		x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
		return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.APPLENET.N_CTX
        ctx_init = cfg.TRAINER.APPLENET.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            if cfg.TRAINER.APPLENET.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)
        print(f'Input size of attn: "{vis_dim}"')

        self.injection = InjectionBlock(vis_dim, ctx_dim)
        self.multi = multi_scale()
        self.projector = projector()

        self.meta_net = nn.Sequential(OrderedDict([
			("linear1", nn.Linear(vis_dim, vis_dim // 16)),
			("relu", nn.ReLU(inplace=True)),
			("linear2", nn.Linear(vis_dim // 16, vis_dim))
		]))

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
    
    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]
        
        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,     # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts
    
     
    def forward(self, im_features, data):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx 
        multi = self.multi(data)
        im_features = im_features.unsqueeze(1)
        final_features = self.projector(im_features)
        fcs = torch.cat((multi,final_features),1)
        bias = self.injection(fcs) 
        bias = bias.to(torch.float16)
        alpha1 = self.meta_net(bias.type(torch.float))
        alpha2 = self.meta_net(bias.type(torch.float))
        alpha3 = self.meta_net(bias.type(torch.float))
        alpha4 = self.meta_net(bias.type(torch.float))

        alpha_a = torch.cat((alpha1, alpha2),1)
        alpha_b = torch.cat((alpha_a, alpha3),1)
        alpha = torch.cat((alpha_b, alpha4),1)

        ctx = ctx.unsqueeze(0)   
        ctx_shifted = torch.add(ctx,alpha)   
               
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)
        
        return prompts, ctx_shifted


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, label):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        image_features, data = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts, ctx_shifted = self.prompt_learner(image_features, data)
  
        logits = []
        for pts_i, imf_i in zip(prompts, image_features):
            text_features = self.text_encoder(pts_i, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            l_i = logit_scale * imf_i @ text_features.t()
            logits.append(l_i)
        logits = torch.stack(logits)

        return logits, ctx_shifted, label

class AppleLoss(_Loss):
    def __init__(self, T):
        super(AppleLoss, self).__init__()
        self.T = T

    def forward(self, logits, ctx_shifted, label):
        ce_loss = F.cross_entropy(logits, label)
        
        transprompts = ctx_shifted.permute(0,2,1)
        pmul = torch.matmul(ctx_shifted / self.T, transprompts / self.T)
        identity = torch.eye(4).cuda()
        mask = identity.repeat(4, 1, 1).cuda()     

        sim_loss = torch.linalg.det(torch.sub(pmul / self.T , mask / self.T)).mean()

        total_loss = ce_loss + 0.05*torch.log10(-sim_loss)   # lambda = 0.05 : weight

        return total_loss


@TRAINER_REGISTRY.register()
class AppleNet(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.APPLENET.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.APPLENET.PREC == "fp32" or cfg.TRAINER.APPLENET.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"
        
        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                param.requires_grad_(False)
        
        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.APPLENET.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        print(f'gpu detected : {device_count}')
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

        if cfg.LOSS.NAME == "apple":
            self.criterion = AppleLoss(T=cfg.LOSS.T)
        else:
            raise NotImplementedError

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        cfg = self.cfg
        model = self.model
        optim = self.optim
        scaler = self.scaler
        
        prec = self.cfg.TRAINER.APPLENET.PREC
        if prec == "amp":
            with autocast():
                loss,ctx_shifted,_ = model(image, label)
                total_loss = self.criterion(loss,ctx_shifted,label)
            optim.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss,ctx_shifted,_ = model(image, label)
            optim.zero_grad()
            total_loss = self.criterion(loss,ctx_shifted, label)
            total_loss.sum().backward()
            optim.step()

        loss_summary = {"loss": total_loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = ((self.epoch + 1) %
                                self.cfg.TRAIN.CHECKPOINT_FREQ == 0 if
                                self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False)

        if do_test:
            curr_result = self.test()
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.save_model(self.epoch,
                                self.output_dir,
                                model_name="model-best.pth.tar")

            self.set_model_mode("train")

        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)

    @torch.no_grad()

    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            logits, ctx_shifted, label = self.model_inference(input, label)
            self.evaluator.process(logits, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]
