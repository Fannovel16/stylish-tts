import math
import random
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional, Tuple
from transformers import Qwen3Model, Qwen3Config


# ---------------------------------------------------------------------------
# Dataclasses & Config
# ---------------------------------------------------------------------------


@dataclass
class OmniVoiceGenerationConfig:
    num_step: int = 1
    guidance_scale: float = 2.0
    t_shift: float = 0.1
    layer_penalty_factor: float = 0.0
    position_temperature: float = 5.0
    class_temperature: float = 0.0


@dataclass
class GenerationTask:
    batch_size: int
    texts: List[str]  # kept as strings, matching OmniVoice
    target_lens: List[int]
    text_ids: List[torch.Tensor]  # pre-tokenised 1D tensors


# ---------------------------------------------------------------------------
# Core Model
# ---------------------------------------------------------------------------


class TokenPredictor(nn.Module):
    def __init__(
        self,
        text_vocab: int,
        audio_vocab: int,
        hidden_dim: int = 512,
        hidden_layers: int = 8,
        attn_heads: int = 8,
        kv_heads: int = 4,
    ):
        super().__init__()
        self.text_vocab = text_vocab
        self.audio_vocab = audio_vocab
        self.audio_mask_id = audio_vocab
        self.text_mask_id = text_vocab

        self.pre_align = Qwen3Model(
            Qwen3Config(
                vocab_size=0,
                hidden_size=hidden_dim,
                intermediate_size=hidden_dim * 2,
                num_hidden_layers=hidden_layers // 2,
                num_attention_heads=attn_heads,
                num_key_value_heads=kv_heads,
                head_dim=128,
                max_window_layers=0,
                use_cache=False,
                use_sliding_window=False,
                max_position_embeddings=60 * 50,
            )
        )
        self.post_align = Qwen3Model(
            Qwen3Config(
                vocab_size=0,
                hidden_size=hidden_dim,
                intermediate_size=hidden_dim * 2,
                num_hidden_layers=hidden_layers // 2,
                num_attention_heads=attn_heads,
                num_key_value_heads=kv_heads,
                head_dim=128,
                max_window_layers=0,
                use_cache=False,
                use_sliding_window=False,
                max_position_embeddings=60 * 50,
            )
        )

        self.embed_text = nn.Embedding(text_vocab + 1, hidden_dim)
        self.embed_pitch = nn.Linear(1, hidden_dim)
        self.downsample = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=2, stride=2)
        self.audio_head = nn.Linear(hidden_dim, audio_vocab, bias=False)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, text_ids, audio_ids, pitch, alignment):
        text_embeds = self.embed_text(text_ids)
        pitch_embeds = self.embed_pitch(pitch.unsqueeze(-1))
        B, T_text = text_ids.shape
        _, _, T_audio = alignment.shape

        hidden_state = self.pre_align(
            inputs_embeds=text_embeds,
            attention_mask=torch.ones(
                B, 1, T_text, T_text, dtype=torch.bool, device=self.device
            ),
        ).last_hidden_state
        hidden_state = (hidden_state.mT @ alignment).mT + pitch_embeds
        hidden_state = self.downsample(hidden_state.mT).mT
        hidden_state = self.post_align(
            inputs_embeds=hidden_state,
            attention_mask=torch.ones(
                B, 1, T_audio // 2, T_audio // 2, dtype=torch.bool, device=self.device
            ),
        ).last_hidden_state
        return self.audio_head(hidden_state)


class MaskedTokenPredictor(nn.Module):
    """
    Single-codebook OmniVoice NAR predictor with pitch conditioning and CFG.
    """

    def __init__(
        self,
        text_vocab: int,
        audio_vocab: int,
        hidden_dim: int = 512,
        hidden_layers: int = 8,
        attn_heads: int = 16,
        kv_heads: int = 4,
    ):
        super().__init__()
        self.text_vocab = text_vocab
        self.audio_vocab = audio_vocab
        self.audio_mask_id = audio_vocab
        self.text_mask_id = text_vocab
        self.drop_cond_ratio: float = 0.1
        self.prompt_ratio_range: Tuple[float, float] = (0.0, 0.3)
        self.mask_ratio_range: Tuple[float, float] = (0.0, 1.0)

        self.llm = Qwen3Model(
            Qwen3Config(
                vocab_size=text_vocab + audio_vocab + 1,  # text | audio | mask
                hidden_size=hidden_dim,
                intermediate_size=hidden_dim * 2,
                num_hidden_layers=hidden_layers,
                num_attention_heads=attn_heads,
                num_key_value_heads=kv_heads,
                head_dim=128,
                max_window_layers=0,
                use_cache=False,
                use_sliding_window=False,
                max_position_embeddings=2048,
            )
        )
        del self.llm.embed_tokens

        self.embed_text = nn.Embedding(text_vocab + 1, hidden_dim // 4)
        self.embed_audio = nn.Embedding(audio_vocab + 1, hidden_dim // 2)
        self.embed_pitch = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4),
        )
        self.audio_head = nn.Linear(
            hidden_dim, audio_vocab, bias=False
        )  # mask token never predicted

    @property
    def device(self):
        return next(self.parameters()).device

    def _prepare_embed_inputs(
        self,
        text_ids: torch.Tensor,  # [B, N_text]
        audio_ids: torch.Tensor,  # [B, N_audio]
        pitch: torch.Tensor,  # [B, N_audio]
    ) -> torch.Tensor:  # [B, N_text + N_audio, hidden_dim]
        # shared_embed = self.llm.get_input_embeddings()
        # text_embeds = shared_embed(text_ids)  # [B, N_text,  H]
        text_embeds = self.embed_text(text_ids)

        # audio_embeds = shared_embed(audio_ids + self.text_vocab)  # [B, N_audio, H]
        audio_embeds = self.embed_audio(audio_ids)
        pitch_embeds = self.embed_pitch(pitch.unsqueeze(-1))
        # audio_embeds = audio_embeds + self.embed_pitch(
        #     pitch.unsqueeze(-1)
        # )  # superimpose pitch
        return torch.cat(
            [text_embeds, pitch_embeds, audio_embeds],
            dim=2,  # dim=1
        )  # [B, N_text + N_audio, H]

    def make_noisy_sample(
        self,
        text_ids: torch.Tensor,  # [B, N_text]
        audio_ids: torch.Tensor,  # [B, N_audio]
        pitch: torch.Tensor,  # [B, N_audio]
    ):
        drop_cond = random.uniform(0, 1) < self.drop_cond_ratio
        mask_ratio = torch.tensor(
            [random.uniform(*self.mask_ratio_range) for _ in range(text_ids.shape[0])],
            device=self.device,
        ).unsqueeze(-1)

        prompt_ratio = 0.0
        # if drop_cond:
        #     prompt_ratio = 0.0
        # else:
        #     prompt_ratio = random.uniform(*self.prompt_ratio_range)

        prompt_length = int(audio_ids.shape[1] * prompt_ratio)
        input_audio_ids = audio_ids.clone()
        target_audio_ids = audio_ids.clone()

        maskable_region = audio_ids[:, prompt_length:]
        token_mask = torch.rand(maskable_region.shape, device=self.device) < mask_ratio
        # Prevent CE NaN loss if mask_ratio = 0
        if not token_mask.any():
            token_mask[:, random.randint(0, token_mask.shape[1] - 1)] = True

        input_audio_ids[:, prompt_length:][token_mask] = self.audio_mask_id
        target_audio_ids[:, prompt_length:][
            ~token_mask
        ] = -100  # Only compute loss on masked tokens

        if drop_cond:
            # input_text_ids = torch.empty(
            #     (text_ids.shape[0], 0), dtype=text_ids.dtype, device=text_ids.device
            # )
            # target_text_ids = torch.empty(
            #     (text_ids.shape[0], 0), dtype=text_ids.dtype, device=text_ids.device
            # )
            input_text_ids = torch.full_like(text_ids, self.text_mask_id)
            target_text_ids = torch.full_like(text_ids, -100)
            input_pitch = torch.zeros_like(pitch)
        else:
            input_text_ids = text_ids.clone()
            target_text_ids = torch.full_like(text_ids, -100)
            input_pitch = pitch.clone()
            # No loss on prompt region
            target_audio_ids[:, :prompt_length] = -100

        return (
            input_text_ids,
            input_audio_ids,
            input_pitch,
            target_text_ids,
            target_audio_ids,
        )

    def forward(
        self,
        text_ids: torch.Tensor,  # [B, N_text]
        audio_ids: torch.Tensor,  # [B, N_audio]
        pitch: torch.Tensor,  # [B, N_audio]
    ) -> torch.Tensor:  # [B, N_text + N_audio, audio_vocab]
        inputs_embeds = self._prepare_embed_inputs(text_ids, audio_ids, pitch)
        B, N_total, _ = inputs_embeds.shape
        # Bidirectional mask
        attention_mask = torch.ones(
            B, 1, N_total, N_total, dtype=torch.bool, device=self.device
        )
        hidden_states = self.llm(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask
        ).last_hidden_state
        return self.audio_head(hidden_states)

    @torch.no_grad()
    def generate_iterative(
        self,
        text_ids: torch.Tensor,  # [B, N_text]
        pitch: torch.Tensor,  # [B, target_len]
        gen_config: OmniVoiceGenerationConfig = OmniVoiceGenerationConfig(),
    ) -> torch.Tensor:  # [B, target_len]
        B = text_ids.shape[0]
        target_len = pitch.shape[1]
        N_text = text_ids.shape[1]

        audio_ids = torch.full(
            (B, target_len), self.audio_mask_id, dtype=torch.long, device=self.device
        )
        # u_text_ids = torch.empty((B, 0), dtype=text_ids.dtype, device=text_ids.device)
        u_text_ids = torch.full_like(text_ids, self.text_mask_id)
        u_pitch = torch.zeros_like(pitch)

        timesteps = _get_time_steps(
            t_start=0.0,
            t_end=1.0,
            num_step=gen_config.num_step + 1,
            t_shift=gen_config.t_shift,
            device=self.device,
        ).tolist()

        rem = target_len
        schedule = []
        for step in range(gen_config.num_step):
            num = (
                rem
                if step == gen_config.num_step - 1
                else min(
                    math.ceil(target_len * (timesteps[step + 1] - timesteps[step])),
                    rem,
                )
            )
            schedule.append(int(num))
            rem -= int(num)

        # --- Iterative unmasking loop ---
        for step in range(gen_config.num_step):
            k = schedule[step]
            if k <= 0:
                continue

            c_logits_full = self(
                text_ids, audio_ids, pitch
            )  # [B, N_text + target_len, audio_vocab]
            u_logits_full = self(
                u_text_ids, audio_ids, u_pitch
            )  # [B,          target_len, audio_vocab]

            # Extract audio-region logits for each half
            # c_logits = c_logits_full[:, N_text:, :]  # [B, target_len, audio_vocab]
            c_logits = c_logits_full[:, 0:, :]
            u_logits = u_logits_full[:, 0:, :]  # [B, target_len, audio_vocab]

            pred_tokens, scores = self._predict_tokens_with_scoring(
                c_logits, u_logits, gen_config
            )

            if gen_config.position_temperature > 0.0:
                scores = _gumbel_sample(scores, gen_config.position_temperature)

            # Prevent already-unmasked positions from being selected again
            scores.masked_fill_(audio_ids != self.audio_mask_id, -float("inf"))

            # Unmask top-k positions per item in the batch
            for i in range(B):
                _, topk_idx = torch.topk(scores[i], k)
                audio_ids[i][topk_idx] = pred_tokens[i][topk_idx]

        return audio_ids  # [B, target_len]

    def _predict_tokens_with_scoring(
        self,
        c_logits: torch.Tensor,  # [B, T, audio_vocab]  conditional
        u_logits: torch.Tensor,  # [B, T, audio_vocab]  unconditional
        gen_config: OmniVoiceGenerationConfig,
    ):  # → pred_tokens [B, T], scores [B, T]
        # CFG combination — identical formula to OmniVoice
        if gen_config.guidance_scale != 0:
            c_log_probs = F.log_softmax(c_logits, dim=-1)
            u_log_probs = F.log_softmax(u_logits, dim=-1)
            log_probs = F.log_softmax(
                c_log_probs + gen_config.guidance_scale * (c_log_probs - u_log_probs),
                dim=-1,
            )
        else:
            log_probs = F.log_softmax(c_logits, dim=-1)

        if gen_config.class_temperature > 0.0:
            filtered = _filter_top_k(log_probs, ratio=0.1)
            pred_tokens = _gumbel_sample(filtered, gen_config.class_temperature).argmax(
                dim=-1
            )
        else:
            pred_tokens = log_probs.argmax(dim=-1)

        confidence_scores = log_probs.max(dim=-1)[0]
        return pred_tokens, confidence_scores


def _filter_top_k(logits: torch.Tensor, ratio: float = 0.1) -> torch.Tensor:
    k = math.ceil(ratio * logits.shape[-1])
    val, ind = logits.topk(k, dim=-1)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(-1, ind, val)
    return probs


def _gumbel_sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    scaled_logits = logits / temperature
    u = torch.rand_like(scaled_logits)
    gumbel_noise = -torch.log(-torch.log(u + 1e-10) + 1e-10)
    return scaled_logits + gumbel_noise


def _get_time_steps(
    t_start: float = 0.0,
    t_end: float = 1.0,
    num_step: int = 10,
    t_shift: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    timesteps = torch.linspace(t_start, t_end, num_step + 1).to(device)
    timesteps = t_shift * timesteps / (1 + (t_shift - 1) * timesteps)
    return timesteps


if __name__ == "__main__":
    model = MaskedTokenPredictor(500, 500).cuda()
    text_ids, audio_ids, pitch = (
        torch.arange(500).repeat(2, 1).cuda(),  # torch.arange(125).repeat(2, 1).cuda(),
        torch.arange(500).repeat(2, 1).cuda(),
        torch.rand(1, 500).repeat(2, 1).cuda(),
    )
    input_text_ids, input_audio_ids, input_pitch, target_text_ids, target_audio_ids = (
        model.make_noisy_sample(text_ids, audio_ids, pitch)
    )
    ce_loss = F.cross_entropy(
        rearrange(
            model(input_text_ids, input_audio_ids, input_pitch),
            "b t c -> (b t) c",
        ),
        # rearrange(torch.cat([target_text_ids, target_audio_ids], 1), "b t -> (b t)")
        rearrange(target_audio_ids, "b t -> (b t)"),
    )
    print(ce_loss)
    print(model.generate_iterative(text_ids, pitch).shape)
