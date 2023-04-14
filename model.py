
import loralib as lora
from transformers import BartForSequenceClassification,BartConfig,GPT2Model
from bart import BartModel
import torch
import torch.nn as nn


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class LoRAVAE(nn.Module):
    def __init__(self,args):
        super(LoRAVAE, self).__init__()
        self.config = BartConfig.from_pretrained("facebook/bart-base")
        self.bart = BartModel.from_pretrained("facebook/bart-base",args)
        self.encoder = self.bart.encoder
        self.decoder = self.bart.decoder
        self.args = args
        self.linear = lora.Linear(self.args.hidden_dim,2 * self.args.z_dim,bias=False)

        self.logits = nn.Linear(args.hidden_dim, self.config.vocab_size, bias=False)


    def reparameterize(self, mu, logvar, nsamples=1):
        batch_size, nz = mu.size()
        std = logvar.mul(0.5).exp()

        mu_expd = mu.unsqueeze(1).expand(batch_size, nsamples, nz)
        std_expd = std.unsqueeze(1).expand(batch_size, nsamples, nz)

        eps = torch.zeros_like(std_expd).normal_()

        return mu_expd + torch.mul(eps, std_expd)


    def forward(self,input_ids, attention_mask,labels):

        reconstrution_mask = (labels != self.config.pad_token_id).float() # 50257 is the padding token for GPT2
        sent_length = torch.sum(reconstrution_mask, dim=1)
        enc_output = self.encoder(input_ids,attention_mask)
        enc_hidden_state = enc_output[0]

        eos_mask = input_ids.eq(self.config.eos_token_id).to(enc_hidden_state.device)

        pool_hidden_state = enc_hidden_state[eos_mask, :].view(enc_hidden_state.size(0), -1, enc_hidden_state.size(-1))[
            :, -1, :
        ]

        mu, log_var = self.linear(pool_hidden_state).chunk(2,-1)


        z_latent = self.reparameterize(mu,log_var)

        kl_loss = 0.5 * (mu.pow(2) + log_var.exp() - log_var - 1)

        # kl_mask = (kl_loss > self.args.dim_target_kl).float()

        # kl_loss = (kl_mask * kl_loss).sum(dim=1)

        decoder_input_ids = shift_tokens_right(
            input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
        )

        dec_output = self.decoder(decoder_input_ids,attention_mask=attention_mask,past_key_values=z_latent)
        dec_output = dec_output['last_hidden_state']

        logits = self.logits(dec_output)

        loss_fct = nn.CrossEntropyLoss()

        reconstruction_loss = loss_fct(logits.view(-1, logits.size(-1)),input_ids.contiguous().view(-1))

        if self.args.length_weighted_loss:
            loss = reconstruction_loss / sent_length + self.args.beta * kl_loss
        else:
            loss = reconstruction_loss + self.args.beta * kl_loss

        return reconstruction_loss, kl_loss, loss



