import torch
import argparse

from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from model import LoRAVAE
from dataset import YelpDatabase


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_dim",type=int,default=768)
    parser.add_argument("--rank",type=int,default=16,help="the rank of lora")
    parser.add_argument("--seed",type=int,default=3407)
    parser.add_argument("--z_dim",type=int,default=32,help="Latent space dimension.")
    parser.add_argument("--latent_mode",type=str,default="embedding")
    parser.add_argument("--dim_target_kl",type=float,default=3.0)
    parser.add_argument("--path",type=str,default="C:\LoRAVAE\yelp_data")
    parser.add_argument("--ckpt",type=str,default="facebook/bart-base")
    parser.add_argument("--max_len",type=int,default=512)
    parser.add_argument("--train_batch_size",type=int,default=2)
    parser.add_argument("--test_batch_size",type=int,default=1)
    parser.add_argument("--train",type=bool,default=True)
    parser.add_argument("--num_warmup_steps",type=int,default=10)
    parser.add_argument("--lr",type=float,default=5e-5)
    parser.add_argument("--weight_decay",type=float,default=1e-2)
    parser.add_argument("--epochs",type=int,default=20)
    parser.add_argument("--max_grad_norm",type=float,default=1.0)
    parser.add_argument("--length_weighted_loss",type=bool,default=True)
    parser.add_argument("--beta",type=float,default=1.0)

    args = parser.parse_args()
    return args


def train(args):
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')


    train_dataset = YelpDatabase(args)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, collate_fn=train_dataset.collate_fn,shuffle=True)


    args.train = False
    test_dataset = YelpDatabase(args)
    test_dataloader = DataLoader(test_dataset,batch_size=args.test_batch_size,collate_fn=test_dataset.collate_fn,shuffle=False)


    model = LoRAVAE(args).to(device)

    num_warmup_steps = args.num_warmup_steps
    epochs = args.epochs
    max_grad_norm = args.max_grad_norm
    lr = args.lr
    weight_decay = args.weight_decay
    total_steps = len(train_dataset) * epochs

    optimizer = torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=num_warmup_steps,num_training_steps=total_steps)

    for epoch in tqdm(range(epochs)):
        recon_losses = 0
        kl_losses = 0
        losses = 0
        cnt = 0

        for input_ids, attention_mask in train_dataloader:
            cnt += 1
            model.train()

            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

            reconstruction_loss, kl_loss, loss = model(input_ids,attention_mask,input_ids)

            reconstruction_loss = reconstruction_loss.mean()
            loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),max_norm=max_grad_norm)
            optimizer.step()
            scheduler.step()
            losses += loss.item()
            kl_losses += kl_loss.item()
            recon_losses += reconstruction_loss.item()

        print("loss = {0}, kl_loss = {1}, recon_loss = {2}".format(losses / cnt, kl_losses / cnt, recon_losses / cnt))

    print("train finished!")


if __name__ == '__main__':
    args = parser_args()

    print("args:",args)

    torch.manual_seed(args.seed)

    train(args)
