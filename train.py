import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig, AdamW, get_linear_schedule_with_warmup
from model import BertFRIDE
from models_extra import XLNetFRIDE, RoBERTaFRIDE, GPT2FRIDE
from utils import preprocess_data, CustomReviewDataset
from tqdm import tqdm

def get_model(model_type, num_labels, device):
    if model_type.lower() == "bert":
        from transformers import BertModel
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        config = BertConfig.from_pretrained('bert-base-uncased', num_labels=num_labels)
        model = BertFRIDE.from_pretrained('bert-base-uncased', config=config, num_labels=num_labels)
    elif model_type.lower() == "xlnet":
        from transformers import XLNetTokenizer, XLNetConfig
        tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        config = XLNetConfig.from_pretrained('xlnet-base-cased', num_labels=num_labels)
        model = XLNetFRIDE.from_pretrained('xlnet-base-cased', config=config, num_labels=num_labels)
    elif model_type.lower() == "roberta":
        from transformers import RobertaTokenizer, RobertaConfig
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        config = RobertaConfig.from_pretrained('roberta-base', num_labels=num_labels)
        model = RoBERTaFRIDE.from_pretrained('roberta-base', config=config, num_labels=num_labels)
    elif model_type.lower() == "gpt2":
        from transformers import GPT2Tokenizer, GPT2Config
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        # GPT-2 does not have padding token by default, so set it
        tokenizer.pad_token = tokenizer.eos_token
        config = GPT2Config.from_pretrained('gpt2', num_labels=num_labels)
        model = GPT2FRIDE.from_pretrained('gpt2', config=config, num_labels=num_labels)
    else:
        raise ValueError("Unsupported model type. Choose from bert, xlnet, roberta, or gpt2.")
    
    model.to(device)
    return model, tokenizer

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = get_model(args.model_type, args.num_labels, device)

    # Load and preprocess data
    df = pd.read_csv(args.data_path)
    texts, labels = preprocess_data(df)
    dataset = CustomReviewDataset(texts, labels, tokenizer, max_len=args.max_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)

    model.train()
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        epoch_loss = 0.0
        for batch in tqdm(dataloader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
        print(f"Average Loss: {epoch_loss/len(dataloader):.4f}")
    
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Training complete. Model saved to:", args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/sample_reviews.csv", help="Path to CSV data file")
    parser.add_argument("--output_dir", type=str, default="saved_model", help="Directory to save the model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_len", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--num_labels", type=int, default=4, help="Number of target design insight labels")
    parser.add_argument("--model_type", type=str, default="bert", help="Model type: bert, xlnet, roberta, or gpt2")
    args = parser.parse_args()
    train(args)
