import urllib.request
import zipfile
import os
from pathlib import Path
import pandas as pd
import tiktoken
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from gpt_download import download_and_load_gpt2
from chapter05 import GPTModel
from chapter05_5 import load_weights_into_gpt
from chapter04 import generate_text_simple
from chapter05 import text_to_token_ids, token_ids_to_text

url = "http://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return

    with urllib.request.urlopen(url) as response:
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File downloaded and saved as {data_file_path}")

def create_balanced_dataset(df):
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])
    return balanced_df

def random_split(df, train_frac, validation_frac):
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df

class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)

        self.encoded_texts = [
                tokenizer.encode(text) for text in self.data["Text"]
        ]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length

            self.encoded_texts = [
                    encoded_text[:self.max_length]
                    for encoded_text in self.encoded_texts
            ]

        self.encoded_texts = [
                encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
                for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
                torch.tensor(encoded, dtype=torch.long),
                torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length

def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)

        with torch.no_grad():
            logits = model(input_batch)[:, -1, :]
        predicted_labels = torch.argmax(logits, dim=-1)

        num_examples += predicted_labels.shape[0]
        correct_predictions += (predicted_labels == target_batch).sum().item()
    return correct_predictions / num_examples

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :] # Logits of last output token
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()
    return total_loss / num_batches


def main():
    # Downloading file
    download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)

    # Loading it into panda dataframes
    df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
    print(df["Label"].value_counts())

    # Get balanced dataset for finetuning
    balanced_df = create_balanced_dataset(df)
    print("Balanced dataset value counts:\n", balanced_df["Label"].value_counts())

    # Convert string class labels to 0 and 1
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

    # 70% for training, 10% for validation adn 20% for testing (common proportions in ML)
    train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)

    # Save to csv to re-use later
    train_df.to_csv("train.csv", index=None)
    validation_df.to_csv("validation.csv", index=None)
    test_df.to_csv("test.csv", index=None)

    # Verifying which is <|endoftext|> token for padding
    tokenizer = tiktoken.get_encoding("gpt2")
    print("<|endoftext|> token id is ", tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))

    # Creating data sets
    train_dataset = SpamDataset(
            csv_file="train.csv",
            max_length=None,
            tokenizer = tokenizer
    )
    print("Longest sequence in the dataset is ", train_dataset.max_length)
    val_dataset = SpamDataset(
            csv_file="validation.csv",
            max_length=train_dataset.max_length,
            tokenizer=tokenizer
    )
    test_dataset = SpamDataset(
            csv_file="test.csv",
            max_length=train_dataset.max_length,
            tokenizer=tokenizer
    )

    # Creating dataloaders
    num_workers = 0
    batch_size = 8
    torch.manual_seed(123)

    train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
    )
    val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=False,
    )
    test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=False,
    )

    # Ensure data loaders are working correctly, iterate over the training loader and print tensor dimensions of the last batch
    for input_batch, target_batch in train_loader:
        pass
    print("Input batch dimensions:", input_batch.shape)
    print("Label batch dimensions:", target_batch.shape)

    print(f"{len(train_loader)} training batches")
    print(f"{len(val_loader)} validation batches")
    print(f"{len(test_loader)} test batches")

    # 6.4 Initialize a model with pretrained weights
    CHOOSE_MODEL = "gpt2-small (124M)"
    INPUT_PROMPT = "Every effort moves"
    BASE_CONFIG = {
            "vocab_size": 50257,     # Vocabulary size
            "context_length": 1024,  # Context length
            "drop_rate": 0.0,        # Dropout rate
            "qkv_bias": True         # Query-key-value bias
    }
    model_configs = {
            "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
            "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
            "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
            "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    assert train_dataset.max_length <= BASE_CONFIG["context_length"], (
            f"Dataset length {train_dataset.max_length} exceeds model's context "
            f"length {BASE_CONFIG['context_length']}. Reinitialize data sets with "
            f"`max_length={BASE_CONFIG['context_length']}`"
    )

    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")
    
    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval()
    
    # Model is loaded, lets ensure it generate coherent text
    text_1 = "Every effort moves you"
    token_ids = generate_text_simple(
            model=model,
            idx=text_to_token_ids(text_1, tokenizer),
            max_new_tokens=15,
            context_size=BASE_CONFIG["context_length"]
    )
    print(token_ids_to_text(token_ids, tokenizer))

    # Lets see if the model can classify spam messages by prompting it with instructions
    text_2 = (
            "Is the following text 'spam'? Answer with 'yes' or 'no':"
            " 'You are a winner you have been specially"
            " selected to receive $1000 cash or a $2000 award.'"
    )
    token_ids = generate_text_simple(
            model=model,
            idx=text_to_token_ids(text_2, tokenizer),
            max_new_tokens=23,
            context_size=BASE_CONFIG["context_length"]
    )
    print(token_ids_to_text(token_ids, tokenizer))

    # 6.5 Onto modifying the pre-trained model
    # Let's first print the current state
    print(model)

    # Freeze the model, make the layers non-trainable
    for param in model.parameters():
        param.requires_grad = False

    # Replace output head
    torch.manual_seed(123)
    num_classes = 2
    model.out_head = torch.nn.Linear(
            in_features=BASE_CONFIG["emb_dim"],
            out_features=num_classes
    )
    # Make the last LayerNorm and Transformer block trainable
    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True
    for param in model.final_norm.parameters():
        param.requires_grad = True

    # TODO(Ilya): Do exercise 6.2 in finetuning the whole model and assess the impact on predictive performance
    inputs = tokenizer.encode("Do you have time")
    inputs = torch.tensor(inputs).unsqueeze(0)
    print("Inputs:", inputs)
    print("Inputs dimensions:", inputs.shape) # shape: (batch_size, num_tokens)
    with torch.no_grad():
        outputs = model(inputs)
    print("Outputs:\n", outputs)
    print("Outputs dimensions:", outputs.shape) # shape: (batch_size, num_tokens, num_classes)
    print("Last output token:", outputs[:, -1, :])

    # TODO(Ilya): Do exercise 6.3: try finetuning the first output token instead of the last one and observe the changes 
    # in predictive performance

    # Obtaining the class label for that example
    logits = outputs[:, -1, :]
    label = torch.argmax(logits)
    print("Class label:", label.item())

    # Test out classification accuracies
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    torch.manual_seed(123)
    train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=10)
    val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=10)
    test_accuracy = calc_accuracy_loader(test_loader, model, device, num_batches=10)

    print(f"Training accuracy: {train_accuracy*100:.2f}%")
    print(f"Validation accuracy: {val_accuracy*100:.2f}%")
    print(f"Test accuracy: {test_accuracy*100:.2f}%")

    # Computing initial loss for each data set
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
        test_loss = calc_loss_loader(test_loader, model, device, num_batches=5)
    print("The initial loss values are as follows:\n")
    print(f"Training loss: {train_loss:.3f}")
    print(f"Validation loss: {val_loss:.3f}")
    print(f"Test loss: {test_loss:.3f}")





if __name__ == "__main__":
    main()


