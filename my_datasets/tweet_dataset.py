from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer
import linecache

tweet_tokenizer = RobertaTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
tweet_tokenizer_fn = lambda text: tweet_tokenizer(text, return_tensors='pt')['input_ids']

# Dataset class for tweet
class TweetDataset(Dataset):
  def __init__(self, text_path, labels_path):
    self.text_path = text_path
    self.labels_path = labels_path
    self.num_data = 0
    self.tokenizer = RobertaTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    with open(text_path, "r") as f:
      self.num_data = len(f.readlines())

    with open(labels_path, "r") as f:
      self.labels = [int(l.strip()) for l in f]

  def __len__(self):
    return self.num_data

  # Tokenize the tweet
  def tokenize_tweet(self, text):
    new_text = []
    for t in text.split(" "):
      t = '@user' if t.startswith('@') and len(t) > 1 else t
      t = 'http' if t.startswith('http') else t
      new_text.append(t)
    new_text = " ".join(new_text)
    # Feed these arguments so that we may use batched things
    tokens = tweet_tokenizer_fn(new_text)
    # tokens = tweet_tokenizer(new_text, return_tensors='pt')['input_ids']
    return tokens[0]

  def __getitem__(self, idx):
    text = linecache.getline(self.text_path, idx+1)
    tokens = self.tokenize_tweet(text)
    label = self.labels[idx]
    return tokens, label

