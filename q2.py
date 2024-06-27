import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
import networkx as nx

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Define a prompt-based system to convert text to graph
class PromptSystem(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(model.config.hidden_size, 128)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.linear(x))
        x = self.dropout(x)
        x = self.classifier(x)
        return x

prompt_system = PromptSystem()

# Load a dataset of text samples
with open('text_data.txt', 'r') as f:
    text_samples = [line.strip() for line in f.readlines()]

# Convert text samples to input IDs and attention masks
input_ids = []
attention_masks = []
for text in text_samples:
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids.append(inputs['input_ids'].flatten())
    attention_masks.append(inputs['attention_mask'].flatten())

# Create a dataset class for our data
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_masks):
        self.input_ids = input_ids
        self.attention_masks = attention_masks

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids[idx]),
            'attention_mask': torch.tensor(self.attention_masks[idx])
        }

    def __len__(self):
        return len(self.input_ids)

dataset = TextDataset(input_ids, attention_masks)

# Create a data loader for our dataset
data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Pre-train the prompt-based system
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
prompt_system.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(prompt_system.parameters(), lr=1e-5)
for epoch in range(5):
    prompt_system.train()
    total_loss = 0
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        optimizer.zero_grad()
        outputs = prompt_system(input_ids)
        loss = criterion(outputs, torch.zeros_like(outputs))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {total_loss / len(data_loader)}')

# Tune the prompt-based system on a validation set
val_dataset = TextDataset(val_input_ids, val_attention_masks)
val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
prompt_system.eval()
with torch.no_grad():
    total_val_loss = 0
    for batch in val_data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = prompt_system(input_ids)
        loss = criterion(outputs, torch.zeros_like(outputs))
        total_val_loss += loss.item()
    print(f'Validation Loss: {total_val_loss / len(val_data_loader)}')

# Use the trained prompt-based system to convert text to graph
def convert_text_to_graph(text):
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = inputs['input_ids'].flatten()
    attention_mask = inputs['attention_mask'].flatten()
    output = prompt_system(torch.tensor(input_ids).unsqueeze(0))
    graph_edges = []
    graph_nodes = []
    for edge in output[0]:
        if edge > 0.5:
            graph_edges.append((edge[0].item(), edge[1].item()))
            graph_nodes.extend([edge[0].item(), edge[1].item()])
    graph_nodes = list(set(graph_nodes))
    graph_edges.sort(key=lambda x: x[1], reverse=True)
    G = nx.Graph()
    for node in graph_nodes:
        G.add_node(node)
    for edge in graph_edges:
        G.add_edge(edge[0], edge[1])
    return G

# Test the conversion function
text_samples = ['This is a test sentence.', 'Another test sentence.']
graphs = []
for text in text_samples:
    graph = convert_text_to_graph(text)
    graphs.append(graph)

# Visualize the graphs using NetworkX
import matplotlib.pyplot as plt
for i, G in enumerate(graphs):
    plt.figure(figsize=(8, 8))
    nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray')
    plt.title(f'Graph {i+1}')
    plt.show()
