# src/othermodels/deepseek.py

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class DeepSeekModel(torch.nn.Module):
    def __init__(self, model_name='DeepSeekModel', pretrained=True):
        super(DeepSeekModel, self).__init__()

        # Load the pre-trained DeepSeek model and tokenizer from Hugging Face
        # You can replace 'DeepSeekModel' with the actual model name on Hugging Face if it's public
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load the model
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def forward(self, inputs):
        # Tokenize the inputs
        inputs = self.tokenizer(inputs, return_tensors='pt', padding=True, truncation=True)

        # Forward pass through the model
        outputs = self.model(**inputs)

        return outputs.logits  # Return the logits for classification

    def predict(self, inputs):
        # Tokenize and predict
        with torch.no_grad():
            logits = self.forward(inputs)
            predictions = torch.argmax(logits, dim=-1)
        return predictions
