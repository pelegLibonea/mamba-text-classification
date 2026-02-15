import numpy as np
import evaluate
from sklearn.metrics import f1_score, accuracy_score, classification_report

# Load the "accuracy" module from the evaluate library.
accuracy = evaluate.load("accuracy")

# Create a preprocessing function to encode text and truncate strings longer than the maximum input token length.
def preprocess_function(tokenizer, examples):
    samples = tokenizer(examples["text"], truncation=True)
    samples.pop('attention_mask')
    return samples

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Get the index of the class with the highest probability in predictions.
    predictions = np.argmax(predictions, axis=1)
    
    # Use the "accuracy" module to compute accuracy based on predictions and labels.
    acc = accuracy.compute(predictions=predictions, references=labels)
    
    # Compute F1 score for multi-class support (similar to reference code)
    # Use macro averaging for balanced F1 across all classes
    f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
    f1_weighted = f1_score(labels, predictions, average='weighted', zero_division=0)
    
    return {
        "accuracy": acc["accuracy"],
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }