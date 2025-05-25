import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from torch.nn.functional import softmax
import argparse # Import argparse for command-line argument parsing

# Load model and tokenizer
model_path = '5CD-AI/Vietnamese-Sentiment-visobert'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Set device
# Check if CUDA (GPU support) is available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Move the model to the selected device
model.to(device)
# Set the model to evaluation mode (disables dropout, batch normalization updates, etc.)
model.eval()

def predict_sentiment_batch(texts, batch_size=32):
    """
    Predict sentiment for a list of texts using batch processing.
    
    Args:
        texts (list): A list of text strings for sentiment analysis.
        batch_size (int): The number of texts to process in a single batch.
    
    Returns:
        list: A list of dictionaries. Each dictionary contains:
              'label' (str): The predicted sentiment label ('POSITIVE', 'NEGATIVE', 'NEUTRAL').
              'score' (float): The confidence score for the predicted label.
    """
    results = []
    
    # Process texts in batches for efficiency
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize the current batch of texts
        # padding=True: Pads sequences to the length of the longest sequence in the batch.
        # truncation=True: Truncates sequences that are longer than max_length.
        # max_length=256: Sets the maximum sequence length for tokenization.
        # return_tensors='pt': Returns PyTorch tensors.
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors='pt'
        )
        
        # Move tokenized inputs to the selected computing device (GPU or CPU)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Perform inference without calculating gradients
        with torch.no_grad():
            outputs = model(**inputs)
            # Apply softmax to the output logits to obtain probability distributions
            predictions = softmax(outputs.logits, dim=-1)
        
        # Process predictions for each text in the current batch
        for pred_tensor in predictions:
            pred_cpu = pred_tensor.cpu().numpy()  # Move tensor to CPU and convert to NumPy array
            predicted_class_idx = np.argmax(pred_cpu)  # Index of the class with the highest probability
            confidence_score = float(pred_cpu[predicted_class_idx]) # Confidence of the prediction
            
            # Map the predicted class index to a human-readable sentiment label
            # The label mapping depends on the model's configuration (number of labels).
            if model.config.num_labels == 2:
                # For models with 2 labels (e.g., Positive/Negative)
                label = 'POSITIVE' if predicted_class_idx == 1 else 'NEGATIVE'
            else:  # Assuming 3 classes (e.g., Positive/Negative/Neutral)
                # For '5CD-AI/Vietnamese-Sentiment-visobert', labels are:
                # LABEL_0: NEGATIVE, LABEL_1: NEUTRAL, LABEL_2: POSITIVE
                label_map = {0: 'NEGATIVE', 1: 'POSITIVE', 2: 'NEUTRAL'}
                label = label_map.get(predicted_class_idx, 'UNKNOWN') # Default to 'UNKNOWN' if index is unexpected
            
            results.append({
                'label': label,
                'score': confidence_score
            })
    
    return results

def process_dataframe(df, text_column='text', batch_size=32):
    """
    Applies sentiment classification to a text column in a pandas DataFrame.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        text_column (str): The name of the column containing the text to analyze.
        batch_size (int): The batch size for processing texts.
    
    Returns:
        pd.DataFrame: The DataFrame with two new columns: 'sentiment_label' and 
                      'sentiment_score', containing the prediction results.
    """
    # Validate that the specified text column exists in the DataFrame
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in the DataFrame.")

    # Extract texts, convert to string, and handle potential NaN values by replacing with empty strings
    texts_to_process = df[text_column].fillna('').astype(str).tolist()
    
    # Obtain sentiment predictions for all texts
    sentiment_predictions = predict_sentiment_batch(texts_to_process, batch_size=batch_size)
    
    # Create a copy of the DataFrame to avoid modifying the original
    df_result = df.copy()
    # Add new columns for sentiment labels and scores
    df_result['sentiment_label'] = [pred['label'] for pred in sentiment_predictions]
    df_result['sentiment_score'] = [pred['score'] for pred in sentiment_predictions]
    
    return df_result

def predict_single_batch_alternative(texts):
    """
    Predicts sentiment for a single list of texts (processed as one batch).
    Suitable for smaller lists where batching into multiple smaller chunks is not necessary.
    
    Args:
        texts (list): A list of text strings.
    
    Returns:
        list: A list of dictionaries, each with 'label' and 'score'.
    """
    # Tokenize the input texts
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors='pt'
    )
    
    # Move inputs to the configured device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = softmax(outputs.logits, dim=-1)
    
    results = []
    for pred_tensor in predictions:
        pred_cpu = pred_tensor.cpu().numpy()
        predicted_class_idx = np.argmax(pred_cpu)
        confidence_score = float(pred_cpu[predicted_class_idx])
        
        # Map class index to label
        if model.config.num_labels == 2:
            label = 'POSITIVE' if predicted_class_idx == 1 else 'NEGATIVE'
        else: 
            label_map = {0: 'NEGATIVE', 1: 'NEUTRAL', 2: 'POSITIVE'}
            label = label_map.get(predicted_class_idx, 'UNKNOWN')
            
        results.append({'label': label, 'score': confidence_score})
    
    return results

# Main execution block for command-line interface
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Perform sentiment analysis on Vietnamese text from a CSV file.")
    parser.add_argument("input_file", type=str, help="Path to the input CSV file.")
    parser.add_argument("output_file", type=str, help="Path to save the output CSV file with sentiment predictions.")
    parser.add_argument("--text_column", type=str, default="text", help="Name of the column in the CSV containing the text data (default: 'text').")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing texts (default: 32).")
    
    args = parser.parse_args()
    
    print(f"Using device: {device}")
    print(f"Loading data from: {args.input_file}")
    print(f"Text column: {args.text_column}")
    print(f"Batch size: {args.batch_size}")

    try:
        # Load the input CSV file into a pandas DataFrame
        input_df = pd.read_csv(args.input_file)
        print(f"Successfully loaded {len(input_df)} rows from {args.input_file}.")

        # Process the DataFrame to add sentiment predictions
        print("Processing DataFrame for sentiment analysis...")
        df_with_sentiment = process_dataframe(input_df, text_column=args.text_column, batch_size=args.batch_size)
        print("Sentiment analysis complete.")

        # Save the resulting DataFrame to the specified output CSV file
        df_with_sentiment.to_csv(args.output_file, index=False)
        print(f"Results saved to: {args.output_file}")

    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input_file}")
    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
