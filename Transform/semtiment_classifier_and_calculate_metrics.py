import pandas as pd
import numpy as np
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
import warnings
import argparse # For command-line argument parsing

# Suppress warnings (e.g., from transformers or pandas)
warnings.filterwarnings('ignore')

def init_sentiment_model():
    """
    Initializes and loads the pre-trained Vietnamese sentiment analysis model and tokenizer.
    The model is moved to GPU if available, otherwise CPU.

    Returns:
        tuple: (model, tokenizer, device)
               - model: The loaded PyTorch model for sequence classification.
               - tokenizer: The tokenizer associated with the model.
               - device: The torch.device (cuda or cpu) the model is on.
    """
    model_path = '5CD-AI/Vietnamese-Sentiment-visobert'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval() # Set model to evaluation mode
    
    return model, tokenizer, device

def predict_sentiment_batch(texts, model, tokenizer, device, batch_size=32):
    """
    Predicts sentiment for a batch of texts using the provided model and tokenizer.

    Args:
        texts (list): A list of text strings to analyze.
        model: The pre-trained sentiment analysis model.
        tokenizer: The tokenizer for the model.
        device: The torch.device to use for computation.
        batch_size (int): Number of texts to process in each batch.

    Returns:
        list: A list of dictionaries, each containing 'label' and 'score' for a text.
              Returns NEUTRAL with 0.5 score for empty or unprocessable texts.
    """
    if not texts: # Handle empty list of texts
        return []
    
    # Ensure all texts are strings, replace NaN with empty strings
    valid_texts = [str(text) if pd.notna(text) else "" for text in texts]
    results = []
    
    for i in range(0, len(valid_texts), batch_size):
        batch_texts = valid_texts[i:i + batch_size]
        
        # If the entire batch consists of empty strings after validation
        if not any(text.strip() for text in batch_texts):
            results.extend([{'label': 'NEUTRAL', 'score': 0.5}] * len(batch_texts))
            continue
            
        try:
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=256, # Max sequence length for the model
                return_tensors='pt' # Return PyTorch tensors
            )
            
            inputs = {k: v.to(device) for k, v in inputs.items()} # Move inputs to device
            
            with torch.no_grad(): # Disable gradient calculations for inference
                outputs = model(**inputs)
                predictions = softmax(outputs.logits, dim=-1) # Apply softmax to get probabilities
            
            for pred_tensor in predictions:
                pred_cpu = pred_tensor.cpu().numpy() # Move to CPU and convert to numpy
                predicted_class_idx = np.argmax(pred_cpu) # Get class with highest probability
                confidence = float(pred_cpu[predicted_class_idx])
                
                # Map predicted class index to label
                # For '5CD-AI/Vietnamese-Sentiment-visobert':
                if model.config.num_labels == 2: # Binary classification
                    label = 'POSITIVE' if predicted_class_idx == 1 else 'NEGATIVE'
                else: # Multiclass classification (assuming 3 classes for this model)
                    label_map = {0: 'NEGATIVE', 2: 'NEUTRAL', 1: 'POSITIVE'}
                    label = label_map.get(predicted_class_idx, 'NEUTRAL') # Default to NEUTRAL
                
                results.append({'label': label, 'score': confidence})
        
        except Exception as e:
            # If an error occurs during batch processing, assign neutral sentiment
            print(f"Error processing sentiment for a batch: {e}")
            results.extend([{'label': 'NEUTRAL', 'score': 0.5}] * len(batch_texts))
            
    return results

class ArtistAnalyzer:
    """
    Analyzes artist popularity and sentiment based on news articles and Google Trends data.
    """
    def __init__(self, df_news, df_trends, article_weight=0.4, trends_weight=0.6):
        """
        Initializes the ArtistAnalyzer.

        Args:
            df_news (pd.DataFrame): DataFrame with news articles.
                                    Expected columns: [title, comments, artist, date].
            df_trends (pd.DataFrame): DataFrame with Google Trends data.
                                      Expected columns: [date, artist_1, artist_2, ...].
            article_weight (float): Weight for article frequency in popularity score.
            trends_weight (float): Weight for Google Trends data in popularity score.
        """
        self.df_news = df_news.copy()
        self.df_trends = df_trends.copy()
        self.article_weight = article_weight
        self.trends_weight = trends_weight
        
        self.model, self.tokenizer, self.device = init_sentiment_model()
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepares and cleans the input DataFrames."""
        print("Preparing data...")
        # Convert 'date' columns to datetime objects, coercing errors to NaT
        if 'date' in self.df_news.columns:
            self.df_news['date'] = pd.to_datetime(self.df_news['date'], errors='coerce')
        else:
            raise ValueError("Column 'date' not found in news DataFrame.")

        if 'date' in self.df_trends.columns:
            self.df_trends['date'] = pd.to_datetime(self.df_trends['date'], errors='coerce')
        else:
            raise ValueError("Column 'date' not found in trends DataFrame.")

        # Drop rows where date conversion failed
        self.df_news.dropna(subset=['date'], inplace=True)
        self.df_trends.dropna(subset=['date'], inplace=True)

        # Add 'week' column (Period object representing the week)
        self.df_news['week'] = self.df_news['date'].dt.to_period('W')
        self.df_trends['week'] = self.df_trends['date'].dt.to_period('W')
        
        # Identify artist columns from the trends DataFrame
        self.artists_trends_cols = [col for col in self.df_trends.columns if col not in ['date', 'week']]
        
        # Create a mapping from original artist column names (in trends) to cleaned names
        self.artist_mapping = {
            artist_col: artist_col.replace('_', ' ').strip() for artist_col in self.artists_trends_cols
        }
        # Get a list of unique cleaned artist names
        self.cleaned_artist_names = list(self.artist_mapping.values())

        print(f"Identified {len(self.cleaned_artist_names)} unique artists from trends data.")
        print("Data preparation complete.")

    def parse_comments(self):
        """
        Parses the 'comments' column from df_news.
        Assumes comments are stored as a single string, separated by '|||'.
        Creates a new DataFrame (self.df_comments) with one row per individual comment.
        """
        print("Parsing comments...")
        if 'comments' not in self.df_news.columns:
            print("Warning: 'comments' column not found in news DataFrame. Skipping comment parsing.")
            # Create an empty df_comments with expected columns if 'comments' is missing
            self.df_comments = pd.DataFrame(columns=[
                'original_index', 'title', 'artist', 'date', 'week', 
                'comment', 'comment_index', 'total_comments_in_article'
            ])
            return self.df_comments

        parsed_rows = []
        for idx, row in self.df_news.iterrows():
            article_comments_str = row['comments']
            individual_comments = []

            if pd.notna(article_comments_str) and isinstance(article_comments_str, str) and article_comments_str.strip():
                individual_comments = [c.strip() for c in article_comments_str.split('|||') if c.strip()]
            
            if not individual_comments: # No valid comments or NaN
                parsed_rows.append({
                    'original_index': idx, 'title': row['title'], 'artist': row['artist'],
                    'date': row['date'], 'week': row['week'], 'comment': "",
                    'comment_index': 0, 'total_comments_in_article': 0
                })
            else:
                for c_idx, comment_text in enumerate(individual_comments):
                    parsed_rows.append({
                        'original_index': idx, 'title': row['title'], 'artist': row['artist'],
                        'date': row['date'], 'week': row['week'], 'comment': comment_text,
                        'comment_index': c_idx, 'total_comments_in_article': len(individual_comments)
                    })
                    
        self.df_comments = pd.DataFrame(parsed_rows)
        print(f"Parsed {len(self.df_comments)} individual comment entries from {self.df_news.shape[0]} articles.")
        return self.df_comments
        
    def analyze_sentiment(self, batch_size=32):
        """Analyzes sentiment of individual comments stored in self.df_comments."""
        if not hasattr(self, 'df_comments'): # Ensure comments are parsed
            self.parse_comments()
        
        print("Analyzing sentiment of comments...")
        if self.df_comments.empty or 'comment' not in self.df_comments.columns:
            print("No comments to analyze or 'comment' column missing.")
            # Add empty sentiment columns if df_comments exists but is empty or lacks 'comment'
            if hasattr(self, 'df_comments'):
                 self.df_comments['sentiment_label'] = pd.Series(dtype='object')
                 self.df_comments['sentiment_score'] = pd.Series(dtype='float')
            return

        # Filter out rows where 'comment' is an empty string for sentiment prediction
        non_empty_comment_rows = self.df_comments[self.df_comments['comment'].str.strip() != '']
        comments_to_analyze = non_empty_comment_rows['comment'].tolist()
        
        if not comments_to_analyze:
            print("No non-empty comments found for sentiment analysis.")
            # Assign neutral to all rows if no non-empty comments
            self.df_comments['sentiment_label'] = 'NEUTRAL'
            self.df_comments['sentiment_score'] = 0.5
            return
            
        sentiment_predictions = predict_sentiment_batch(
            comments_to_analyze, self.model, self.tokenizer, self.device, batch_size
        )
        
        # Initialize sentiment columns with default neutral values
        self.df_comments['sentiment_label'] = 'NEUTRAL'
        self.df_comments['sentiment_score'] = 0.5
        
        # Assign predicted sentiments to the corresponding non-empty comment rows
        # Ensure indices align correctly
        pred_labels = [p['label'] for p in sentiment_predictions]
        pred_scores = [p['score'] for p in sentiment_predictions]

        self.df_comments.loc[non_empty_comment_rows.index, 'sentiment_label'] = pred_labels
        self.df_comments.loc[non_empty_comment_rows.index, 'sentiment_score'] = pred_scores
        
        print(f"Sentiment analysis complete for {len(comments_to_analyze)} non-empty comments.")

    def calculate_weekly_popularity(self):
        """Calculates weekly popularity scores for each artist."""
        print("Calculating weekly popularity...")
        weekly_popularity_data = []
        
        # Consolidate unique weeks from both news and trends data
        all_weeks = pd.concat([self.df_news['week'], self.df_trends['week']]).unique()
        all_weeks = sorted([w for w in all_weeks if pd.notna(w)]) # Sort and remove NaT

        # Determine max articles per week across all artists for normalization
        # Group by week and count unique original_index from df_news related to any artist
        # This is a simplification; a more robust normalization might be needed.
        if not self.df_news.empty:
            max_articles_overall_per_week = self.df_news.groupby('week')['original_index'].nunique().max()
            if pd.isna(max_articles_overall_per_week) or max_articles_overall_per_week == 0:
                 max_articles_overall_per_week = 1 # Avoid division by zero
        else:
            max_articles_overall_per_week = 1


        for week in all_weeks:
            week_entry = {'week': week}
            for artist_col, cleaned_name in self.artist_mapping.items():
                # Article frequency: Count unique articles mentioning the artist in the given week
                num_articles = 0
                if not self.df_news.empty and 'artist' in self.df_news.columns:
                    artist_news_this_week = self.df_news[
                        (self.df_news['week'] == week) &
                        (self.df_news['artist'].astype(str).str.contains(cleaned_name, case=False, na=False))
                    ]
                    num_articles = artist_news_this_week['original_index'].nunique()

                normalized_articles = num_articles / max_articles_overall_per_week if max_articles_overall_per_week > 0 else 0
                
                # Trends score
                trends_this_week = self.df_trends[self.df_trends['week'] == week]
                trends_score = 0
                if not trends_this_week.empty and artist_col in trends_this_week.columns:
                    trends_score = trends_this_week[artist_col].iloc[0]
                trends_score = trends_score if pd.notna(trends_score) else 0 # Handle NaN trends score

                # Popularity score calculation
                popularity = (self.article_weight * normalized_articles) + \
                             (self.trends_weight * (trends_score / 100.0)) # Assuming trends are 0-100
                
                week_entry[f'{cleaned_name}_articles'] = num_articles
                week_entry[f'{cleaned_name}_trends'] = trends_score
                week_entry[f'{cleaned_name}_popularity'] = popularity
            weekly_popularity_data.append(week_entry)
            
        df_weekly_popularity = pd.DataFrame(weekly_popularity_data)
        if 'week' in df_weekly_popularity.columns:
            df_weekly_popularity['week'] = df_weekly_popularity['week'].astype(str) # Convert Period to string for CSV
        print("Weekly popularity calculation complete.")
        return df_weekly_popularity

    def calculate_weekly_sentiment(self):
        """Calculates weekly sentiment metrics for each artist."""
        print("Calculating weekly sentiment...")
        if not hasattr(self, 'df_comments') or self.df_comments.empty:
            print("No comments data available for weekly sentiment. Skipping.")
            return pd.DataFrame(columns=['week']) # Return empty DataFrame with 'week' column

        weekly_sentiment_data = []
        # Use unique weeks from comments data, as sentiment is derived from comments
        comment_weeks = sorted([w for w in self.df_comments['week'].unique() if pd.notna(w)])

        for week in comment_weeks:
            week_entry = {'week': week}
            for cleaned_name in self.cleaned_artist_names:
                # Filter comments for the current artist and week
                artist_comments_this_week = self.df_comments[
                    (self.df_comments['week'] == week) &
                    (self.df_comments['artist'].astype(str).str.contains(cleaned_name, case=False, na=False)) &
                    (self.df_comments['comment'].str.strip() != '') # Consider only non-empty comments
                ]
                
                num_comments = len(artist_comments_this_week)
                num_articles_related = artist_comments_this_week['original_index'].nunique()

                if num_comments > 0:
                    sentiment_counts = artist_comments_this_week['sentiment_label'].value_counts(normalize=True)
                    pos_pct = sentiment_counts.get('POSITIVE', 0.0)
                    neg_pct = sentiment_counts.get('NEGATIVE', 0.0)
                    neu_pct = sentiment_counts.get('NEUTRAL', 0.0)
                    avg_sentiment_score = pos_pct - neg_pct # Simple sentiment score: (Pos% - Neg%)
                else:
                    pos_pct, neg_pct, neu_pct, avg_sentiment_score = 0.0, 0.0, 0.0, 0.0
                
                week_entry[f'{cleaned_name}_articles_with_comments'] = num_articles_related
                week_entry[f'{cleaned_name}_total_comments'] = num_comments
                week_entry[f'{cleaned_name}_positive_pct'] = pos_pct
                week_entry[f'{cleaned_name}_negative_pct'] = neg_pct
                week_entry[f'{cleaned_name}_neutral_pct'] = neu_pct
                week_entry[f'{cleaned_name}_avg_sentiment'] = avg_sentiment_score
            weekly_sentiment_data.append(week_entry)
            
        df_weekly_sentiment = pd.DataFrame(weekly_sentiment_data)
        if 'week' in df_weekly_sentiment.columns:
             df_weekly_sentiment['week'] = df_weekly_sentiment['week'].astype(str) # Convert Period to string
        print("Weekly sentiment calculation complete.")
        return df_weekly_sentiment

    def calculate_overall_metrics(self):
        """Calculates overall (all-time) metrics for each artist."""
        print("Calculating overall metrics...")
        overall_metrics_data = []

        # Max total articles across all artists for normalization
        if not self.df_news.empty and 'artist' in self.df_news.columns:
             max_total_articles_overall = self.df_news.groupby('artist')['original_index'].nunique().max()
             if pd.isna(max_total_articles_overall) or max_total_articles_overall == 0:
                  max_total_articles_overall = 1
        else:
             max_total_articles_overall = 1


        for artist_col, cleaned_name in self.artist_mapping.items():
            # Overall article count (unique articles associated with the artist)
            total_articles = 0
            if not self.df_news.empty and 'artist' in self.df_news.columns:
                 artist_news = self.df_news[self.df_news['artist'].astype(str).str.contains(cleaned_name, case=False, na=False)]
                 total_articles = artist_news['original_index'].nunique()
            
            normalized_total_articles = total_articles / max_total_articles_overall if max_total_articles_overall > 0 else 0

            # Average Google Trends score
            avg_trends = 0
            if artist_col in self.df_trends.columns:
                avg_trends = self.df_trends[artist_col].mean()
            avg_trends = avg_trends if pd.notna(avg_trends) else 0

            # Overall popularity
            overall_pop = (self.article_weight * normalized_total_articles) + \
                          (self.trends_weight * (avg_trends / 100.0))
            
            # Overall sentiment
            total_artist_comments = pd.DataFrame()
            if hasattr(self, 'df_comments') and not self.df_comments.empty:
                total_artist_comments = self.df_comments[
                    (self.df_comments['artist'].astype(str).str.contains(cleaned_name, case=False, na=False)) &
                    (self.df_comments['comment'].str.strip() != '')
                ]

            num_total_comments = len(total_artist_comments)
            if num_total_comments > 0:
                overall_sent_counts = total_artist_comments['sentiment_label'].value_counts(normalize=True)
                ov_pos_pct = overall_sent_counts.get('POSITIVE', 0.0)
                ov_neg_pct = overall_sent_counts.get('NEGATIVE', 0.0)
                ov_neu_pct = overall_sent_counts.get('NEUTRAL', 0.0)
                ov_avg_sent = ov_pos_pct - ov_neg_pct
            else:
                ov_pos_pct, ov_neg_pct, ov_neu_pct, ov_avg_sent = 0.0, 0.0, 0.0, 0.0
                
            overall_metrics_data.append({
                'artist': cleaned_name,
                'total_articles': total_articles,
                'avg_trends_score': avg_trends,
                'overall_popularity': overall_pop,
                'total_comments': num_total_comments,
                'overall_positive_pct': ov_pos_pct,
                'overall_negative_pct': ov_neg_pct,
                'overall_neutral_pct': ov_neu_pct,
                'overall_avg_sentiment': ov_avg_sent
            })
            
        df_overall_metrics = pd.DataFrame(overall_metrics_data)
        print("Overall metrics calculation complete.")
        return df_overall_metrics
        
    def run_full_analysis(self):
        """Runs the complete analysis pipeline."""
        print("Starting full artist analysis workflow...")
        
        # Sentiment analysis must run first as other calculations might depend on df_comments
        self.analyze_sentiment()
        
        results = {
            'weekly_popularity': self.calculate_weekly_popularity(),
            'weekly_sentiment': self.calculate_weekly_sentiment(),
            'overall_metrics': self.calculate_overall_metrics(),
            'processed_news_data': self.df_news.copy(), # Return a copy of the processed news data
            'parsed_comments_data': self.df_comments.copy() if hasattr(self, 'df_comments') else pd.DataFrame()
        }
        print("Full artist analysis workflow complete!")
        return results

def main():
    """Main function to run the artist analyzer from the command line."""
    parser = argparse.ArgumentParser(description="Analyze artist popularity and sentiment from news and trends data.")
    parser.add_argument("news_file", type=str, help="Path to the CSV file containing news data.")
    parser.add_argument("trends_file", type=str, help="Path to the CSV file containing Google Trends data.")
    parser.add_argument("--article_weight", type=float, default=0.4, help="Weight for article frequency in popularity (0.0-1.0).")
    parser.add_argument("--trends_weight", type=float, default=0.6, help="Weight for Google Trends in popularity (0.0-1.0).")
    
    parser.add_argument("--output_weekly_popularity", type=str, default="weekly_popularity.csv", help="Output path for weekly popularity CSV.")
    parser.add_argument("--output_weekly_sentiment", type=str, default="weekly_sentiment.csv", help="Output path for weekly sentiment CSV.")
    parser.add_argument("--output_overall_metrics", type=str, default="overall_metrics.csv", help="Output path for overall metrics CSV.")
    parser.add_argument("--output_parsed_comments", type=str, default=None, help="Optional: Output path for parsed comments CSV.")
    
    args = parser.parse_args()

    # Validate weights
    if not (0.0 <= args.article_weight <= 1.0 and 0.0 <= args.trends_weight <= 1.0):
        print("Error: Weights must be between 0.0 and 1.0.")
        return
    if abs((args.article_weight + args.trends_weight) - 1.0) > 1e-9: # Check if weights sum to 1
        print("Warning: Article weight and trends weight do not sum to 1.0. Normalizing them.")
        total_weight = args.article_weight + args.trends_weight
        if total_weight > 0 :
            args.article_weight /= total_weight
            args.trends_weight /= total_weight
        else: # Avoid division by zero if both are zero
            args.article_weight = 0.5
            args.trends_weight = 0.5


    print(f"Loading news data from: {args.news_file}")
    print(f"Loading trends data from: {args.trends_file}")
    
    try:
        df_news_input = pd.read_csv(args.news_file)
        df_trends_input = pd.read_csv(args.trends_file)
        print("Data loaded successfully.")

        analyzer = ArtistAnalyzer(df_news_input, df_trends_input, 
                                  article_weight=args.article_weight, 
                                  trends_weight=args.trends_weight)
        
        analysis_results = analyzer.run_full_analysis()
        
        # Save results
        analysis_results['weekly_popularity'].to_csv(args.output_weekly_popularity, index=False)
        print(f"Weekly popularity data saved to {args.output_weekly_popularity}")
        
        analysis_results['weekly_sentiment'].to_csv(args.output_weekly_sentiment, index=False)
        print(f"Weekly sentiment data saved to {args.output_weekly_sentiment}")
        
        analysis_results['overall_metrics'].to_csv(args.output_overall_metrics, index=False)
        print(f"Overall metrics data saved to {args.output_overall_metrics}")

        if args.output_parsed_comments and not analysis_results['parsed_comments_data'].empty:
            analysis_results['parsed_comments_data'].to_csv(args.output_parsed_comments, index=False)
            print(f"Parsed comments data saved to {args.output_parsed_comments}")
            
    except FileNotFoundError as e:
        print(f"Error: File not found. Details: {e}")
    except ValueError as ve:
        print(f"ValueError during processing: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    main()
