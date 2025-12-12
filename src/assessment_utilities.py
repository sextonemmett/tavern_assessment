"""
Utility functions for the Data Science skills assessment.
This file contains helper functions that candidates can use during the assessment.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Union, Tuple, Optional
import requests
import os
import hashlib
import json
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import torch

# Skeleton functions for candidates to implement

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw input dataframe into a feature-engineered dataframe ready for modeling.
    
    This function should:
    1. Perform any necessary data cleaning
    2. Create all features needed for your model
    3. Handle any text processing using assessment_utilities.py functions
    
    Args:
        df (pd.DataFrame): Raw input dataframe with original variables
            
    Returns:
        pd.DataFrame: Dataframe with all features needed for the model
    """
    # Your feature engineering code here
    # ...
    
    return df  # Replace with your engineered dataframe


def predict_increased_trump_approval(df: pd.DataFrame) -> np.ndarray:
    """
    End-to-end function that transforms the raw input dataframe into a feature-engineered dataframe
    ready for modeling and then trains and evaluates an XGBoost classifier with cross-validation.
    
    This function should:
    1. Call compute_features() to transform the raw input dataframe into a 
       feature-engineered dataframe ready for modeling.
    2. Apply any transformations needed (e.g., scaling)
    3. Train an XGBoost classifier and evaluate it with cross-validation, reporting
       the mean and standard deviation of AUC across folds.
    
    Args:
        df (pd.DataFrame): Raw input dataframe with original variables
            
    Returns:
        np.ndarray: Mean and standard deviation of AUC across folds
    """
    # Step 1: Generate features
    features_df = compute_features(df)
    
    # Step 2: Apply any necessary transformations (scaling, etc.)
    # Your transformation code here
    # ...
    
    # Step 3: Train and evaluate a model
    # Your modeling code here
    # ...
    

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the dataset from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame containing the dataset
    """
    return pd.read_csv(file_path)


def get_embeddings(texts: List[str]) -> np.ndarray:
    """
    Get embeddings for a list of texts using Sentence Transformers.
    Uses a caching system to avoid recomputing embeddings for texts that have been processed before.
    
    Args:
        texts: List of text strings to embed
        
    Returns:
        NumPy array of embeddings
    """
    # Do not change the model name and cache directory
    model_name = 'paraphrase-MiniLM-L3-v2'
    embedding_dim = 384
    cache_dir = 'cached_embeddings'
 
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Load the model
    model = SentenceTransformer(model_name) 
    
    # Initialize array for all embeddings
    all_embeddings = np.zeros((len(texts), embedding_dim))
    
    # Process each text
    for i, text in enumerate(texts):
        if text is None or not isinstance(text, str) or text.strip() == "":
            # Handle empty or None text
            all_embeddings[i] = np.zeros(embedding_dim)
            continue
            
        # Create a hash of the text to use as a filename
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        cache_file = os.path.join(cache_dir, f"{text_hash}.npy")
        
        if os.path.exists(cache_file):
            # Load from cache
            try:
                embedding = np.load(cache_file)
                all_embeddings[i] = embedding
            except Exception as e:
                print(f"Error loading cached embedding: {e}")
                # Compute embedding if loading fails
                embedding = model.encode([text])[0]
                all_embeddings[i] = embedding
                # Save to cache
                np.save(cache_file, embedding)
        else:
            # Compute embedding
            embedding = model.encode([text])[0]
            all_embeddings[i] = embedding
            # Save to cache
            np.save(cache_file, embedding)
    
    return all_embeddings

def call_llama(prompt: str, 
               system_prompt: str = "You are a helpful AI assistant.",
               temperature: float = 0.1, 
               max_tokens: int = 1000,
               timeout: int = 60,
               expected_format: str = "text",
               debug: bool = False,
               **kwargs) -> str:
    """
    Call a local Llama 3.2 model with a prompt using Ollama API.
    
    Args:
        prompt: The user prompt to send to the model
        system_prompt: System prompt to set context for the model
        temperature: Controls randomness (0.0 = deterministic, 1.0 = creative)
        max_tokens: Maximum number of tokens to generate
        timeout: Maximum time to wait for API response in seconds
        expected_format: Type of response expected:
            - 'text': Any non-empty text (default)
            - 'number': A numeric value (will be extracted from response)
            - 'word': A single word (first word will be extracted if multiple)
            - 'category': A value from a predefined list (requires valid_categories kwarg)
            - 'binary': A yes/no response (will normalize variations)
        debug: Whether to print detailed debug information
        **kwargs: Additional keyword arguments:
            - valid_categories: List of valid categories (required for 'category' format)
        
    Returns:
        Model's response as a string, formatted according to expected_format
    """
    import json
    import time
    
    if debug:
        print(f"\n[DEBUG] call_llama: Starting with prompt: '{prompt[:50]}...'")
        print(f"[DEBUG] call_llama: Using model: llama3.2, timeout: {timeout}s")
    
    start_time = time.time()
    
    try:
        # Set up the API endpoint for Ollama
        API_URL = "http://localhost:11434/api/chat"
        
        # Prepare the request payload
        payload = {
            "model": "llama3.2",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        if debug:
            print(f"[DEBUG] call_llama: Sending request to {API_URL}")
            print(f"[DEBUG] call_llama: Payload: {json.dumps(payload)[:100]}...")
        
        # Make the API call with timeout
        try:
            response = requests.post(API_URL, json=payload, timeout=timeout)
            response.raise_for_status()  # Raise an exception for HTTP errors
        except requests.exceptions.Timeout:
            error_msg = f"Timeout error: Ollama API did not respond within {timeout} seconds"
            print(error_msg)
            return f"Error: {error_msg}"
        except requests.exceptions.ConnectionError:
            error_msg = "Connection error: Could not connect to Ollama API. Is Ollama running?"
            print(error_msg)
            return f"Error: {error_msg}"
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP error: {e}"
            print(error_msg)
            return f"Error: {error_msg}"
 
        # Parse the response - handle streaming response format
        response_text = response.text
        
        if debug:
            print(f"[DEBUG] call_llama: Received response in {time.time() - start_time:.2f}s")
            print(f"[DEBUG] call_llama: Response text: {response_text[:200]}...")
        
        # Ollama returns multiple JSON objects - we need to find the one with content
        lines = response_text.strip().split('\n')
        content = None
        
        # First try to find any JSON object with non-empty content
        for line in lines:
            try:
                obj = json.loads(line)
                # Check if this object has a non-empty message content
                line_content = obj.get("message", {}).get("content")
                if line_content:
                    content = line_content
                    break
            except json.JSONDecodeError:
                if debug:
                    print(f"[DEBUG] call_llama: Could not parse line as JSON: {line[:50]}...")
                continue
        
        # If we didn't find any content, try to parse the first object
        if not content:
            try:
                first_obj = json.loads(lines[0]) if lines else {}
                content = first_obj.get("message", {}).get("content", "")
            except (json.JSONDecodeError, IndexError):
                if debug:
                    print("[DEBUG] call_llama: Could not parse first line as JSON")
                # If all else fails, use the raw response
                content = response_text.strip()
        
        # Validate the response format if expected_format is provided
        if expected_format and content:
            if expected_format == 'number':
                # Try to convert to a number
                try:
                    # Strip any non-numeric characters
                    numeric_content = ''.join(c for c in content if c.isdigit() or c == '.')
                    float(numeric_content)  # Just to validate
                    content = numeric_content
                    if debug:
                        print(f"[DEBUG] call_llama: Validated numeric response: {content}")
                except ValueError:
                    error_msg = f"Response validation error: Expected a number but got '{content}'"
                    print(error_msg)
                    if debug:
                        print(f"[DEBUG] call_llama: {error_msg}")
                    return f"Error: {error_msg}"
            
            elif expected_format == 'word':
                # Check if the response is a single word (no spaces, punctuation allowed)
                content = content.strip()
                if ' ' in content:
                    # Extract just the first word
                    first_word = content.split()[0].strip()
                    if debug:
                        print(f"[DEBUG] call_llama: Expected single word but got multiple. Using first word: '{first_word}'")
                    content = first_word
                
                # Remove any punctuation at the beginning or end
                content = content.strip('.,;:!?"\'-()[]{}')
                if debug:
                    print(f"[DEBUG] call_llama: Validated word response: '{content}'")
            
            elif expected_format == 'category':
                # This requires a valid_categories list to be passed in the kwargs
                if 'valid_categories' not in kwargs:
                    error_msg = "Response validation error: 'valid_categories' must be provided for 'category' format"
                    print(error_msg)
                    return f"Error: {error_msg}"
                
                valid_categories = kwargs['valid_categories']
                content = content.strip().lower()
                
                # Direct match
                if content in [c.lower() for c in valid_categories]:
                    # Find the original case version
                    for c in valid_categories:
                        if c.lower() == content:
                            content = c
                            break
                    if debug:
                        print(f"[DEBUG] call_llama: Validated category response: '{content}'")
                else:
                    # Try to find a partial match or match with spaces removed
                    content_nospaces = ''.join(content.split())
                    best_match = None
                    best_score = 0
                    
                    for category in valid_categories:
                        # Check for exact match with spaces removed
                        category_nospaces = ''.join(category.lower().split())
                        if content_nospaces == category_nospaces:
                            best_match = category
                            break
                        
                        # Check if the content is contained within the category
                        if content in category.lower():
                            # Score based on length ratio
                            score = len(content) / len(category.lower())
                            if score > best_score:
                                best_score = score
                                best_match = category
                    
                    if best_match and best_score > 0.5:  # Require at least 50% match
                        content = best_match
                        if debug:
                            print(f"[DEBUG] call_llama: Fuzzy matched category '{content_nospaces}' to '{best_match}'")
                    else:
                        options = ", ".join(valid_categories)
                        error_msg = f"Response validation error: '{content}' is not a valid category. Valid options: {options}"
                        print(error_msg)
                        if debug:
                            print(f"[DEBUG] call_llama: {error_msg}")
                        return f"Error: {error_msg}"
            
            elif expected_format == 'binary':
                # Special case for yes/no responses
                content = content.strip().lower()
                
                # Direct match
                if content in ['yes', 'no']:
                    if debug:
                        print(f"[DEBUG] call_llama: Validated binary response: '{content}'")
                else:
                    # Try to match variations
                    if content in ['y', 'yeah', 'yep', 'yea', 'affirmative', 'correct', 'true', '1']:
                        content = 'yes'
                        if debug:
                            print(f"[DEBUG] call_llama: Normalized binary response to 'yes'")
                    elif content in ['n', 'nope', 'nah', 'negative', 'incorrect', 'false', '0']:
                        content = 'no'
                        if debug:
                            print(f"[DEBUG] call_llama: Normalized binary response to 'no'")
                    else:
                        error_msg = f"Response validation error: '{content}' is not a valid binary response. Use 'yes' or 'no'."
                        print(error_msg)
                        if debug:
                            print(f"[DEBUG] call_llama: {error_msg}")
                        return f"Error: {error_msg}"
            
            elif expected_format == 'text':
                # For text format, just ensure it's a non-empty string and trim whitespace
                content = content.strip()
                if not content:
                    error_msg = "Response validation error: Expected non-empty text but got empty response"
                    print(error_msg)
                    if debug:
                        print(f"[DEBUG] call_llama: {error_msg}")
                    return f"Error: {error_msg}"
                if debug:
                    print(f"[DEBUG] call_llama: Validated text response: '{content[:50]}...'")
            
            else:
                # Unknown format
                if debug:
                    print(f"[DEBUG] call_llama: Unknown expected_format '{expected_format}', returning raw content")
        
        if not content:
            error_msg = "No content returned from Ollama API"
            print(error_msg)
            return f"Error: {error_msg}"
        
        if debug:
            print(f"[DEBUG] call_llama: Successfully extracted content: '{content}'")
            print(f"[DEBUG] call_llama: Total processing time: {time.time() - start_time:.2f}s")
        
        return content
    
    except Exception as e:
        error_msg = f"Unexpected error calling Ollama API: {str(e)}"
        print(error_msg)
        if debug:
            import traceback
            print(f"[DEBUG] call_llama: Exception details:\n{traceback.format_exc()}")
        return f"Error: {error_msg}"
