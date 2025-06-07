"""
LLM Services for ACME Security Bot.

This module provides Language Model (LLM) based services for the security bot,
including text analysis, classification, and response generation. It uses
open-source models from Hugging Face and implements various optimization
techniques for efficient inference.

Key Features:
- Automatic model downloading and caching
- GPU acceleration when available
- Optimized inference settings
- Error handling and retries
- Disk space management
- Fallback mechanisms for reliability
"""
import sys
from pathlib import Path
import logging
import re
import os
import time
import torch
from typing import Dict, List, Tuple, Optional
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    logging as transformers_logging
)
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from enum import Enum
from huggingface_hub import snapshot_download
import shutil

# Configure logging and warnings
transformers_logging.set_verbosity_error()  # Only show errors, not warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import project-specific modules
from src.models.enums import RequestType

logger = logging.getLogger(__name__)

__all__ = ['LLMAnalyzer', 'LLMClassificationService']

class LLMAnalyzer:
    """
    Enhanced LLM Analyzer using open-source models.
    
    This class provides a robust interface for working with various open-source
    language models, with features including:
    - Automatic model downloading and caching
    - GPU acceleration when available
    - Optimized inference settings
    - Error handling and retries
    - Disk space management
    """

    def __init__(self, model_name="Qwen/Qwen3-0.6B"): 
        """
        Initialize the LLM analyzer with the specified model.
        
        This method handles model downloading, verification, and initialization
        with proper error handling and retries. It also manages disk space
        and optimizes model settings for efficient inference.
        
        Args:
            model_name (str): Name of the Hugging Face model to use.
                            Defaults to "Qwen/Qwen3-0.6B".
                            
        Raises:
            RuntimeError: If there's insufficient disk space or model loading fails.
            ValueError: If model test fails.
        """
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                logger.info(f"Attempting to load {model_name} (attempt {retry_count + 1}/{max_retries})")
                
                # Handle model downloading and verification
                try:
                    # Set up cache directory
                    cache_dir = os.getenv("TRANSFORMERS_CACHE", "./models")
                    os.makedirs(cache_dir, exist_ok=True)
                    
                    # Check available disk space
                    total, used, free = shutil.disk_usage(cache_dir)
                    free_gb = free / (1024 * 1024 * 1024)  # Convert to GB
                    
                    # Model size estimates for space management
                    model_sizes = {
                        "Qwen/Qwen3-0.6B": 2.0,  # GB
                        "microsoft/phi-2": 1.5,
                        "TinyLlama/TinyLlama-1.1B-Chat-v1.0": 1.0,
                        "facebook/opt-125m": 0.3
                    }
                    
                    required_space = model_sizes.get(model_name, 3.0)  # Default to 3GB if unknown
                    if free_gb < required_space * 2:  # Need 2x space for safe download
                        raise RuntimeError(f"Insufficient disk space. Need {required_space*2:.1f}GB, have {free_gb:.1f}GB free")
                    
                    # Download or verify model
                    model_path = os.path.join(cache_dir, f"models--{model_name.replace('/', '--')}")
                    if not os.path.exists(model_path):
                        logger.info(f"Downloading {model_name} to {model_path}...")
                        
                        # Clean up old models if space is tight
                        if free_gb < required_space * 3:  # Less than 3x required space
                            self._cleanup_old_models(cache_dir)
                        
                        # Download with progress tracking
                        snapshot_download(
                            model_name,
                            local_dir=model_path,
                            local_dir_use_symlinks=False,
                            resume_download=True
                        )
                    else:
                        logger.info(f"Model already downloaded at {model_path}")
                        
                        # Verify model integrity
                        if not self._verify_model_files(model_path):
                            logger.warning("Model files appear corrupted, re-downloading...")
                            shutil.rmtree(model_path, ignore_errors=True)
                            snapshot_download(
                                model_name,
                                local_dir=model_path,
                                local_dir_use_symlinks=False,
                                resume_download=True
                            )
                        
                except Exception as download_err:
                    logger.error(f"Model download failed: {download_err}")
                    raise
                
                # Initialize and configure tokenizer
                logger.info("Initializing tokenizer...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True,  # Required for some models
                    use_fast=True,
                    cache_dir=cache_dir,
                    padding_side='left'
                )
                
                # Ensure tokenizer has padding token
                if self.tokenizer.pad_token_id is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                
                # Initialize model with optimized settings
                logger.info("Loading model...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,  # Use FP16 for efficiency
                    device_map="auto",
                    trust_remote_code=True,  # Required for some models
                    low_cpu_mem_usage=True,
                    cache_dir=cache_dir,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
                # Optimize for GPU if available
                if torch.cuda.is_available():
                    self.model = self.model.cuda()
                    # Enable CUDA optimizations
                    torch.backends.cudnn.benchmark = True
                else:
                    logger.warning("GPU not available, using CPU (this will be slow)")
                
                # Set model to evaluation mode
                self.model.eval()
                
                # Verify model functionality
                logger.info("Testing model...")
                test_prompt = "Classify this: Need access to production server"
                test_response = self.generate_response(test_prompt, max_new_tokens=20)
                if not test_response or len(test_response.strip()) == 0:
                    raise ValueError("Model test failed - empty response")
                
                logger.info(f"✅ Loaded {model_name} successfully")
                break
                
            except Exception as e:
                retry_count += 1
                if retry_count == max_retries:
                    logger.error(f"Failed to load {model_name} after {max_retries} attempts: {e}")
                    raise
                else:
                    logger.warning(f"Attempt {retry_count} failed, retrying in 5 seconds... Error: {e}")
                    time.sleep(5)

    def _cleanup_old_models(self, cache_dir: str):
        """
        Clean up old model files to free disk space.
        
        This method removes old model files based on last modified time,
        keeping only the most recent models to maintain sufficient free space.
        
        Args:
            cache_dir (str): Path to the model cache directory.
        """
        try:
            # Get list of model directories with metadata
            model_dirs = []
            for d in os.listdir(cache_dir):
                path = os.path.join(cache_dir, d)
                if os.path.isdir(path) and d.startswith('models--'):
                    # Get last modified time and size
                    mtime = os.path.getmtime(path)
                    size = sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file())
                    model_dirs.append((path, mtime, size))
            
            # Sort by last modified time (oldest first)
            model_dirs.sort(key=lambda x: x[1])
            
            # Remove old models until we have enough space
            for path, _, size in model_dirs[:-1]:  # Keep the most recent
                try:
                    logger.info(f"Removing old model: {path}")
                    shutil.rmtree(path)
                except Exception as e:
                    logger.warning(f"Failed to remove {path}: {e}")
                
                # Check if we have enough space now
                free_gb = shutil.disk_usage(cache_dir).free / (1024 * 1024 * 1024)
                if free_gb >= 5:  # Stop if we have 5GB free
                    break
                    
        except Exception as e:
            logger.warning(f"Failed to clean up old models: {e}")

    def _verify_model_files(self, model_path: str) -> bool:
        """
        Verify the integrity of downloaded model files.
        
        This method checks for the presence and validity of essential model files
        to ensure the model was downloaded correctly.
        
        Args:
            model_path (str): Path to the model directory.
            
        Returns:
            bool: True if all required files are present and valid, False otherwise.
        """
        try:
            # Check for essential model files
            required_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json']
            for file in required_files:
                file_path = os.path.join(model_path, file)
                if not os.path.exists(file_path):
                    return False
                    
                # Verify file size is reasonable
                if os.path.getsize(file_path) < 1024:
                    return False
            
            return True
            
        except Exception:
            return False

    def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generate a response using the loaded language model.
        
        This method handles the complete generation process with optimized
        settings for efficient inference. It includes proper error handling
        and resource management.
        
        Args:
            prompt (str): The input prompt for generation.
            **kwargs: Additional generation parameters to override defaults.
            
        Returns:
            str: The generated response text.
            
        Raises:
            Exception: If generation fails.
        """
        try:
            # Encode the prompt efficiently
            with torch.inference_mode():  # Faster than no_grad
                # Encode and move to GPU if available
                inputs = self.tokenizer.encode(
                    prompt, 
                    return_tensors="pt",
                    add_special_tokens=True,
                    padding=False  # Don't pad single sequence
                )
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                
                # Set optimized generation parameters
                generation_kwargs = {
                    "max_new_tokens": 128,    # Increased for better responses
                    "temperature": 0.7,       # Balanced temperature
                    "do_sample": True,
                    "top_p": 0.95,           # Slightly higher top_p
                    "top_k": 50,             # Added top_k for better quality
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "use_cache": True,        # Enable KV-cache
                    "num_beams": 1,           # Disable beam search for speed
                    "early_stopping": True,
                    "repetition_penalty": 1.1  # Light penalty to prevent repetition
                }
                
                # Update with user parameters
                generation_kwargs.update(kwargs)
                
                # Generate with optimized settings
                outputs = self.model.generate(
                    inputs,
                    **generation_kwargs
                )
            
                # Decode efficiently
                response = self.tokenizer.decode(
                    outputs[0, inputs.shape[1]:],  # Only decode new tokens
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False  # Faster decoding
                )
            
            return response.strip()

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    def classify_with_logits(self, prompt: str) -> Tuple[str, float]:
        """
        Get classification with confidence from logits.
        
        This method uses the model's logits to determine the most likely
        classification and its confidence score.
        
        Args:
            prompt (str): The input text to classify.
            
        Returns:
            Tuple[str, float]: The predicted class and its confidence score.
            
        Raises:
            Exception: If classification fails.
        """
        try:
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            outputs = self.model(inputs)
            logits = outputs.logits
            
            # Get probabilities for the last token
            probs = torch.softmax(logits[0, -1], dim=0)
            
            # Get the most likely token
            token_id = torch.argmax(probs).item()
            token = self.tokenizer.decode([token_id])
            confidence = probs[token_id].item()
            
            return token, confidence

        except Exception as e:
            logger.error(f"Classification with logits failed: {e}")
            raise

class LLMClassificationService:
    """
    Service for classifying security requests using LLM and ML techniques.
    
    This class provides a comprehensive classification system that combines
    LLM-based analysis with traditional ML methods and pattern matching.
    It includes multiple fallback mechanisms for reliability.
    """

    def __init__(self, model_name="Qwen/Qwen3-0.6B", max_length=512, num_beams=4, 
                 temperature=0.7, top_p=0.9, do_sample=True, no_repeat_ngram_size=3):
        """
        Initialize the classification service.
        
        Args:
            model_name (str): Name of the Hugging Face model to use.
            max_length (int): Maximum sequence length for generation.
            num_beams (int): Number of beams for beam search.
            temperature (float): Sampling temperature for generation.
            top_p (float): Nucleus sampling parameter.
            do_sample (bool): Whether to use sampling.
            no_repeat_ngram_size (int): Size of n-grams to prevent repetition.
        """
        self.analyzer = None
        self._is_trained = False
        self.fallback_model = None
        self.max_length = max_length
        self.num_beams = num_beams
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.no_repeat_ngram_size = no_repeat_ngram_size
        
        # Initialize TF-IDF with default text
        try:
            self.text_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 4),
                min_df=1,
                max_df=0.98,
                analyzer='char_wb',
            )
            
            # Initialize with a more comprehensive set of security terms
            initial_texts = [
                "vendor approval contractor third-party access permission network firewall",
                "data export cloud resource admin sudo root elevated privilege database",
                "query report analytics aws azure gcp port vpn dns proxy ip server",
                "instance ec2 container deploy environment security compliance audit",
                "install setup download docker npm pip maven gradle vscode git java python"
            ]
            
            self.text_vectorizer.fit(initial_texts)
            logger.info("✅ TF-IDF vectorizer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize TF-IDF vectorizer: {e}")
            # Create a minimal vectorizer as fallback
            self.text_vectorizer = TfidfVectorizer(max_features=10)
            self.text_vectorizer.fit(["fallback"])
        
        # Try to initialize the primary LLM
        try:
            logger.info(f"Initializing primary LLM ({model_name})...")
            self.analyzer = LLMAnalyzer(model_name)
            self._is_trained = True
            logger.info("✅ Primary LLM initialized successfully")
        except Exception as primary_error:
            logger.warning(f"Primary LLM initialization failed: {primary_error}")
            
            # Try fallback models in order of preference
            fallback_models = [
                "microsoft/phi-2",  # Smaller but efficient
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Very small
                "facebook/opt-125m"  # Tiny model for extreme fallback
            ]
            
            for fallback_model in fallback_models:
                try:
                    logger.info(f"Trying fallback model: {fallback_model}")
                    self.analyzer = LLMAnalyzer(fallback_model)
                    self._is_trained = True
                    logger.info(f"✅ Fallback LLM ({fallback_model}) initialized successfully")
                    break
                except Exception as fallback_error:
                    logger.warning(f"Fallback model {fallback_model} failed: {fallback_error}")
            
            if not self._is_trained:
                logger.error("❌ All LLM models failed to initialize")
                self._is_trained = False
            
    def classify_request(self, request_text: str) -> Tuple[RequestType, float]:
        """
        Classify a security request using multiple methods.
        
        This method implements a multi-stage classification process:
        1. LLM-based classification with few-shot learning
        2. ML-based classification using TF-IDF and patterns
        3. Pattern-based validation as fallback
        
        Args:
            request_text (str): The request text to classify.
            
        Returns:
            Tuple[RequestType, float]: The classified request type and confidence score.
        """
        if not request_text or not request_text.strip():
            logger.warning("Empty request text, defaulting to PERMISSION_CHANGE")
            return RequestType.PERMISSION_CHANGE, 0.51

        # Try LLM classification first
        if self._is_trained and self.analyzer is not None:
            try:
                # Few-shot examples for better context
                few_shot_examples = """Here are some examples:

Request: "Need to install Docker Desktop for development work"
Category: DEVTOOL_INSTALL

Request: "Open port 443 for external API access"
Category: NETWORK_ACCESS

Request: "Access to AWS EC2 production instance"
Category: CLOUD_RESOURCE_ACCESS

Request: "Export customer data for compliance audit"
Category: DATA_EXPORT

Request: "Onboard new vendor Acme Corp for services"
Category: VENDOR_APPROVAL

Request: "Need admin access to production database"
Category: PERMISSION_CHANGE"""

                # Enhanced prompt with better context and validation rules
                prompt = f"""You are an expert security request classifier. Your task is to classify the request into EXACTLY ONE category.

Classification Rules:
1. VENDOR_APPROVAL: For vendor onboarding, third-party approvals, contractor access
2. PERMISSION_CHANGE: For admin access, elevated permissions, role changes
3. NETWORK_ACCESS: For firewall rules, port access, VPN, network connectivity
4. DATA_EXPORT: For database access, data extraction, report generation
5. CLOUD_RESOURCE_ACCESS: For AWS/Azure/GCP access, cloud servers, VMs
6. DEVTOOL_INSTALL: For development tools, software installation, IDE setup

Important Guidelines:
- Choose the MOST SPECIFIC category that applies
- If request mentions cloud platforms (AWS/Azure/GCP), prefer CLOUD_RESOURCE_ACCESS
- For database access, if it's about data extraction use DATA_EXPORT
- Network changes should use NETWORK_ACCESS unless specifically cloud-related
- Installation requests should use DEVTOOL_INSTALL unless server/cloud related

{few_shot_examples}

Request: "{request_text}"

Step 1 - Identify key elements (respond with ONLY these exact words):
ACCESS: Is it about accessing something? (YES/NO)
CLOUD: Does it mention cloud platforms or services? (YES/NO)
DATA: Is it about data or databases? (YES/NO)
NETWORK: Is it about network or connectivity? (YES/NO)
INSTALL: Is it about installing software? (YES/NO)
VENDOR: Is it about vendors or third parties? (YES/NO)

Step 2 - Category (respond with ONLY the category name):"""

                # Try classification with different parameters
                best_result = None
                best_confidence = 0.0
                
                # Multiple attempts with different settings
                attempts = [
                    (0.1, 0.9),  # Conservative
                    (0.2, 0.85),  # Balanced
                    (0.3, 0.8)    # More exploratory
                ]
                
                for temp, top_p in attempts:
                    try:
                        response = self.analyzer.generate_response(
                            prompt, 
                            max_new_tokens=150,  # Increased for step-by-step response
                            temperature=temp,
                            top_p=top_p,
                            repetition_penalty=1.2
                        )
                        
                        if response and len(response.strip()) > 0:
                            # Parse step-by-step response
                            elements = self._parse_key_elements(response)
                            classification = self._normalize_response(response)
                            
                            if classification:
                                result, base_confidence = self._map_to_request_type(classification, request_text)
                                
                                # Adjust confidence based on key elements
                                adjusted_confidence = self._calculate_adjusted_confidence(
                                    result, base_confidence, elements, request_text
                                )
                                
                                if adjusted_confidence > best_confidence:
                                    best_result = (result, adjusted_confidence)
                                    best_confidence = adjusted_confidence
                                    
                                    # If we have high confidence, return immediately
                                    if adjusted_confidence >= 0.85:
                                        logger.info(f"✅ High confidence LLM classification: {result.value}")
                                        return result, adjusted_confidence
                    
                    except Exception as e:
                        logger.warning(f"LLM attempt failed: {e}")
                        continue
                
                # Return best result if confidence is reasonable
                if best_result and best_result[1] >= 0.7:
                    logger.info(f"✅ LLM classification successful: {best_result[0].value}")
                    return best_result
                
            except Exception as e:
                logger.warning(f"LLM classification failed, falling back to ML: {e}")
        
        # Fallback to ML classification
        try:
            ml_result = self._ml_classification(request_text)
            if ml_result and ml_result[1] >= 0.6:
                logger.info("✅ ML classification successful")
                return ml_result
        except Exception as e:
            logger.warning(f"ML classification failed: {e}")
        
        # Final fallback to pattern matching
        try:
            pattern_result = self._pattern_based_validation(request_text)
            if pattern_result and pattern_result[1] > 0:
                logger.info("✅ Pattern matching successful")
                return pattern_result
        except Exception as e:
            logger.warning(f"Pattern matching failed: {e}")
        
        # Ultimate fallback
        logger.warning("All classification methods failed, using default")
        return RequestType.PERMISSION_CHANGE, 0.51

    def _parse_key_elements(self, response: str) -> Dict[str, bool]:
        """
        Parse key elements from the LLM response.
        
        Args:
            response (str): The LLM response to parse.
            
        Returns:
            Dict[str, bool]: Dictionary of key elements and their presence.
        """
        elements = {
            'ACCESS': False,
            'CLOUD': False,
            'DATA': False,
            'NETWORK': False,
            'INSTALL': False,
            'VENDOR': False
        }
        
        try:
            # Extract the part between Step 1 and Step 2
            step1_match = re.search(r'Step 1.*?Step 2', response, re.DOTALL)
            if step1_match:
                step1_text = step1_match.group(0)
                
                # Parse each element
                for element in elements.keys():
                    if re.search(rf'{element}:.*?YES', step1_text, re.IGNORECASE):
                        elements[element] = True
        except Exception:
            pass
        
        return elements

    def _calculate_adjusted_confidence(self, result: RequestType, base_confidence: float,
                                    elements: Dict[str, bool], request_text: str) -> float:
        """
        Calculate adjusted confidence based on key elements and context.
        
        Args:
            result (RequestType): The classified request type.
            base_confidence (float): Initial confidence score.
            elements (Dict[str, bool]): Key elements from parsing.
            request_text (str): Original request text.
            
        Returns:
            float: Adjusted confidence score.
        """
        
        # Start with base confidence
        confidence = base_confidence
        
        # Adjust based on key elements matching
        element_rules = {
            RequestType.CLOUD_RESOURCE_ACCESS: {
                'required': ['CLOUD'],
                'supporting': ['ACCESS'],
                'conflicting': ['INSTALL']
            },
            RequestType.DATA_EXPORT: {
                'required': ['DATA'],
                'supporting': ['ACCESS', 'EXPORT'],
                'conflicting': ['INSTALL', 'NETWORK']
            },
            RequestType.NETWORK_ACCESS: {
                'required': ['NETWORK'],
                'supporting': ['ACCESS'],
                'conflicting': ['INSTALL', 'DATA']
            },
            RequestType.PERMISSION_CHANGE: {
                'required': ['ACCESS'],
                'supporting': [],
                'conflicting': ['INSTALL', 'VENDOR']
            },
            RequestType.VENDOR_APPROVAL: {
                'required': ['VENDOR'],
                'supporting': ['ACCESS'],
                'conflicting': ['INSTALL', 'DATA']
            },
            RequestType.DEVTOOL_INSTALL: {
                'required': ['INSTALL'],
                'supporting': [],
                'conflicting': ['CLOUD', 'NETWORK']
            }
        }
        
        if result in element_rules:
            rules = element_rules[result]
            
            # Check required elements
            required_present = all(elements.get(elem, False) for elem in rules['required'])
            if not required_present:
                confidence *= 0.5
            
            # Check supporting elements
            supporting_present = any(elements.get(elem, False) for elem in rules['supporting'])
            if supporting_present:
                confidence = min(0.95, confidence * 1.2)
            
            # Check conflicting elements
            conflicting_present = any(elements.get(elem, False) for elem in rules['conflicting'])
            if conflicting_present:
                confidence *= 0.7
        
        # Additional context-based adjustments
        text_lower = request_text.lower()
        
        # Cloud platform mentions
        cloud_platforms = ['aws', 'azure', 'gcp', 'cloud']
        if result == RequestType.CLOUD_RESOURCE_ACCESS and any(p in text_lower for p in cloud_platforms):
            confidence = min(0.95, confidence * 1.2)
        
        # Network-specific terms
        network_terms = ['port', 'firewall', 'vpn', 'network']
        if result == RequestType.NETWORK_ACCESS and any(t in text_lower for t in network_terms):
            confidence = min(0.95, confidence * 1.2)
        
        # Development tool terms
        dev_tools = ['docker', 'vscode', 'git', 'npm', 'python', 'java']
        if result == RequestType.DEVTOOL_INSTALL and any(t in text_lower for t in dev_tools):
            confidence = min(0.95, confidence * 1.2)
        
        # Cap confidence
        return min(0.95, max(0.51, confidence))

    def _ml_classification(self, request_text: str) -> Tuple[Optional[RequestType], float]:
        """
        Perform ML-based classification using TF-IDF and patterns.
        
        Args:
            request_text (str): The request text to classify.
            
        Returns:
            Tuple[Optional[RequestType], float]: Classification result and confidence.
        """
        try:
            # Extract features
            features = self._create_feature_vector(request_text)
            text_lower = request_text.lower()
            
            # Define category-specific patterns and weights
            category_patterns = {
                RequestType.DATA_EXPORT: {
                    'high': [
                        (r'\b(export|extract)\s+(data|database|records?|files?)\b', 0.8),
                        (r'\b(query|access)\s+(database|data)\b', 0.7),
                        (r'\b(generate|create)\s+(report|analytics)\b', 0.7),
                        (r'\b(csv|excel|json|xml)\b', 0.6),
                        (r'\bs3\s+bucket\b', 0.6)
                    ],
                    'terms': ['export', 'data', 'database', 'query', 'report', 'analytics', 'records', 'files']
                },
                RequestType.NETWORK_ACCESS: {
                    'high': [
                        (r'\b(firewall|port)\s+(rule|access|change)\b', 0.8),
                        (r'\bport\s+\d+\b', 0.8),
                        (r'\b(vpn|network)\s+access\b', 0.7),
                        (r'\b(dns|proxy|ip)\s+(change|update)\b', 0.7)
                    ],
                    'terms': ['firewall', 'port', 'vpn', 'network', 'dns', 'proxy', 'ip']
                },
                RequestType.CLOUD_RESOURCE_ACCESS: {
                    'high': [
                        (r'\b(aws|azure|gcp)\s+(instance|resource)\b', 0.8),
                        (r'\b(ec2|vm|container)\s+access\b', 0.8),
                        (r'\b(cloud|server)\s+(access|permission)\b', 0.7),
                        (r'\b(production|staging)\s+(server|environment)\b', 0.7)
                    ],
                    'terms': ['cloud', 'aws', 'azure', 'gcp', 'server', 'instance', 'ec2', 'vm']
                },
                RequestType.VENDOR_APPROVAL: {
                    'high': [
                        (r'\b(vendor|contractor|third[-\s]party)\s+(approval|access)\b', 0.8),
                        (r'\bonboard\s+(vendor|contractor)\b', 0.8),
                        (r'\bexternal\s+(access|party)\b', 0.7)
                    ],
                    'terms': ['vendor', 'contractor', 'third-party', 'external', 'onboard']
                },
                RequestType.PERMISSION_CHANGE: {
                    'high': [
                        (r'\b(admin|root|sudo)\s+access\b', 0.8),
                        (r'\belevated\s+(permission|access|privilege)\b', 0.8),
                        (r'\b(permission|access)\s+change\b', 0.7),
                        (r'\brole\s+(change|update)\b', 0.7)
                    ],
                    'terms': ['admin', 'permission', 'access', 'privilege', 'role', 'elevated']
                }
            }
            
            # Calculate scores using patterns and term frequency
            scores = {}
            for req_type, patterns in category_patterns.items():
                score = 0.0
                
                # Pattern matching score
                for pattern, weight in patterns['high']:
                    if re.search(pattern, text_lower):
                        score += weight
                
                # Term frequency score
                term_count = sum(text_lower.count(term) for term in patterns['terms'])
                score += min(0.5, term_count * 0.1)  # Cap term frequency score at 0.5
                
                # Context score based on surrounding words
                context_score = 0.0
                for term in patterns['terms']:
                    if term in text_lower:
                        # Look at words around the term
                        words = text_lower.split()
                        try:
                            idx = next(i for i, w in enumerate(words) if term in w)
                            context = words[max(0, idx-2):min(len(words), idx+3)]
                            context_score += sum(1 for w in context if w in patterns['terms']) * 0.1
                        except StopIteration:
                            pass
                
                score += min(0.3, context_score)  # Cap context score at 0.3
                scores[req_type] = score
            
            # Get the highest scoring category
            best_type = max(scores.items(), key=lambda x: x[1])
            
            # Calculate confidence based on score difference and absolute score
            sorted_scores = sorted(scores.values(), reverse=True)
            if len(sorted_scores) > 1:
                score_diff = sorted_scores[0] - sorted_scores[1]
                base_confidence = 0.6 + (score_diff * 0.3)  # Score difference contribution
                absolute_confidence = min(0.2, sorted_scores[0] * 0.2)  # Absolute score contribution
                confidence = min(0.95, base_confidence + absolute_confidence)
            else:
                confidence = 0.6
            
            # Only return if we have a reasonable score
            if best_type[1] >= 0.5:  # Increased threshold from 0
                return best_type[0], confidence
            
            return None, 0.0
            
        except Exception as e:
            logger.error(f"ML classification failed: {e}")
            return None, 0.0

    def _create_feature_vector(self, request_text: str) -> np.ndarray:
        """
        Create feature vector for ML classification.
        
        Args:
            request_text (str): The request text to process.
            
        Returns:
            np.ndarray: Feature vector for classification.
        """
        try:
            # Text features using TF-IDF
            text_features = self.text_vectorizer.transform([request_text]).toarray()[0]
            text_lower = request_text.lower()
            
            # Enhanced security keyword features with context
            security_keywords = {
                'data_operations': [
                    'export', 'extract', 'query', 'report', 'analytics', 'database',
                    'records', 'files', 'csv', 'excel', 'json', 'xml', 's3', 'bucket'
                ],
                'network': [
                    'firewall', 'port', 'vpn', 'network', 'dns', 'proxy', 'ip',
                    'subnet', 'routing', 'traffic', 'bandwidth', 'protocol'
                ],
                'cloud': [
                    'aws', 'azure', 'gcp', 'cloud', 'server', 'instance', 'ec2',
                    'vm', 'container', 'kubernetes', 'docker', 'deploy'
                ],
                'access_control': [
                    'admin', 'root', 'sudo', 'permission', 'privilege', 'role',
                    'grant', 'revoke', 'elevated', 'access', 'authorization'
                ],
                'vendor': [
                    'vendor', 'contractor', 'third-party', 'external', 'partner',
                    'supplier', 'company', 'organization', 'client', 'onboard'
                ],
                'security_terms': [
                    'security', 'compliance', 'audit', 'policy', 'restriction',
                    'sensitive', 'confidential', 'protected', 'encrypted'
                ],
                'time_sensitivity': [
                    'urgent', 'immediate', 'asap', 'emergency', 'critical',
                    'temporary', 'permanent', 'duration', 'period'
                ]
            }
            
            # Calculate weighted keyword features
            keyword_features = []
            for category, keywords in security_keywords.items():
                # Basic count
                count = sum(1 for keyword in keywords if keyword in text_lower)
                
                # Context-aware count (words appearing near security terms)
                context_count = 0
                words = text_lower.split()
                for i, word in enumerate(words):
                    if word in keywords:
                        context = words[max(0, i-2):min(len(words), i+3)]
                        context_count += sum(1 for w in context if any(w in kw for kw in security_keywords.values()))
                
                keyword_features.extend([count, min(5, context_count)])
            
            # Enhanced text complexity features
            word_count = len(request_text.split())
            avg_word_length = len(request_text) / max(1, word_count)
            special_chars = sum(1 for c in request_text if not c.isalnum() and not c.isspace())
            
            complexity_features = [
                word_count,
                avg_word_length,
                special_chars,
                len(re.findall(r'\d+', request_text)),  # Number count
                len(re.findall(r'[A-Z][a-z]+', request_text)),  # Proper noun count
                sum(1 for c in request_text if c.isupper()) / max(1, len(request_text)),
                len(re.findall(r'[!?]', request_text)),  # Urgency indicators
                len(re.findall(r'\b(please|urgent|asap)\b', text_lower))  # Request strength
            ]
            
            # Combine all features with appropriate scaling
            all_features = np.concatenate([
                text_features,
                np.array(keyword_features) / max(1, max(keyword_features)),
                np.array(complexity_features) / max(1, max(complexity_features))
            ])
            
            return all_features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return np.zeros(1)  # Return empty feature vector on error

    def _normalize_response(self, response: str) -> str:
        """
        Normalize LLM response to standard format.
        
        Args:
            response (str): Raw LLM response.
            
        Returns:
            str: Normalized category name.
        """
        if not response or not response.strip():
            logger.debug("Empty response received")
            return ""
            
        # Convert to lowercase and strip whitespace
        clean = response.lower().strip()
        
        # Define valid categories and common variations
        valid_categories = {
            'vendor_approval': ['vendor approval', 'vendorapproval', 'vendor-approval'],
            'permission_change': ['permission change', 'permissionchange', 'permission-change'],
            'network_access': ['network access', 'networkaccess', 'network-access'],
            'data_export': ['data export', 'dataexport', 'data-export'],
            'cloud_resource_access': ['cloud resource access', 'cloudresourceaccess', 'cloud-resource-access']
        }
        
        # First remove any markdown formatting
        clean = re.sub(r'[`*_#]', '', clean)
        
        # Remove common prefixes that shouldn't be there
        prefixes = [
            'answer:', 'answer is', 'the answer is',
            'classification:', 'category:', 'type:',
            'response:', 'class:', 'result:',
            'the classification is', 'i choose', 'i select',
            'it is', 'this is'
        ]
        for prefix in prefixes:
            if clean.startswith(prefix):
                clean = clean[len(prefix):].strip()
        
        # Remove everything after certain characters
        clean = re.sub(r'[\.\n,:].*$', '', clean)
        
        # Standardize spacing and remove special characters
        clean = re.sub(r'\s+', ' ', clean)  # Normalize spaces
        clean = clean.strip()
        
        # First try exact match with valid categories
        clean_underscore = clean.replace(' ', '_').replace('-', '_')
        if clean_underscore in valid_categories:
            return clean_underscore
            
        # Try matching with variations
        for category, variations in valid_categories.items():
            if clean in variations or clean_underscore in variations:
                return category
            
        # Try matching after removing all spaces and special characters
        clean_compact = re.sub(r'[^a-z]', '', clean)
        for category, variations in valid_categories.items():
            compact_variations = [re.sub(r'[^a-z]', '', v) for v in variations]
            if clean_compact in compact_variations:
                return category
                
        # If we still don't have a match, check if any valid category appears in the text
        for category in valid_categories:
            if category.replace('_', '') in clean_compact:
                return category
        
        # If no valid category is found, return empty string
        logger.debug(f"Could not normalize response '{response}' to a valid category")
        return ""

    def _map_to_request_type(self, classification: str, request_text: str) -> Tuple[RequestType, float]:
        """
        Map normalized classification to RequestType with confidence.
        
        Args:
            classification (str): Normalized classification.
            request_text (str): Original request text.
            
        Returns:
            Tuple[RequestType, float]: Mapped request type and confidence.
        """
        
        # Direct mapping with common variations
        mapping = {
            # Vendor approval variations
            "vendor_approval": (RequestType.VENDOR_APPROVAL, 0.9),
            "vendorapproval": (RequestType.VENDOR_APPROVAL, 0.9),
            "vendor": (RequestType.VENDOR_APPROVAL, 0.85),
            "contractor": (RequestType.VENDOR_APPROVAL, 0.85),
            "thirdparty": (RequestType.VENDOR_APPROVAL, 0.85),
            
            # Permission change variations
            "permission_change": (RequestType.PERMISSION_CHANGE, 0.9),
            "permissionchange": (RequestType.PERMISSION_CHANGE, 0.9),
            "admin": (RequestType.PERMISSION_CHANGE, 0.85),
            "adminaccess": (RequestType.PERMISSION_CHANGE, 0.9),
            "elevated": (RequestType.PERMISSION_CHANGE, 0.85),
            
            # Network access variations
            "network_access": (RequestType.NETWORK_ACCESS, 0.9),
            "networkaccess": (RequestType.NETWORK_ACCESS, 0.9),
            "firewall": (RequestType.NETWORK_ACCESS, 0.85),
            "port": (RequestType.NETWORK_ACCESS, 0.85),
            "vpn": (RequestType.NETWORK_ACCESS, 0.85),
            
            # Data export variations
            "data_export": (RequestType.DATA_EXPORT, 0.9),
            "dataexport": (RequestType.DATA_EXPORT, 0.9),
            "database": (RequestType.DATA_EXPORT, 0.85),
            "data": (RequestType.DATA_EXPORT, 0.8),
            
            # Cloud resource variations
            "cloud_resource_access": (RequestType.CLOUD_RESOURCE_ACCESS, 0.9),
            "cloudresourceaccess": (RequestType.CLOUD_RESOURCE_ACCESS, 0.9),
            "cloud": (RequestType.CLOUD_RESOURCE_ACCESS, 0.85),
            "server": (RequestType.CLOUD_RESOURCE_ACCESS, 0.85),
            "ssh": (RequestType.CLOUD_RESOURCE_ACCESS, 0.85),
        }

        # Try direct mapping first
        if classification in mapping:
            return mapping[classification]

        # If no direct match, try pattern matching with confidence adjustment
        text_lower = request_text.lower()

        patterns = [
            (r'\b(onboard|approve)\s+(vendor|contractor|third[-\s]party)', RequestType.VENDOR_APPROVAL, 0.85),
            (r'\b(admin|root|sudo|elevated)\s+(access|permission)', RequestType.PERMISSION_CHANGE, 0.85),
            (r'\b(firewall|port|vpn)\b', RequestType.NETWORK_ACCESS, 0.85),
            (r'\b(database|data)\s+(access|export)', RequestType.DATA_EXPORT, 0.85),
            (r'\b(server|cloud|aws|azure|gcp)\b', RequestType.CLOUD_RESOURCE_ACCESS, 0.85)
        ]

    def _pattern_based_validation(self, request_text: str) -> Tuple[Optional[RequestType], float]:
        """
        Validate classification using pattern matching.
        
        Args:
            request_text (str): The request text to validate.
            
        Returns:
            Tuple[Optional[RequestType], float]: Validated type and confidence.
        """
        text_lower = request_text.lower()
        
        # Define patterns for each request type
        patterns = {
            RequestType.DEVTOOL_INSTALL: [
                r'\b(install|setup|download)\s+(vscode|visual\s*studio|extension|docker|java|jdk|sdk|tool)\b',
                r'\bsetup.*development\b',
                r'\bdownload.*(tool|software|package)\b'
            ],
            RequestType.PERMISSION_CHANGE: [
                r'\btemporary\s+admin\s+access\b',
                r'\badmin\s+access\b',
                r'\belevated\s+permission\b'
            ],
            RequestType.NETWORK_ACCESS: [
                r'\bopen\s+port\s+\d+\b',
                r'\bvpn\s+access\b',
                r'\bnetwork\s+connectivity\b'
            ]
        }
        
        # Check patterns
        for req_type, type_patterns in patterns.items():
            for pattern in type_patterns:
                if re.search(pattern, text_lower):
                    logger.debug(f"Pattern match: {pattern} → {req_type.value}")
                    return req_type, confidence
        
        return None, 0.0

    def _enhanced_pattern_fallback(self, request_text: str) -> Tuple[RequestType, float]:
        """
        Enhanced pattern-based fallback classification.
        
        Args:
            request_text (str): The request text to classify.
            
        Returns:
            Tuple[RequestType, float]: Fallback classification and confidence.
        """
        text_lower = request_text.lower()
        
        # Comprehensive pattern matching with confidence scores
        patterns = {
            RequestType.DATA_EXPORT: {
                'patterns': [
                    (r'\b(export|extract)\s+(data|file|records?)\b', 0.95),
                    (r'\b(data|file|records?)\s+(export|extraction)\b', 0.95),
                    (r'\b(database|data)\s+(access|query)\b', 0.90),
                    (r'\b(s3|bucket|storage)\s+(upload|access|transfer)\b', 0.90),
                    (r'\b\d+\s*(mb|gb|tb)\b.*\b(data|file|record)', 0.90),
                    (r'\b(analyze|query|report)\s+data\b', 0.85),
                    (r'\b(csv|json|excel|database)\s+(export|dump|backup)\b', 0.85)
                ],
                'keywords': ['export', 'data', 'database', 'query', 'sql', 'report', 'analytics', 
                           's3', 'bucket', 'dump', 'backup', 'extract', 'csv', 'json', 'excel']
            },
            RequestType.VENDOR_APPROVAL: {
                'patterns': [
                    (r'\b(onboard|approve)\s+(vendor|contractor|third[-\s]party|partner)\b', 0.95),
                    (r'\bvendor\s+(access|approval|request)\b', 0.90),
                    (r'\bexternal\s+(access|party|entity)\b', 0.85),
                    (r'\b(contractor|supplier|partner)\s+(access|request)\b', 0.85)
                ],
                'keywords': ['vendor', 'contractor', 'third-party', 'supplier', 'partner', 'external', 'onboard']
            },
            RequestType.PERMISSION_CHANGE: {
                'patterns': [
                    (r'\b(admin|root|sudo)\s+(access|permission)\b', 0.95),
                    (r'\belevated\s+(permission|access|privilege)\b', 0.95),
                    (r'\b(permission|access)\s+(change|modify|update)\b', 0.90),
                    (r'\brole\s+(change|update|modify)\b', 0.90)
                ],
                'keywords': ['admin', 'permission', 'privilege', 'escalate', 'role', 'rights', 'grant']
            },
            RequestType.NETWORK_ACCESS: {
                'patterns': [
                    (r'\b(firewall|port)\s+(access|change|rule)\b', 0.95),
                    (r'\bvpn\s+access\b', 0.95),
                    (r'\bnetwork\s+(access|configuration)\b', 0.90),
                    (r'\b(open|allow)\s+port\b', 0.90)
                ],
                'keywords': ['firewall', 'port', 'vpn', 'network', 'connection', 'dns', 'proxy']
            },
            RequestType.CLOUD_RESOURCE_ACCESS: {
                'patterns': [
                    (r'\b(server|instance)\s+access\b', 0.95),
                    (r'\b(cloud|aws|azure|gcp)\s+(resource|access)\b', 0.95),
                    (r'\b(system|infrastructure)\s+access\b', 0.90),
                    (r'\b(production|staging)\s+(server|environment)\b', 0.90)
                ],
                'keywords': ['server', 'cloud', 'aws', 'azure', 'instance', 'ssh', 'system']
            }
        }

        best_type = None
        best_confidence = 0.0

        # Check each request type
        for req_type, matchers in patterns.items():
            # Check regex patterns
            for pattern, confidence in matchers['patterns']:
                if re.search(pattern, text_lower):
                    if confidence > best_confidence:
                        best_type = req_type
                        best_confidence = confidence
            
            # Check keywords if no high-confidence match found
            if best_confidence < 0.9:
                keyword_matches = sum(1 for keyword in matchers['keywords'] if keyword in text_lower)
                if keyword_matches > 0:
                    confidence = min(0.85, 0.6 + (keyword_matches * 0.05))
                    if confidence > best_confidence:
                        best_type = req_type
                        best_confidence = confidence

        # If still no good match, try to infer from context
        if best_type is None:
            # Check for data-related terms
            data_terms = ['data', 'file', 'record', 'report', 'information', 'content', 'document']
            if any(term in text_lower for term in data_terms):
                best_type = RequestType.DATA_EXPORT
                best_confidence = 0.7
            # Check for system access terms
            elif any(term in text_lower for term in ['access', 'login', 'account', 'credential']):
                best_type = RequestType.PERMISSION_CHANGE
                best_confidence = 0.6
            else:
                # Default to PERMISSION_CHANGE as it's the most general security request type
                best_type = RequestType.PERMISSION_CHANGE
                best_confidence = 0.51

        return best_type, best_confidence

    def train(self, historical_data):
        """
        Compatibility method for training interface.
        
        Note: This is a no-op as the LLM-based system doesn't require training.
        """
        logger.info("✅ LLM classification ready - no training needed!")
        self._is_trained = True

# Simple test function
def test_fixed_llm():
    """
    Test the fixed LLM implementation.
    
    This function runs a series of test cases to verify the LLM
    classification functionality.
    """
    try:
        service = LLMClassificationService()

        test_cases = [
            "Onboard vendor Harmon White for campaign analytics",
            "Need admin access to production database",
            "Open port 443 for external API access"
        ]

        for test in test_cases:
            result = service.classify_request(test)
            logger.info(f"Test: '{test[:40]}...' → {result[0].value} ({result[1]:.2f})")

    except Exception as e:
        logger.error(f"Test failed: {e}")

if __name__ == "__main__":
    test_fixed_llm()
