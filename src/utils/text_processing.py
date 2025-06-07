# src/utils/text_processing.py - ENHANCED VERSION WITH INTEGRATED FIXES

import re
import logging
from typing import List, Dict, Any
from src.models.enums import RequestType

logger = logging.getLogger(__name__)

try:
    from ..services.LLM_services import LLMTextProcessor
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    logger.warning("LLM not available - using enhanced regex processing")


class TextProcessor:
    """Enhanced Text Processor with integrated ML performance fixes"""
    
    @staticmethod
    def clean_slack_text(text: str) -> str:
        """Clean Slack-specific formatting from text"""
        if not text:
            return ""

        # Remove user mentions
        text = re.sub(r'<@[A-Z0-9]+>', '', text)

        # Remove channel mentions
        text = re.sub(r'<#[A-Z0-9]+\|[^>]+>', '', text)

        # Clean mailto links but preserve the email
        text = re.sub(r'<mailto:([^|>]+)\|[^>]+>', r'\1', text)

        # Remove other URLs
        text = re.sub(r'<http[^>]+>', '', text)

        # Remove special formatting
        text = re.sub(r'[*_~`]', '', text)

        # Clean up whitespace
        text = ' '.join(text.split())

        return text.strip()

    @staticmethod
    def extract_email_from_slack_format(text: str) -> str:
        """Extract email from Slack's mailto format"""
        if not text:
            return text

        # Handle Slack mailto format: <mailto:email@domain.com|email@domain.com>
        mailto_match = re.search(r'<mailto:([^|>]+)\|[^>]+>', text)
        if mailto_match:
            return mailto_match.group(1)

        # Return original text if no mailto format found
        return text

    @staticmethod
    def parse_follow_up_answers(text: str, expected_fields: List[str]) -> Dict[str, str]:
        """ENHANCED: Parse follow-up answers with integrated ML fixes"""

        answers = {}

        if not text or not expected_fields:
            logger.debug("Empty text or no expected fields, returning empty dict")
            return answers

        try:
            # Clean the text first
            cleaned_text = TextProcessor.clean_slack_text(text)
            logger.debug(f"Cleaned text: '{cleaned_text}'")
            logger.debug(f"Expected fields: {expected_fields}")

            if not cleaned_text.strip():
                logger.debug("Cleaned text is empty, returning empty dict")
                return answers

            # INTEGRATED FIX: Enhanced parsing with request type awareness
            answers = TextProcessor._enhanced_follow_up_parsing(cleaned_text, expected_fields)
            
            if answers:
                logger.debug(f"Enhanced parsing successful: {answers}")
                return answers

            # TRY LLM PARSING if available
            if LLM_AVAILABLE:
                try:
                    llm_processor = LLMTextProcessor()
                    llm_result = llm_processor.parse_follow_up_answers(cleaned_text, expected_fields)

                    if llm_result and len(llm_result) >= len(expected_fields) * 0.75:
                        logger.info(f"✅ LLM successfully parsed {len(llm_result)}/{len(expected_fields)} fields")
                        return llm_result
                    else:
                        logger.info(f"LLM parsed {len(llm_result)}/{len(expected_fields)} fields, trying fallback")

                except Exception as e:
                    logger.warning(f"LLM parsing failed: {e}, using enhanced fallback")

            # Enhanced line-by-line parsing
            lines = [line.strip() for line in cleaned_text.split('\n') if line.strip()]
            logger.debug(f"Split into {len(lines)} lines: {lines}")

            if len(lines) >= len(expected_fields):
                for i, field in enumerate(expected_fields):
                    if i < len(lines) and lines[i].strip():
                        value = lines[i].strip()
                        if 'email' in field.lower() or 'approval' in field.lower():
                            value = TextProcessor.extract_email_from_slack_format(value)
                        answers[field] = value
                        logger.debug(f"Line method: {field} = {value}")

                if len(answers) == len(expected_fields):
                    logger.debug(f"Line parsing successful: {answers}")
                    return answers

            # Enhanced word-by-word assignment for single line responses
            if len(lines) == 1:
                words = cleaned_text.split()
                logger.debug(f"Single line with {len(words)} words: {words}")

                if len(words) >= len(expected_fields):
                    for i, field in enumerate(expected_fields):
                        if i < len(words) and words[i].strip():
                            value = words[i].strip()
                            if 'email' in field.lower() or 'approval' in field.lower():
                                value = TextProcessor.extract_email_from_slack_format(value)
                            answers[field] = value
                            logger.debug(f"Word method: {field} = {value}")

            logger.debug(f"Final parsed answers: {answers}")
            return answers

        except Exception as e:
            logger.error(f"Error parsing follow-up answers: {e}")
            logger.error(f"Text: '{text}', Fields: {expected_fields}")
            return {}

    @staticmethod
    def _enhanced_follow_up_parsing(text: str, expected_fields: List[str]) -> Dict[str, str]:
        """INTEGRATED FIX: Enhanced follow-up parsing with request type awareness"""
        
        answers = {}
        text_lower = text.lower()
        
        # INTEGRATED FIX: Context-aware parsing based on field types
        field_patterns = {
            # Network access fields
            'destination': [
                r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b',  # IP address
                r'\b([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b',  # Domain
                r'(?:to|destination|server|host)\s+([^\s,]+)'
            ],
            'port': [
                r'\b(22|80|443|3306|5432|8080|9000|3389)\b',  # Common ports
                r'port\s+(\d{1,5})',
                r'(\d{1,5})(?:\s*$|[^\d])'
            ],
            'protocol': [
                r'\b(ssh|tcp|udp|http|https|ftp|smtp|rdp)\b'
            ],
            'business_justification': [
                r'(?:for|because|need|purpose)[\s:]+([^.!?\n]+)',
                r'(?:to|in order to)\s+([^.!?\n]+)'
            ],
            
            # Permission change fields
            'target_system': [
                r'\b([A-Z][a-zA-Z0-9\s]*(?:server|system|database))\b',
                r'\b(production|staging|development|aws|azure)\b'
            ],
            'permission_level': [
                r'\b(admin|administrator|root|sudo|read-only|write|full)\b'
            ],
            'duration': [
                r'\b(\d+\s*(?:hours?|days?|weeks?|months?))\b',
                r'\b(temporary|permanent|indefinite)\b'
            ],
            'manager_approval': [
                r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
            ],
            
            # Data export fields
            'dataset_name': [
                r'\b([a-zA-Z][a-zA-Z0-9_-]*(?:logs?|db|database|data))\b',
                r'(?:dataset|table|database)\s+([a-zA-Z][a-zA-Z0-9_-]*)'
            ],
            'access_level': [
                r'\b(read-only|readonly|read|write|admin|full)\b'
            ],
            'data_classification': [
                r'\b(public|internal|confidential|restricted|secret)\b'
            ],
            
            # Cloud resource fields
            'cloud_provider': [
                r'\b(aws|azure|gcp|google\s+cloud)\b'
            ],
            'resource_type': [
                r'\b(ec2|s3|lambda|rds|vm|virtual\s+machine|instance)\b'
            ],
            
            # DevTool fields
            'tool_name': [
                r'\b(docker|vscode|visual\s+studio|npm|python|java|git|maven)\b'
            ],
            'version': [
                r'\b(latest|\d+\.\d+|\d+)\b'
            ],
            
            # Vendor fields
            'vendor_name': [
                r'\b([A-Z][a-zA-Z\s]+(?:Inc|LLC|Corp|Corporation|Ltd)?)\b'
            ],
            'service_description': [
                r'(?:service|platform|tool|software)[\s:]+([^.!?\n]+)'
            ]
        }
        
        # Pattern-based extraction for each expected field
        for field in expected_fields:
            if field in field_patterns:
                for pattern in field_patterns[field]:
                    match = re.search(pattern, text if 'email' in field else text_lower, re.IGNORECASE)
                    if match:
                        value = match.group(1).strip()
                        if value and field not in answers:
                            # Special handling for email fields
                            if 'email' in field.lower() or 'approval' in field.lower():
                                value = TextProcessor.extract_email_from_slack_format(value)
                            answers[field] = value
                            logger.debug(f"Pattern match for {field}: {value}")
                            break
        
        # If we got most fields through patterns, return
        if len(answers) >= len(expected_fields) * 0.7:
            return answers
        
        # Enhanced intelligent parsing for remaining fields
        remaining_fields = [f for f in expected_fields if f not in answers]
        if remaining_fields:
            intelligent_answers = TextProcessor._intelligent_parse_by_context(text, remaining_fields)
            answers.update(intelligent_answers)
        
        return answers

    @staticmethod
    def _intelligent_parse_by_context(text: str, expected_fields: List[str]) -> Dict[str, str]:
        """INTEGRATED FIX: Intelligent parsing based on request context"""
        
        answers = {}
        text_lower = text.lower()
        
        # Determine request context from field types
        if any(field in ['destination', 'port', 'protocol'] for field in expected_fields):
            # Network access context
            return TextProcessor._parse_network_request(text, expected_fields)
        elif any(field in ['target_system', 'permission_level'] for field in expected_fields):
            # Permission change context
            return TextProcessor._parse_permission_request(text, expected_fields)
        elif any(field in ['dataset_name', 'access_level'] for field in expected_fields):
            # Data access context
            return TextProcessor._parse_data_request(text, expected_fields)
        elif any(field in ['vendor_name', 'service_description'] for field in expected_fields):
            # Vendor approval context
            return TextProcessor._parse_vendor_request(text, expected_fields)
        elif any(field in ['tool_name', 'version'] for field in expected_fields):
            # DevTool install context
            return TextProcessor._parse_devtool_request(text, expected_fields)
        elif any(field in ['cloud_provider', 'resource_type'] for field in expected_fields):
            # Cloud resource context
            return TextProcessor._parse_cloud_request(text, expected_fields)
        
        return answers

    @staticmethod
    def _parse_devtool_request(text: str, expected_fields: List[str]) -> Dict[str, str]:
        """INTEGRATED FIX: Parse DevTool installation requests"""
        
        answers = {}
        text_lower = text.lower()
        
        patterns = {
            'tool_name': [
                r'\b(docker|vscode|visual\s+studio|npm|node|python|java|git|maven|gradle|eclipse|intellij)\b',
                r'install\s+([a-zA-Z][a-zA-Z0-9\s]*)',
                r'setup\s+([a-zA-Z][a-zA-Z0-9\s]*)'
            ],
            'version': [
                r'version\s+(\d+\.\d+|\d+|latest)',
                r'v(\d+\.\d+|\d+)',
                r'\b(latest|stable|lts)\b'
            ],
            'purpose': [
                r'for\s+([a-zA-Z][a-zA-Z0-9\s]*(?:development|programming|coding))',
                r'(?:development|programming|coding|building)\s+([^.!?\n]*)'
            ]
        }
        
        for field in expected_fields:
            if field in patterns:
                for pattern in patterns[field]:
                    match = re.search(pattern, text_lower)
                    if match:
                        value = match.group(1).strip()
                        if value:
                            answers[field] = value
                            break
        
        return answers

    @staticmethod
    def _parse_vendor_request(text: str, expected_fields: List[str]) -> Dict[str, str]:
        """INTEGRATED FIX: Parse vendor approval requests"""
        
        answers = {}
        text_lower = text.lower()
        
        patterns = {
            'vendor_name': [
                r'vendor\s+([A-Z][a-zA-Z\s]+)',
                r'onboard\s+([A-Z][a-zA-Z\s]+)',
                r'approve\s+([A-Z][a-zA-Z\s]+)',
                r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b'
            ],
            'service_description': [
                r'for\s+([a-zA-Z][a-zA-Z0-9\s]*(?:service|platform|solution))',
                r'providing\s+([^.!?\n]*)',
                r'service[s]?\s+([^.!?\n]*)'
            ],
            'data_types': [
                r'data\s+([a-zA-Z][a-zA-Z0-9\s]*)',
                r'access\s+to\s+([a-zA-Z][a-zA-Z0-9\s]*(?:data|information))'
            ],
            'contract_duration': [
                r'(\d+\s*(?:months?|years?|days?))',
                r'duration\s+(\d+\s*(?:months?|years?))',
                r'contract\s+(\d+\s*(?:months?|years?))'
            ]
        }
        
        for field in expected_fields:
            if field in patterns:
                for pattern in patterns[field]:
                    match = re.search(pattern, text if field == 'vendor_name' else text_lower)
                    if match:
                        value = match.group(1).strip()
                        if value:
                            answers[field] = value
                            break
        
        return answers

    @staticmethod
    def _parse_cloud_request(text: str, expected_fields: List[str]) -> Dict[str, str]:
        """INTEGRATED FIX: Parse cloud resource access requests"""
        
        answers = {}
        text_lower = text.lower()
        
        patterns = {
            'cloud_provider': [
                r'\b(aws|amazon|azure|microsoft|gcp|google\s+cloud)\b'
            ],
            'resource_type': [
                r'\b(ec2|s3|lambda|rds|virtual\s+machine|vm|instance|server|storage)\b'
            ],
            'access_level': [
                r'\b(read|write|admin|full|console|ssh)\b'
            ],
            'business_justification': [
                r'for\s+([^.!?\n]*)',
                r'need\s+to\s+([^.!?\n]*)',
                r'purpose\s+([^.!?\n]*)'
            ]
        }
        
        for field in expected_fields:
            if field in patterns:
                for pattern in patterns[field]:
                    match = re.search(pattern, text_lower)
                    if match:
                        value = match.group(1).strip()
                        if value:
                            answers[field] = value
                            break
        
        return answers

    @staticmethod
    def _parse_network_request(text: str, expected_fields: List[str]) -> Dict[str, str]:
        """Enhanced network access request parsing with better accuracy"""
        
        answers = {}
        text_lower = text.lower()
        
        patterns = {
            'destination': [
                r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b',  # IP address (prioritize this)
                r'\b([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b',  # Domain name
                r'(?:to|from|destination|target|host|server)\s+([^\s]+(?:\.[^\s]+)*)'
            ],
            'port': [
                r'port\s+(\d+(?:[-,]\s*\d+)*)',  # Port with context word
                r'(?::\s*)(\d{1,5})(?:\s|$)',  # Port after colon
                r'(?:on|to|via)?\s+port\s+(\d{1,5})'  # More specific port patterns
            ],
            'protocol': [
                r'\b(ssh|tcp|udp|http|https|ftp|smtp|telnet|rdp|icmp|dns)\b'
            ],
            'business_justification': [
                r'(?:for|because|need|require|business|purpose|justification)[\s:]+([^.!?\n]*)',
                r'(?:to|in order to)\s+([^.!?\n]*)'
            ]
        }

        for field in expected_fields:
            if field in patterns:
                for pattern in patterns[field]:
                    match = re.search(pattern, text_lower, re.IGNORECASE)
                    if match:
                        value = match.group(1).strip()
                        if value:
                            answers[field] = value
                            logger.debug(f"Network pattern match: {field} = {value}")
                            break

        # Special handling for SSH protocol
        if 'protocol' in expected_fields and 'protocol' not in answers:
            if 'ssh' in text_lower:
                answers['protocol'] = 'TCP'
                logger.debug("SSH detected, defaulting protocol to TCP")

        # Try to extract destination from IP addresses
        if 'destination' in expected_fields and 'destination' not in answers:
            ip_match = re.search(r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b', text)
            if ip_match:
                answers['destination'] = ip_match.group(1)
                logger.debug(f"IP address extracted as destination: {ip_match.group(1)}")

        return answers

    @staticmethod
    def _parse_permission_request(text: str, expected_fields: List[str]) -> Dict[str, str]:
        """Enhanced permission escalation request parsing"""
        
        answers = {}
        text_lower = text.lower()
        
        patterns = {
            'target_system': [
                r'(?:to|on|for|system|server)\s+([A-Z][a-zA-Z0-9\s]*)',
                r'\b(AWS|GCP|Azure|Linux|Windows|Ubuntu|CentOS)\b'
            ],
            'permission_level': [
                r'\b(admin|administrator|root|sudo|read-only|readonly|write|full)\b'
            ],
            'duration': [
                r'\b(\d+\s*(?:hours?|days?|weeks?|months?))\b',
                r'\b(temporary|permanent|indefinite)\b'
            ],
            'manager_approval': [
                r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
            ]
        }

        for field in expected_fields:
            if field in patterns:
                for pattern in patterns[field]:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        value = match.group(1).strip()
                        if value:
                            if 'email' in field.lower() or 'approval' in field.lower():
                                value = TextProcessor.extract_email_from_slack_format(value)
                            answers[field] = value
                            logger.debug(f"Permission pattern match: {field} = {value}")
                            break

        return answers

    @staticmethod
    def _parse_data_request(text: str, expected_fields: List[str]) -> Dict[str, str]:
        """Enhanced data access request parsing"""
        
        answers = {}
        text_lower = text.lower()
        
        patterns = {
            'dataset_name': [
                r'(?:dataset|database|table|logs?|files?)\s+([a-zA-Z0-9_-]+)',
                r'\b([a-zA-Z][a-zA-Z0-9_-]*(?:logs?|db|database|data))\b'
            ],
            'access_level': [
                r'\b(read-only|readonly|read|write|admin|full)\b'
            ],
            'purpose': [
                r'(?:for|purpose|need|require)\s+([^.!?\n]*)'
            ],
            'data_classification': [
                r'\b(public|internal|confidential|restricted|secret)\b'
            ]
        }

        for field in expected_fields:
            if field in patterns:
                for pattern in patterns[field]:
                    match = re.search(pattern, text_lower, re.IGNORECASE)
                    if match:
                        value = match.group(1).strip()
                        if value:
                            answers[field] = value
                            logger.debug(f"Data pattern match: {field} = {value}")
                            break

        return answers

    @staticmethod
    def _smart_parse_single_line(text: str, expected_fields: List[str]) -> Dict[str, str]:
        """Enhanced smart parsing for single line responses"""

        answers = {}

        if not text or not expected_fields:
            return answers

        try:
            text_lower = text.lower()
            original_words = text.split()

            # Enhanced field patterns
            field_patterns = {
                # Access levels
                'access_level': r'\b(read-only|readonly|read|write|admin|administrator|full)\b',
                'permission_level': r'\b(read-only|readonly|read|write|admin|administrator|full)\b',
                'access_type': r'\b(read-only|readonly|read|write|admin|administrator|user|full)\b',

                # Data classification
                'data_classification': r'\b(public|internal|confidential|restricted|secret|low|medium|high)\b',

                # Duration patterns
                'duration': r'\b(\d+[-\s]*(hours?|days?|weeks?|months?)|temporary|permanent|indefinite)\b',
                'contract_duration': r'\b(\d+[-\s]*(hours?|days?|weeks?|months?)|temporary|permanent|indefinite)\b',

                # Protocol patterns
                'protocol': r'\b(tcp|udp|http|https|ssh|ftp|smtp)\b',

                # Port patterns
                'port': r'\b(\d{1,5})\b',

                # Destination patterns
                'destination': r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b',

                # Email patterns
                'manager_approval': r'(?:<mailto:)?([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})(?:\|[^>]+>)?',

                # Business purpose patterns
                'purpose': r'\b(debug|debugging|analysis|testing|monitoring|backup|maintenance|development|production|troubleshooting|investigation|research|audit)\b',
                'business_justification': r'\b(debug|debugging|analysis|testing|monitoring|backup|maintenance|development|production|business|urgent|emergency|troubleshooting)\b',

                # Dataset/system name patterns
                'dataset_name': r'\b([a-zA-Z][a-zA-Z0-9_-]*(?:logs?|db|database|data|system|server)[a-zA-Z0-9_-]*)\b',
                'system_name': r'\b([a-zA-Z][a-zA-Z0-9_-]*(?:server|system|host|service)[a-zA-Z0-9_-]*)\b',
                'target_system': r'\b([A-Z][a-zA-Z0-9\s]*(?:server|system|host|service|AWS|GCP|Azure)?)\b',
                'vendor_name': r'\b([A-Z][a-zA-Z0-9\s]*(?:Corp|Corporation|Inc|Ltd|LLC|Company)?)\b',
            }

            # First pass: Try to match patterns for each expected field
            for field in expected_fields:
                if field in field_patterns:
                    pattern = field_patterns[field]
                    match = re.search(pattern,
                                      text if 'email' in field.lower() or 'approval' in field.lower() else text_lower)
                    if match:
                        value = match.group(1)
                        # Special handling for email fields
                        if 'email' in field.lower() or 'approval' in field.lower():
                            value = TextProcessor.extract_email_from_slack_format(value)
                        answers[field] = value
                        logger.debug(f"Pattern match: {field} = {value}")

            # Second pass: For fields we didn't match, try word-by-word assignment
            unmatched_fields = [f for f in expected_fields if f not in answers]

            if unmatched_fields and original_words:
                # Remove words already used in pattern matching
                used_words = set()
                for value in answers.values():
                    if value:
                        used_words.update(value.lower().split())

                available_words = [w for w in original_words if w.lower() not in used_words]

                # Assign remaining words to remaining fields
                for i, field in enumerate(unmatched_fields):
                    if i < len(available_words):
                        word = available_words[i].strip()
                        if word:
                            # Special handling for email fields
                            if 'email' in field.lower() or 'approval' in field.lower():
                                word = TextProcessor.extract_email_from_slack_format(word)
                            answers[field] = word
                            logger.debug(f"Word assignment: {field} = {word}")

            # Third pass: Handle common dataset naming patterns
            if 'dataset_name' in expected_fields and 'dataset_name' not in answers:
                # Look for compound words like "dev_application_logs"
                for word in original_words:
                    if any(keyword in word.lower() for keyword in ['log', 'db', 'data', 'base']):
                        answers['dataset_name'] = word
                        logger.debug(f"Dataset pattern match: dataset_name = {word}")
                        break

            return answers

        except Exception as e:
            logger.error(f"Error in smart parsing: {e}")
            return {}