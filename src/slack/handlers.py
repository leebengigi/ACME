import sys
from pathlib import Path

# Add project root to Python path for proper module imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import logging
import json
import re
from typing import Dict, List

from src.models.data_models import SecurityRequest
from src.models.enums import RequestType, Outcome
from src.services.classification_service import ClassificationService
from src.services.adaptive_bot_system import AdaptiveBotSystem
from src.database.repository import SecurityRequestRepository
from src.slack.message_formatter import MessageFormatter
from src.utils.text_processing import TextProcessor
from src.utils.validators import RequestValidator
from config.settings import Settings

logger = logging.getLogger(__name__)


class SlackHandlers:
    """
    Handles all Slack interactions and message processing for the security bot.
    
    This class manages the bot's interaction with Slack, including:
    - Processing security requests via commands and mentions
    - Handling follow-up questions and responses
    - Managing conversation flow and state
    - Integrating with ML systems for request processing
    """

    def __init__(
            self,
            slack_app,
            classification_service: ClassificationService,
            adaptive_system: AdaptiveBotSystem,
            repository: SecurityRequestRepository,
            settings: Settings
    ):
        """
        Initialize the Slack handlers with required services.
        
        Args:
            slack_app: The Slack Bolt app instance
            classification_service: Service for request classification
            adaptive_system: ML system for risk assessment and decisions
            repository: Database repository for request persistence
            settings: Application configuration settings
        """
        self.app = slack_app
        self.classification_service = classification_service
        self.adaptive_system = adaptive_system
        self.repository = repository
        self.settings = settings

        self._setup_handlers()

    def _setup_handlers(self):
        """Setup all Slack event handlers for commands and messages"""

        @self.app.command("/security-request")
        def handle_security_command(ack, respond, command):
            """Handle /security-request slash command"""
            ack()
            try:
                self._process_security_request(
                    user_id=command['user_id'],
                    channel_id=command['channel_id'],
                    text=command['text'],
                    respond=respond
                )
            except Exception as e:
                logger.error(f"Error in command handler: {e}")
                try:
                    respond(MessageFormatter.format_error_response())
                except Exception as respond_error:
                    logger.error(f"Failed to send error response: {respond_error}")

        @self.app.command("/security-help")
        def handle_help_command(ack, respond, command):
            """Handle /security-help slash command"""
            ack()
            respond(MessageFormatter.format_help_message())

        @self.app.event("app_mention")
        def handle_mention(event, say):
            """Handle direct mentions of the bot"""
            try:
                # Clean the mention from the text
                text = event['text']
                # Remove the bot mention
                text = ' '.join([word for word in text.split() if not word.startswith('<@')])

                self._process_security_request(
                    user_id=event['user'],
                    channel_id=event['channel'],
                    text=text,
                    respond=say,
                    thread_ts=event['ts']
                )
            except Exception as e:
                logger.error(f"Error in mention handler: {e}")
                try:
                    say(MessageFormatter.format_error_response())
                except Exception as say_error:
                    logger.error(f"Failed to send error response: {say_error}")

        @self.app.event("message")
        def handle_follow_up_message(event, say):
            """Handle follow-up messages in threads"""
            # Ignore bot messages
            if event.get("bot_id") or event.get("subtype"):
                return

            try:
                self._handle_follow_up_response(
                    user_id=event['user'],
                    channel_id=event['channel'],
                    text=event['text'],
                    respond=say,
                    thread_ts=event.get('thread_ts')
                )
            except Exception as e:
                logger.error(f"Error in follow-up handler: {e}")
                # Print more detailed error info for debugging
                import traceback
                logger.error(f"Detailed error: {traceback.format_exc()}")

    def _safe_respond(self, respond_func, message):
        """
        Safely send response with error handling and logging.
        
        Args:
            respond_func: The Slack response function to use
            message: The message to send
        """
        try:
            respond_func(message)
        except Exception as e:
            logger.error(f"Failed to send Slack message: {e}")
            if "not_allowed_token_type" in str(e):
                logger.error("Token issue: Make sure you're using a Bot User OAuth Token (xoxb-)")
            elif "missing_scope" in str(e):
                logger.error("Permission issue: Bot needs chat:write scope")

    def _process_security_request(
            self,
            user_id: str,
            channel_id: str,
            text: str,
            respond,
            thread_ts: str = None
    ):
        """
        Process a new security request using the ML system.
        
        This method:
        1. Cleans and validates the input text
        2. Creates a security request object
        3. Classifies the request type
        4. Extracts required fields
        5. Processes the request through the ML system
        6. Handles follow-up questions if needed
        
        Args:
            user_id: Slack user ID of the requester
            channel_id: Slack channel ID where request was made
            text: The request text
            respond: Function to send response
            thread_ts: Thread timestamp for threaded responses
        """
        # Clean the input text
        cleaned_text = TextProcessor.clean_slack_text(text)

        if not cleaned_text or len(cleaned_text.strip()) < 10:
            self._safe_respond(respond, "Please provide a more detailed description of your security request.")
            return

        # Create security request object
        request = SecurityRequest(
            user_id=user_id,
            channel_id=channel_id,
            thread_ts=thread_ts or "",
            request_text=cleaned_text
        )

        # Classify the request using ML
        request.request_type, classification_confidence = self.classification_service.classify_request(cleaned_text)

        logger.info(
            f"Processing request from {user_id}: {request.request_type.value} (classification confidence: {classification_confidence:.3f})")

        # Extract and pre-fill fields from the original request
        request.required_fields = self._extract_fields_from_request(cleaned_text, request.request_type)
        logger.info(f"Pre-filled fields from original request: {request.required_fields}")

        # Check for missing required fields
        missing_fields = self._get_missing_fields(request)

        if missing_fields:
            # Ask follow-up questions for missing fields
            follow_up_message = MessageFormatter.format_follow_up_questions(missing_fields, request.request_type)
            self._safe_respond(respond, follow_up_message)

            # Save partial request to database
            request_id = self.repository.save_request(request)
            self.repository.log_interaction(request_id, "follow_up_questions", follow_up_message)

        else:
            # Process complete request through ML system
            processed_request, processing_info = self.adaptive_system.process_request(request)

            # Format and send response with ML insights
            final_message = MessageFormatter.format_adaptive_response(processed_request, processing_info)
            self._safe_respond(respond, final_message)

            # Save complete request to database
            request_id = self.repository.save_request(processed_request)
            self.repository.log_interaction(request_id, "adaptive_decision", final_message)

    def _extract_fields_from_request(self, request_text: str, request_type: RequestType) -> Dict[str, str]:
        """
        Extract and pre-fill fields from the original request text.
        
        Uses specialized parsing for different request types to extract
        relevant information from the initial request.
        
        Args:
            request_text: The cleaned request text
            request_type: The type of security request
            
        Returns:
            Dictionary of extracted fields and their values
        """
        fields = {}

        if request_type == RequestType.NETWORK_ACCESS:
            # Parse network-specific fields
            network_fields = ['destination', 'port', 'protocol', 'business_justification']
            extracted = TextProcessor._parse_network_request(request_text, network_fields)
            fields.update(extracted)

        elif request_type == RequestType.PERMISSION_CHANGE:
            # Parse permission-specific fields
            permission_fields = ['target_system', 'permission_level', 'duration', 'manager_approval']
            extracted = TextProcessor._parse_permission_request(request_text, permission_fields)
            fields.update(extracted)

        elif request_type == RequestType.DATA_EXPORT:
            # Parse data export-specific fields
            data_fields = ['dataset_name', 'access_level', 'purpose', 'data_classification']
            extracted = TextProcessor._parse_data_request(request_text, data_fields)
            fields.update(extracted)

        return fields

    def _handle_follow_up_response(
            self,
            user_id: str,
            channel_id: str,
            text: str,
            respond,
            thread_ts: str = None
    ):
        """
        Handle follow-up responses to security requests.
        
        This method:
        1. Retrieves the pending request
        2. Parses the follow-up response
        3. Updates required fields
        4. Validates the updated information
        5. Either asks for more information or processes the complete request
        
        Args:
            user_id: Slack user ID
            channel_id: Slack channel ID
            text: The follow-up response text
            respond: Function to send response
            thread_ts: Thread timestamp
        """
        
        # Get pending request from database
        pending = self.repository.get_pending_request(user_id, channel_id)
        
        if not pending:
            logger.warning(f"❌ No pending request found for user {user_id} in channel {channel_id}")
            logger.info(f"🔍 Checking database for any requests from this user...")
            
            # Debug: Check recent requests
            try:
                with self.repository.get_connection() as conn:
                    cur = conn.cursor()
                    cur.execute("SELECT id, user_id, channel_id, outcome, timestamp FROM requests WHERE user_id = ? ORDER BY id DESC LIMIT 5", (user_id,))
                    recent_requests = cur.fetchall()
                    logger.info(f"📋 Recent requests for user {user_id}: {recent_requests}")
            except Exception as e:
                logger.error(f"Database debug query failed: {e}")
            
            return  # Exit if no pending request
        
        logger.info(f"✅ Found pending request: ID {pending[0]}")
        
        # Extract request details
        request_id, required_fields, request_type_str, risk_score = pending
        
        # Validate and parse required fields
        logger.info(f"📝 Required fields type: {type(required_fields)}")
        logger.info(f"📝 Required fields content: {required_fields}")
        if required_fields is None:
            logger.warning(f"required_fields was None for request {request_id}, initializing as empty dict")
            required_fields = {}
        elif isinstance(required_fields, str):
            try:
                required_fields = json.loads(required_fields)
            except json.JSONDecodeError:
                logger.warning(f"Could not parse required_fields JSON for request {request_id}, initializing as empty dict")
                required_fields = {}
        elif not isinstance(required_fields, dict):
            logger.warning(f"required_fields was not a dict for request {request_id} (type: {type(required_fields)}), initializing as empty dict")
            required_fields = {}

        logger.debug(f"Required fields after validation: {required_fields}")

        # Parse request type
        try:
            request_type = RequestType(request_type_str)
        except ValueError:
            logger.warning(f"Unknown request type: {request_type_str}, defaulting to OTHER")
            request_type = RequestType.OTHER

        # Get missing fields
        missing_fields = [
            field for field in self.settings.REQUIRED_FIELDS[request_type]
            if not required_fields.get(field)
        ]

        if not missing_fields:
            logger.debug("No missing fields found")
            return  # No missing fields

        logger.info(f"Processing follow-up for {len(missing_fields)} missing fields: {missing_fields}")

        # Parse follow-up response
        cleaned_text = TextProcessor.clean_slack_text(text)
        answers = self._parse_follow_up_enhanced(cleaned_text, missing_fields)

        logger.info(f"Enhanced parsed answers: {answers}")

        # Update required fields with parsed answers
        updates_made = 0
        for field, answer in answers.items():
            if answer and field in missing_fields:
                # Clean email fields
                if 'email' in field.lower() or 'approval' in field.lower():
                    answer = TextProcessor.extract_email_from_slack_format(answer)

                required_fields[field] = answer
                updates_made += 1
                logger.debug(f"Updated {field} = {answer}")

        # Validate updated fields
        try:
            validation_errors = RequestValidator.validate_required_fields(request_type, required_fields)
        except Exception as e:
            logger.error(f"Validation error: {e}")
            validation_errors = {}

        if validation_errors:
            error_message = "Please correct the following:\n" + "\n".join(
                f"• {field}: {error}" for field, error in validation_errors.items()
            )
            self._safe_respond(respond, error_message)
            return

        # Update database with new information
        try:
            self.repository.update_request_fields(request_id, required_fields)
            self.repository.log_interaction(request_id, "follow_up_response", cleaned_text)
        except Exception as e:
            logger.error(f"Database update error: {e}")
            self._safe_respond(respond, "Sorry, I encountered a database error. Please try again.")
            return

        # Check remaining missing fields
        still_missing = [
            field for field in self.settings.REQUIRED_FIELDS[request_type]
            if not required_fields.get(field)
        ]

        logger.info(f"After updates: {updates_made} fields updated, {len(still_missing)} still missing")

        if still_missing:
            # Ask for remaining fields
            follow_up_message = MessageFormatter.format_follow_up_questions(still_missing, request_type)
            self._safe_respond(respond, follow_up_message)
            try:
                self.repository.log_interaction(request_id, "additional_questions", follow_up_message)
            except Exception as e:
                logger.error(f"Failed to log additional questions: {e}")
        else:
            # Process complete request
            request = SecurityRequest(
                user_id=user_id,
                channel_id=channel_id,
                thread_ts=thread_ts or "",
                request_text="Follow-up completion",
                request_type=request_type,
                risk_score=risk_score,
                required_fields=required_fields
            )

            try:
                # Process through ML system
                processed_request, processing_info = self.adaptive_system.process_request(request)

                # Send final response
                final_message = MessageFormatter.format_adaptive_response(processed_request, processing_info)
                self._safe_respond(respond, final_message)

                # Update database with final decision
                self.repository.finalize_request(request_id, processed_request.outcome.value,
                                                 processed_request.rationale)
                self.repository.log_interaction(request_id, "adaptive_final_decision", final_message)

            except Exception as e:
                logger.error(f"Error in ML system processing: {e}")
                # Fallback response
                fallback_message = f"[INFO NEEDED] I processed your responses but encountered an issue with the ML system. Your request has been logged for manual review.\n\nRequest ID: {request_id}"
                self._safe_respond(respond, fallback_message)

    def _parse_follow_up_enhanced(self, text: str, expected_fields: List[str]) -> Dict[str, str]:
        """
        Enhanced parsing of follow-up responses using multiple methods.
        
        This method tries several parsing strategies in order:
        1. Pattern-based extraction using regex
        2. Line-by-line parsing
        3. Smart word-by-word assignment
        
        Args:
            text: The follow-up response text
            expected_fields: List of fields to extract
            
        Returns:
            Dictionary of extracted fields and their values
        """
        answers = {}

        if not text or not expected_fields:
            return answers

        logger.debug(f"Enhanced parsing - Text: '{text}', Fields: {expected_fields}")

        # Remove trailing content after semicolon
        cleaned_text = text.split(';')[0].strip()
        logger.debug(f"Cleaned text: '{cleaned_text}'")

        # Try pattern-based extraction first
        pattern_answers = self._extract_by_patterns(cleaned_text, expected_fields)
        if len(pattern_answers) >= len(expected_fields) * 0.8:  # Got most fields
            logger.debug(f"Pattern-based extraction successful: {pattern_answers}")
            return pattern_answers

        # Try line-by-line parsing
        lines = [line.strip() for line in cleaned_text.split('\n') if line.strip()]
        if len(lines) >= len(expected_fields):
            for i, field in enumerate(expected_fields):
                if i < len(lines):
                    answers[field] = lines[i]
            logger.debug(f"Line-by-line parsing: {answers}")
            return answers

        # Try smart word-by-word assignment
        words = cleaned_text.split()
        if len(words) >= len(expected_fields):
            # Special handling for network requests
            if 'destination' in expected_fields:
                return self._parse_network_follow_up(words, expected_fields)
            # Direct assignment for other types
            else:
                for i, field in enumerate(expected_fields):
                    if i < len(words):
                        answers[field] = words[i]

        logger.debug(f"Final enhanced parsing result: {answers}")
        return answers

    def _extract_by_patterns(self, text: str, fields: List[str]) -> Dict[str, str]:
        """
        Extract fields using regex patterns.
        
        Uses predefined patterns to extract specific field types like:
        - IP addresses
        - Port numbers
        - Protocols
        - Email addresses
        - Duration values
        - System names
        - Permission levels
        
        Args:
            text: The text to parse
            fields: List of fields to extract
            
        Returns:
            Dictionary of extracted fields and their values
        """
        answers = {}

        patterns = {
            'destination': r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b',
            'port': r'\b(\d{1,5})(?!\.\d)\b',  # Number not part of IP
            'protocol': r'\b(ssh|tcp|udp|http|https|ftp|smtp)\b',
            'business_justification': None,  # Handle separately
            'manager_approval': r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
            'duration': r'\b(\d+\s*(?:hours?|days?|weeks?|months?))\b',
            'target_system': r'\b([A-Za-z][A-Za-z0-9]*)\b',
            'permission_level': r'\b(admin|administrator|read|write|full|readonly)\b'
        }

        for field in fields:
            if field in patterns and patterns[field]:
                match = re.search(patterns[field], text, re.IGNORECASE)
                if match:
                    answers[field] = match.group(1)

        # Handle business justification - everything after technical fields
        if 'business_justification' in fields:
            remaining = text
            for value in answers.values():
                remaining = remaining.replace(str(value), '', 1).strip()
            if remaining:
                answers['business_justification'] = remaining

        return answers

    def _parse_network_follow_up(self, words: List[str], expected_fields: List[str]) -> Dict[str, str]:
        """
        Parse network-specific follow-up responses.
        
        Specialized parsing for network access requests that handles:
        - IP addresses
        - Port numbers
        - Protocols
        - Business justification
        
        Args:
            words: List of words from the response
            expected_fields: List of fields to extract
            
        Returns:
            Dictionary of extracted network-related fields
        """
        answers = {}
        used_indices = set()

        # Look for IP address
        if 'destination' in expected_fields:
            for i, word in enumerate(words):
                if re.match(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', word):
                    answers['destination'] = word
                    used_indices.add(i)
                    break
        # Look for source CIDR
        if 'source' in expected_fields:
            for i, word in enumerate(words):
                if re.match(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?:/\d{1,2})?', word):
                    answers['source'] = word
                    used_indices.add(i)
                    break
        # Business justification - remaining words
        if 'business_justification' in expected_fields:
            remaining_words = [words[i] for i in range(len(words)) if i not in used_indices]
            if remaining_words:
                answers['business_justification'] = ' '.join(remaining_words)

        return answers

    def _get_missing_fields(self, request: SecurityRequest) -> list:
        """
        Get list of missing required fields for a request.
        
        Args:
            request: The security request to check
            
        Returns:
            List of field names that are missing
        """
        if not request.request_type:
            return []

        required = self.settings.REQUIRED_FIELDS.get(request.request_type, [])
        return [field for field in required if not request.required_fields.get(field)]