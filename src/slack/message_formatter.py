from datetime import datetime
from typing import List
from src.models.enums import RequestType, Outcome
from src.models.data_models import SecurityRequest


class MessageFormatter:
    """
    Handles formatting of all bot messages and responses.
    
    This class provides static methods for formatting different types of messages:
    - Follow-up questions for missing information
    - Final decision responses
    - Error messages
    - Help messages
    - Adaptive ML-based responses
    """

    @staticmethod
    def format_follow_up_questions(missing_fields: List[str], request_type: RequestType) -> str:
        """
        Format follow-up questions for missing required fields.
        
        Creates a structured message asking for additional information based on
        the request type and missing fields. Questions are tailored to specific
        field types and request categories.
        
        Args:
            missing_fields: List of field names that need additional information
            request_type: Type of security request being processed
            
        Returns:
            Formatted message string with questions
        """
        # Question templates for different field types
        questions = {
             # VENDOR APPROVAL fields
            "vendor_security_questionnaire": "Has the vendor completed our security questionnaire? Please provide the questionnaire ID or confirmation.",
            "data_classification": "What is the data classification level? (Public/Internal/Confidential/Restricted)",
            "legal_review": "Has this vendor agreement been reviewed by legal? Please provide legal review reference or approval.",

            # PERMISSION CHANGE fields  
            "business_justification": "Please provide a detailed business justification for this request.",
            "duration": "How long do you need these permissions? (e.g., '30 days', 'permanent', '1 week')",
            "manager_approval": "Please provide your manager's email for approval.",
            "target_system": "Which specific system do you need elevated permissions for?",
            "permission_level": "What level of permissions do you need? (read-only/write/admin/full access)",

            # NETWORK ACCESS fields
            "source_CIDR": "What is the source CIDR/IP range that needs access? (e.g., '10.0.1.0/24')",
            "engineering_lead_approval": "Please provide your engineering lead's email for approval.",
            # business_justification already defined above

            # FIREWALL CHANGE fields
            "destination_ip": "What is the destination IP address or range? (e.g., '192.168.1.100' or '10.0.0.0/16')",
            "source_system": "What is the source system name or IP that needs access?",
            # business_justification already defined above

            # DEVTOOL INSTALL fields
            # business_justification already defined above
            # manager_approval already defined above

            # DATA EXPORT fields
            "data_destination": "Where will the exported data be stored or sent? (system name, email, cloud storage, etc.)",
            "PII_involved": "Does this data export involve PII (Personally Identifiable Information)? (Yes/No)",
            # business_justification already defined above

            # CLOUD RESOURCE ACCESS fields
            "data_sensitivty_level": "What is the data sensitivity level for this cloud resource? (Low/Medium/High/Critical)",
            # business_justification already defined above

            # OTHER/Generic field
            "detailed_description": "Please provide a detailed description of your request, including what you need and why."
        }

        # Build list of questions for missing fields
        question_lines = []
        for field in missing_fields:
            question = questions.get(field, f"Please provide {field.replace('_', ' ')}")
            question_lines.append(f"• {question}")

        # Format request type for display
        type_name = request_type.value.replace('_', ' ').title()

        # Construct final message
        return (
                f"I need some additional information to process your **{type_name}** request:\n\n"
                + "\n".join(question_lines) +
                "\n\nPlease answer each question on a separate line."
        )

    @staticmethod
    def format_final_response(request: SecurityRequest) -> str:
        """
        Format the final decision response for a security request.
        
        Creates a structured message containing the request analysis, decision,
        and rationale. Includes risk score and request identification.
        
        Args:
            request: The processed security request object
            
        Returns:
            Formatted decision message
        """
        # Map outcomes to display indicators
        emoji_map = {
            Outcome.APPROVED: "[APPROVED]",
            Outcome.REJECTED: "[REJECTED]",
            Outcome.NEEDS_MORE_INFO: "[INFO NEEDED]"
        }

        emoji = emoji_map.get(request.outcome, "[SECURITY]")
        type_name = request.request_type.value.replace('_', ' ').title() if request.request_type else "Unknown"

        # Generate unique request ID
        request_id = datetime.now().strftime('%Y%m%d-%H%M%S')

        # Construct response message
        return f"""{emoji} **Security Request Analysis**
        Request Type: {type_name}
        Risk Score: {request.risk_score:.1f}/10
        Decision: {request.outcome.value}
        
        Rationale: {request.rationale}
        ---
        *Request ID: {request_id}*"""

    @staticmethod
    def format_error_response(error_message: str = None) -> str:
        """
        Format error response message.
        
        Creates a user-friendly error message with optional custom error details.
        
        Args:
            error_message: Optional custom error message to include
            
        Returns:
            Formatted error message
        """
        default_message = "Sorry, I encountered an error processing your request. Please try again or contact the security team."
        return f"[WARNING] {error_message or default_message}"

    @staticmethod
    def format_help_message() -> str:
        """
        Format the help message with bot usage instructions.
        
        Creates a comprehensive help message explaining:
        - Available request types
        - How to use the bot
        - Example usage
        - Emergency contact information
        
        Returns:
            Formatted help message
        """
        return """[SECURITY] **ACME Security Bot Help**
        I can help you with security requests such as:
        • Network access requests
        • Vendor approvals  
        • Permission escalations
        • Data access requests
        • System access requests

        **How to use:**
        • Use `/security-request` followed by your request
        • Or mention me (@security-bot) with your request
        • I'll ask follow-up questions if needed

        **Example:**
        `/security-request Need access to production database for debugging customer issue`

        For urgent security matters, please contact the security team directly."""

    @staticmethod
    def format_adaptive_response(request: SecurityRequest, processing_info: dict) -> str:
        """
        Format response with adaptive ML insights and analysis.
        
        Creates a detailed response that includes:
        - Request analysis
        - Risk assessment
        - ML confidence levels
        - Key risk factors
        - Training data context
        
        Args:
            request: The processed security request
            processing_info: Additional ML processing information
            
        Returns:
            Formatted response with ML insights
        """
        # Map outcomes to display indicators
        outcome_indicators = {
            Outcome.APPROVED: "[APPROVED]",
            Outcome.REJECTED: "[REJECTED]",
            Outcome.NEEDS_MORE_INFO: "[REVIEW NEEDED]"
        }

        indicator = outcome_indicators.get(request.outcome, "[INFO]")
        type_name = request.request_type.value.replace('_', ' ').title() if request.request_type else "Unknown"

        # Build base response with request details
        response = f"{indicator} *Security Request Analysis (Adaptive AI)* \n\n"
        response += f"Request Type: {type_name}\n"
        response += f"Risk Score: {request.risk_score:.1f}/10\n"
        response += f"Decision: {request.outcome.value}\n\n"
        response += f"AI Analysis: {request.rationale}\n"

        # Add ML-specific insights
        risk_info = processing_info.get('risk_assessment', {})

        # Include confidence level if available
        if risk_info.get('confidence'):
            confidence = risk_info['confidence']
            response += f"Confidence: {confidence:.1%}\n"

        # Add top risk factors
        if risk_info.get('risk_factors'):
            factors = risk_info['risk_factors'][:3]  # Show top 3 factors
            response += f"Key Risk Factors: {', '.join(factors)}\n"

        # Include training data context
        training_samples = processing_info.get('training_samples', 0)
        if training_samples > 0:
            response += f"Training Data: Learned from {training_samples} historical decisions\n"

        # Add request tracking information
        request_id = datetime.now().strftime('%Y%m%d-%H%M%S')
        response += f"\n---\n*Request ID: {request_id} | Powered by Adaptive AI*"

        return response