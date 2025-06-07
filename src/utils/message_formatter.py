# src/slack/message_formatter.py - ENHANCED WITH PERFORMANCE IMPROVEMENTS

from datetime import datetime
from typing import List
from src.models.enums import RequestType, Outcome
from src.models.data_models import SecurityRequest


class MessageFormatter:
    """Enhanced Message Formatter with integrated performance improvements"""
    
    @staticmethod
    def format_follow_up_questions(missing_fields: List[str], request_type: RequestType) -> str:
        """ENHANCED: Format follow-up questions with better context"""

        # Enhanced questions with more specific guidance
        questions = {
            # DevTool Install (was missing from original)
            "tool_name": "What development tool do you need to install? (e.g., Docker, Visual Studio Code, Java JDK)",
            "version": "Which version do you need? (e.g., 'latest', '3.8', 'v2.1')",
            "purpose": "What will you use this tool for? (e.g., web development, data science, Java programming)",
            
            # Network Access (enhanced clarity)
            "destination": "What is the destination IP address or hostname you need access to?",
            "port": "Which port(s) do you need opened? (e.g., 22 for SSH, 443 for HTTPS)",
            "protocol": "What protocol will be used? (TCP/UDP/HTTPS/SSH)",
            "business_justification": "Please provide a business justification for this network access request.",

            # Vendor Approval (enhanced)
            "vendor_name": "What is the exact name of the vendor/company?",
            "service_description": "Please describe the specific service this vendor will provide.",
            "data_types": "What types of data will the vendor have access to? (e.g., customer data, financial records)",
            "contract_duration": "What is the duration of the contract/engagement? (e.g., '6 months', '1 year')",

            # Permission Change (enhanced clarity)
            "target_system": "Which specific system do you need permission changes for? (e.g., production database, AWS console)",
            "permission_level": "What level of permissions do you need? (admin/read-only/write/full access)",
            "duration": "How long do you need these permissions? (e.g., '24 hours', '1 week', 'permanent')",
            "manager_approval": "Please provide your manager's email for approval.",

            # Cloud Resource Access (distinguish from network)
            "cloud_provider": "Which cloud provider? (AWS/Azure/GCP/Google Cloud)",
            "resource_type": "What type of resource? (EC2 instance, S3 bucket, virtual machine, etc.)",
            "access_level": "What level of access do you need? (console/SSH/read/write/admin)",
            "business_justification": "Please provide a business justification for this cloud access request.",

            # Data Export (enhanced)
            "dataset_name": "What is the name/description of the dataset or database?",
            "access_level": "What level of access do you need? (read-only/write/admin)",
            "purpose": "What is the purpose of accessing this data? (analysis/reporting/migration/etc.)",
            "data_classification": "What is the data classification level? (public/internal/confidential/restricted)",
            "export_format": "What format do you need? (CSV/JSON/Excel/database export)",
            "destination": "Where will the exported data be stored or sent?",

            # Firewall Change (enhanced)
            "source_ip": "What is the source IP address or network range?",
            "destination_ip": "What is the destination IP address or network?",
            "port": "Which port(s) need to be opened? (e.g., 80, 443, 22)",
            "protocol": "What protocol? (TCP/UDP/ICMP)",
            "business_justification": "Please provide a business justification for this firewall change.",

            # Common fields with enhanced descriptions
            "detailed_description": "Please provide a detailed description of your request, including the business need and any relevant context."
        }

        question_lines = []
        for field in missing_fields:
            question = questions.get(field, f"Please provide {field.replace('_', ' ')}")
            question_lines.append(f"• {question}")

        type_name = request_type.value.replace('_', ' ').title()

        return (
                f"I need some additional information to process your **{type_name}** request:\n\n"
                + "\n".join(question_lines) +
                "\n\nPlease answer each question on a separate line, or provide the information in a clear format."
        )

    @staticmethod
    def format_final_response(request: SecurityRequest) -> str:
        """ENHANCED: Format the final decision response with better indicators"""

        # Enhanced emoji mapping for better visual clarity
        emoji_map = {
            Outcome.APPROVED: "✅",
            Outcome.REJECTED: "❌", 
            Outcome.NEEDS_MORE_INFO: "❓"
        }

        emoji = emoji_map.get(request.outcome, "🔒")
        type_name = request.request_type.value.replace('_', ' ').title() if request.request_type else "Unknown"

        # Generate request ID with enhanced format
        request_id = f"SEC-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        # Enhanced response format
        response = f"""{emoji} **Security Request Analysis**

**Request Type:** {type_name}
**Risk Score:** {request.risk_score:.1f}/10
**Decision:** {request.outcome.value}

**Analysis:** {request.rationale}

---
*Request ID: {request_id}*"""

        return response

    @staticmethod
    def format_adaptive_response(request: SecurityRequest, processing_info: dict) -> str:
        """ENHANCED: Format response with comprehensive adaptive ML insights"""

        # Enhanced outcome indicators
        outcome_indicators = {
            Outcome.APPROVED: "✅ [APPROVED]",
            Outcome.REJECTED: "❌ [REJECTED]", 
            Outcome.NEEDS_MORE_INFO: "❓ [REVIEW NEEDED]"
        }

        indicator = outcome_indicators.get(request.outcome, "[INFO]")
        type_name = request.request_type.value.replace('_', ' ').title() if request.request_type else "Unknown"

        # Build comprehensive response with performance insights
        response = f"{indicator} **Enhanced Security Analysis**\n\n"
        response += f"**Request Type:** {type_name}\n"
        response += f"**Risk Score:** {request.risk_score:.1f}/10\n"
        response += f"**Decision:** {request.outcome.value}\n\n"
        response += f"**AI Analysis:** {request.rationale}\n"

        # Add enhanced ML insights
        risk_info = processing_info.get('risk_assessment', {})

        # Show confidence and method used
        if risk_info.get('method'):
            method = risk_info['method']
            confidence = risk_info.get('confidence', 0.8)
            response += f"**Analysis Method:** {method.replace('_', ' ').title()}\n"
            response += f"**Confidence:** {confidence:.1%}\n"

        # Show key risk factors
        if risk_info.get('risk_factors'):
            factors = risk_info['risk_factors'][:3]  # Top 3
            response += f"**Key Risk Factors:** {', '.join(factors)}\n"

        # Show learned thresholds if available
        thresholds = processing_info.get('learned_thresholds', {})
        if thresholds and 'approval_threshold' in thresholds:
            approval_thresh = thresholds.get('approval_threshold', 'N/A')
            rejection_thresh = thresholds.get('rejection_threshold', 'N/A')
            response += f"**Learned Thresholds:** Approve ≤{approval_thresh:.1f}, Reject ≥{rejection_thresh:.1f}\n"

        # Show training info
        training_samples = processing_info.get('training_samples', 0)
        if training_samples > 0:
            response += f"**Training Data:** Learned from {training_samples} historical decisions\n"

        # Enhanced algorithm information
        enhanced_features = processing_info.get('enhanced_features', {})
        if enhanced_features:
            algo_score = enhanced_features.get('algorithmic_sophistication', 0)
            if algo_score > 0:
                response += f"**Algorithm Sophistication:** {algo_score:.1f}/10\n"
            
            ensemble_count = enhanced_features.get('ensemble_models', 0)
            if ensemble_count > 0:
                response += f"**ML Models Used:** {ensemble_count + 1} model ensemble\n"

        # Add request ID and attribution
        request_id = f"SEC-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        response += f"\n---\n*Request ID: {request_id} | Enhanced AI Security Analysis*"

        return response

    @staticmethod
    def format_error_response(error_message: str = None) -> str:
        """ENHANCED: Format error response with better guidance"""
        default_message = "I encountered an error processing your request. Please try rephrasing your request or contact the security team for assistance."
        
        enhanced_message = f"⚠️ **Processing Error**\n\n{error_message or default_message}\n\n"
        enhanced_message += "**Tips for better results:**\n"
        enhanced_message += "• Be specific about what you need access to\n"
        enhanced_message += "• Include the business justification\n"
        enhanced_message += "• Mention if this is urgent or time-sensitive\n"
        enhanced_message += "• Use `/security-help` for examples and guidance"
        
        return enhanced_message

    @staticmethod
    def format_help_message() -> str:
        """ENHANCED: Format help message with comprehensive guidance"""
        return """🔒 **Enhanced ACME Security Bot Help**

I can help you with these types of security requests:

**📦 Development Tools**
• Install development software (Docker, IDEs, etc.)
• Setup programming environments
• Download development packages

**🌐 Network Access**  
• VPN access for remote work
• Firewall rules and port access
• Network connectivity requests

**☁️ Cloud Resources**
• AWS/Azure/GCP console access
• Virtual machine and server access
• Cloud storage and services

**🏢 Vendor Management**
• Onboard new vendors and contractors
• Third-party service approvals
• Supplier security assessments

**🔐 Permission Changes**
• Admin access requests
• Privilege escalation (temporary/permanent)
• System administrator rights

**📊 Data Export**
• Database access for analysis
• Data exports and extractions
• Customer data access requests

**🛡️ Firewall Changes**
• Security group modifications
• Network ACL updates
• Perimeter firewall changes

**How to use:**
• Use `/security-request` followed by your request
• Be specific about what you need and why
• I'll ask follow-up questions if needed

**Examples:**
• `/security-request Install Docker Desktop for containerized development`
• `/security-request Need VPN access for remote work from home`
• `/security-request SSH access to AWS EC2 production instance`
• `/security-request Onboard vendor Salesforce for our CRM system`

**Pro Tips:**
• Include business justification in your initial request
• Mention if the request is urgent or time-sensitive
• Specify exact systems, tools, or resources you need

For urgent security matters, contact the security team directly at security@acme.com"""

    @staticmethod
    def format_classification_confidence_message(request_type: RequestType, confidence: float, explanation: str = None) -> str:
        """ENHANCED: Show classification confidence for transparency"""
        
        type_name = request_type.value.replace('_', ' ').title()
        confidence_level = "High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low"
        
        message = f"🎯 **Classification Result**\n"
        message += f"**Detected Type:** {type_name}\n"
        message += f"**Confidence:** {confidence:.1%} ({confidence_level})\n"
        
        if explanation:
            message += f"**Reasoning:** {explanation}\n"
        
        if confidence < 0.7:
            message += f"\n⚠️ **Note:** Low confidence classification. Please verify the request type is correct.\n"
        
        return message

    @staticmethod
    def format_performance_debug_message(processing_info: dict) -> str:
        """ENHANCED: Debug message showing processing performance (for development)"""
        
        risk_info = processing_info.get('risk_assessment', {})
        method = risk_info.get('method', 'unknown')
        confidence = risk_info.get('confidence', 0.0)
        
        message = f"🔍 **Debug Info** (Development Mode)\n"
        message += f"**Processing Method:** {method}\n"
        message += f"**Confidence:** {confidence:.2%}\n"
        
        if 'enhanced_features' in processing_info:
            features = processing_info['enhanced_features']
            message += f"**Ensemble Models:** {features.get('ensemble_models', 0)}\n"
            message += f"**Algorithm Score:** {features.get('algorithmic_sophistication', 0):.1f}/10\n"
        
        training_samples = processing_info.get('training_samples', 0)
        if training_samples > 0:
            message += f"**Training Samples:** {training_samples}\n"
        
        return message