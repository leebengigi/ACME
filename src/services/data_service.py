"""
Data Service Module for Security Request Management.

This module provides comprehensive data management services for security requests,
including data loading, normalization, cleaning, and augmentation. It handles
historical security ticket data and provides synthetic data generation for
training purposes.

Key Features:
- Historical data loading and normalization
- Request type and outcome standardization
- Data cleaning and validation
- Synthetic data generation for training
- Comprehensive data augmentation
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import logging
from typing import Optional
from src.models.enums import RequestType, Outcome

logger = logging.getLogger(__name__)


class DataService:
    """
    Enhanced Data Service with integrated training data augmentation.
    
    This class manages security request data, providing functionality for:
    - Loading and normalizing historical data
    - Standardizing request types and outcomes
    - Data cleaning and validation
    - Synthetic data generation for training
    - Data augmentation for improved ML performance
    
    Attributes:
        data_path (str): Path to the historical data file
        historical_data (Optional[pd.DataFrame]): Loaded and processed historical data
    """
    
    def __init__(self, data_path: str):
        """
        Initialize the data service.
        
        Args:
            data_path (str): Path to the historical data file
        """
        self.data_path = data_path
        self.historical_data: Optional[pd.DataFrame] = None

    def load_and_normalize_data(self) -> pd.DataFrame:
        """
        Load, normalize, and augment historical security tickets.
        
        This method performs a complete data processing pipeline:
        1. Loads historical data or creates sample data if not found
        2. Normalizes request types and outcomes
        3. Cleans and validates the data
        4. Augments with synthetic examples for better ML performance
        
        Returns:
            pd.DataFrame: Processed and augmented historical data
            
        Raises:
            FileNotFoundError: If data file is not found and sample creation fails
        """
        try:
            df = pd.read_csv(self.data_path)
            logger.info(f"Loaded {len(df)} historical tickets")
        except FileNotFoundError:
            logger.warning("Historical data file not found; creating comprehensive sample data.")
            df = self._create_comprehensive_sample_data()

        # Normalize request_type labels
        df = self._normalize_request_types(df)

        # Normalize outcome labels
        df = self._normalize_outcomes(df)

        # Validate and clean data
        df = self._clean_data(df)
        
        # Augment with synthetic examples to fix ML performance
        df = self._augment_with_synthetic_data(df)

        self.historical_data = df
        return df

    def _augment_with_synthetic_data(self, original_data: pd.DataFrame) -> pd.DataFrame:
        """
        Augment data with synthetic examples to resolve ML issues.
        
        This method adds carefully crafted synthetic examples to address
        specific ML classification challenges and improve model performance.
        The synthetic data covers all request types with realistic examples
        and appropriate risk scores.
        
        Args:
            original_data (pd.DataFrame): Original historical data
            
        Returns:
            pd.DataFrame: Augmented dataset with synthetic examples
        """
        logger.info("🔧 Augmenting training data to fix ML performance issues...")
        
        # Comprehensive synthetic examples addressing classification failures
        synthetic_examples = [
            # DEVTOOL_INSTALL examples (25)
            ("Install Docker Desktop for container development", "devtool_install", "Approved", 3.0),
            ("Setup Visual Studio Code with Python extensions", "devtool_install", "Approved", 2.8),
            ("Download Java JDK for enterprise development", "devtool_install", "Approved", 3.2),
            ("Install npm and Node.js for web development", "devtool_install", "Approved", 3.0),
            ("Setup Git for version control", "devtool_install", "Approved", 2.5),
            ("Install IntelliJ IDEA for Java development", "devtool_install", "Approved", 3.1),
            ("Download Python and pip for data science", "devtool_install", "Approved", 2.9),
            ("Install PostgreSQL development tools", "devtool_install", "Approved", 3.3),
            ("Setup Anaconda for machine learning projects", "devtool_install", "Approved", 3.0),
            ("Install Maven for Java project management", "devtool_install", "Approved", 2.7),
            ("Setup Docker development environment", "devtool_install", "Approved", 3.2),
            ("Install Eclipse IDE for Java programming", "devtool_install", "Approved", 2.8),
            ("Download Gradle build tool", "devtool_install", "Approved", 2.6),
            ("Install PyCharm for Python development", "devtool_install", "Approved", 3.0),
            ("Setup Node.js development environment", "devtool_install", "Approved", 2.9),
            ("Install Kubernetes CLI tools", "devtool_install", "Approved", 3.4),
            ("Download MongoDB Compass for database work", "devtool_install", "Approved", 3.1),
            ("Install Postman for API testing", "devtool_install", "Approved", 2.7),
            ("Setup Redis CLI for database access", "devtool_install", "Approved", 2.8),
            ("Install Webpack for build automation", "devtool_install", "Approved", 2.9),
            ("Download Android Studio for mobile development", "devtool_install", "Approved", 3.3),
            ("Install Babel for JavaScript compilation", "devtool_install", "Approved", 2.6),
            ("Setup yarn package manager", "devtool_install", "Approved", 2.7),
            ("Install Sublime Text editor", "devtool_install", "Approved", 2.5),
            ("Download Terraform for infrastructure code", "devtool_install", "Approved", 3.2),
            
            # NETWORK_ACCESS (distinguish from cloud) - 20 examples
            ("Need VPN access for remote work", "network_access", "Approved", 5.5),
            ("Open port 443 for HTTPS traffic", "network_access", "Approved", 6.0),
            ("Configure firewall rule for database access", "network_access", "Needs More Info", 6.5),
            ("Remote desktop access to office network", "network_access", "Approved", 5.8),
            ("WiFi access for guest devices in conference room", "network_access", "Approved", 4.5),
            ("Internet access through corporate proxy", "network_access", "Approved", 5.0),
            ("Network connectivity to partner systems", "network_access", "Needs More Info", 6.8),
            ("Firewall rule for load balancer traffic", "network_access", "Approved", 5.5),
            ("Port forwarding for development server", "network_access", "Approved", 5.2),
            ("Network access to shared file servers", "network_access", "Approved", 4.8),
            ("VPN client setup for remote employees", "network_access", "Approved", 5.3),
            ("Network bridge configuration for testing", "network_access", "Approved", 5.7),
            ("Proxy server access for development team", "network_access", "Approved", 5.1),
            ("Remote network troubleshooting access", "network_access", "Approved", 5.9),
            ("Network monitoring tool access", "network_access", "Approved", 5.4),
            ("Port 22 SSH access to development network", "network_access", "Approved", 5.6),
            ("Firewall exception for backup traffic", "network_access", "Approved", 5.2),
            ("Network segment access for testing", "network_access", "Approved", 5.3),
            ("Internet connectivity for build servers", "network_access", "Approved", 5.0),
            ("Network printer access for remote users", "network_access", "Approved", 4.7),
            
            # CLOUD_RESOURCE_ACCESS (distinguish from network) - 20 examples
            ("SSH access to AWS EC2 production instance", "cloud_resource_access", "Needs More Info", 7.2),
            ("Access to Azure virtual machine console", "cloud_resource_access", "Approved", 5.0),
            ("AWS S3 bucket access for backup storage", "cloud_resource_access", "Approved", 4.5),
            ("GCP Compute Engine access for deployment", "cloud_resource_access", "Approved", 5.5),
            ("Azure Kubernetes service access for containers", "cloud_resource_access", "Approved", 5.8),
            ("AWS Lambda function access for serverless", "cloud_resource_access", "Approved", 4.8),
            ("GCP BigQuery access for data analytics", "cloud_resource_access", "Approved", 5.2),
            ("Azure blob storage access for files", "cloud_resource_access", "Approved", 4.6),
            ("AWS RDS database instance access", "cloud_resource_access", "Needs More Info", 6.5),
            ("Cloud console access for monitoring services", "cloud_resource_access", "Approved", 5.0),
            ("EC2 instance management permissions", "cloud_resource_access", "Approved", 5.3),
            ("Azure VM scaling and management rights", "cloud_resource_access", "Approved", 5.4),
            ("GCP Cloud Functions deployment access", "cloud_resource_access", "Approved", 5.1),
            ("AWS CloudWatch monitoring dashboard", "cloud_resource_access", "Approved", 4.9),
            ("Cloud storage bucket management", "cloud_resource_access", "Approved", 5.2),
            ("Virtual machine snapshot access", "cloud_resource_access", "Approved", 5.1),
            ("Cloud load balancer configuration", "cloud_resource_access", "Approved", 5.4),
            ("Container registry access for images", "cloud_resource_access", "Approved", 4.8),
            ("Cloud database administration panel", "cloud_resource_access", "Needs More Info", 6.2),
            ("Multi-cloud resource management access", "cloud_resource_access", "Needs More Info", 6.0),
            
            # VENDOR_APPROVAL - 15 examples
            ("Onboard vendor Salesforce for CRM system", "vendor_approval", "Needs More Info", 6.0),
            ("Approve Microsoft for Office 365 services", "vendor_approval", "Approved", 4.5),
            ("Third-party integration with Slack platform", "vendor_approval", "Approved", 5.0),
            ("Contractor approval for security audit services", "vendor_approval", "Needs More Info", 6.5),
            ("Vendor assessment for AWS cloud partnership", "vendor_approval", "Approved", 5.2),
            ("Supplier approval for hardware procurement", "vendor_approval", "Approved", 4.8),
            ("Partner integration with payment processor", "vendor_approval", "Needs More Info", 6.8),
            ("Security review for new SaaS provider", "vendor_approval", "Needs More Info", 6.2),
            ("Due diligence for vendor onboarding", "vendor_approval", "Needs More Info", 5.8),
            ("Onboard contractor for development work", "vendor_approval", "Approved", 5.5),
            ("Third-party analytics service approval", "vendor_approval", "Approved", 5.3),
            ("Vendor security questionnaire review", "vendor_approval", "Needs More Info", 5.9),
            ("Supplier contract negotiation support", "vendor_approval", "Approved", 5.1),
            ("Partner API integration approval", "vendor_approval", "Approved", 5.4),
            ("External consultant engagement approval", "vendor_approval", "Needs More Info", 6.1),
            
            # PERMISSION_CHANGE - 15 examples
            ("Admin access to production database server", "permission_change", "Rejected", 8.5),
            ("Elevated permissions for system maintenance", "permission_change", "Needs More Info", 7.0),
            ("Temporary sudo access for security patching", "permission_change", "Approved", 6.5),
            ("Root access for critical security updates", "permission_change", "Needs More Info", 7.5),
            ("Database admin rights for schema migration", "permission_change", "Needs More Info", 7.2),
            ("Administrative privileges for deployment", "permission_change", "Approved", 6.8),
            ("Privilege escalation for incident response", "permission_change", "Approved", 7.0),
            ("System admin access for monitoring setup", "permission_change", "Approved", 6.2),
            ("Network admin rights for configuration", "permission_change", "Needs More Info", 6.8),
            ("Application admin access for updates", "permission_change", "Approved", 6.0),
            ("Temporary admin rights for troubleshooting", "permission_change", "Approved", 6.3),
            ("Elevated access for backup restoration", "permission_change", "Needs More Info", 6.9),
            ("Administrative permissions for audit", "permission_change", "Needs More Info", 7.1),
            ("System administrator role assignment", "permission_change", "Needs More Info", 6.7),
            ("Database administrator privileges", "permission_change", "Needs More Info", 7.3),
            
            # DATA_EXPORT - 15 examples
            ("Export customer data for compliance audit", "data_export", "Needs More Info", 7.0),
            ("Access to sales database for quarterly analysis", "data_export", "Approved", 5.5),
            ("Query user logs for security investigation", "data_export", "Needs More Info", 6.8),
            ("Extract financial data for regulatory audit", "data_export", "Rejected", 8.0),
            ("Database access for business intelligence", "data_export", "Approved", 5.2),
            ("Customer information export for support", "data_export", "Approved", 4.8),
            ("Application logs access for debugging", "data_export", "Approved", 5.0),
            ("User data extraction for system migration", "data_export", "Needs More Info", 6.5),
            ("Transaction data query for fraud analysis", "data_export", "Needs More Info", 6.8),
            ("System metrics export for performance report", "data_export", "Approved", 4.5),
            ("Historical data extraction for analytics", "data_export", "Approved", 5.3),
            ("Database backup for disaster recovery", "data_export", "Approved", 5.1),
            ("Customer survey data export", "data_export", "Approved", 4.9),
            ("Employee data access for HR audit", "data_export", "Needs More Info", 6.4),
            ("Marketing data extraction for campaign", "data_export", "Approved", 5.2),
            
            # FIREWALL_CHANGE - 15 examples
            ("Update firewall rules for new microservice", "firewall_change", "Approved", 5.5),
            ("Modify AWS security group for load balancer", "firewall_change", "Approved", 5.8),
            ("Configure iptables for container networking", "firewall_change", "Approved", 6.0),
            ("Add firewall rule for partner integration", "firewall_change", "Needs More Info", 6.5),
            ("Update perimeter firewall for API gateway", "firewall_change", "Approved", 5.8),
            ("Security group change for database cluster", "firewall_change", "Needs More Info", 6.2),
            ("Firewall configuration for backup service", "firewall_change", "Approved", 5.0),
            ("Network ACL update for multi-tier app", "firewall_change", "Approved", 5.5),
            ("Firewall rule for monitoring system access", "firewall_change", "Approved", 5.2),
            ("Security policy update for compliance", "firewall_change", "Approved", 5.8),
            ("Firewall exception for partner connectivity", "firewall_change", "Needs More Info", 6.3),
            ("Security group modification for scaling", "firewall_change", "Approved", 5.4),
            ("Firewall rule update for new environment", "firewall_change", "Approved", 5.6),
            ("Network security policy adjustment", "firewall_change", "Needs More Info", 6.1),
            ("Firewall configuration for disaster recovery", "firewall_change", "Approved", 5.7)
        ]
        
        # Get current distribution
        current_counts = original_data['request_type'].value_counts()
        logger.info(f"Original distribution: {current_counts.to_dict()}")
        
        # Add synthetic examples
        synthetic_df = pd.DataFrame(synthetic_examples, 
                                  columns=['request_summary', 'request_type', 'outcome', 'security_risk_score'])
        
        # Combine original and synthetic data
        augmented_data = pd.concat([original_data, synthetic_df], ignore_index=True)
        
        # Final statistics
        final_counts = augmented_data['request_type'].value_counts()
        logger.info(f"Augmented dataset: {len(augmented_data)} total (added {len(synthetic_df)} synthetic examples)")
        logger.info(f"Final distribution: {final_counts.to_dict()}")
        
        return augmented_data

    def _normalize_request_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize request type labels to match our enum.
        
        This method standardizes request type labels by:
        1. Converting to lowercase and stripping whitespace
        2. Mapping legacy values to current enum values
        3. Handling invalid types by setting them to OTHER
        
        Args:
            df (pd.DataFrame): Input dataframe with request_type column
            
        Returns:
            pd.DataFrame: Dataframe with normalized request types
        """
        mapping = {
            # Map CSV values to enum values
            'vendor approval': RequestType.VENDOR_APPROVAL.value,
            'permission change': RequestType.PERMISSION_CHANGE.value,
            'permission escalation': RequestType.PERMISSION_CHANGE.value,  # Legacy mapping
            'network access': RequestType.NETWORK_ACCESS.value,
            'firewall change': RequestType.FIREWALL_CHANGE.value,
            'devtool install': RequestType.DEVTOOL_INSTALL.value,
            'data export': RequestType.DATA_EXPORT.value,
            'cloud resource access': RequestType.CLOUD_RESOURCE_ACCESS.value,
            'data access': RequestType.DATA_EXPORT.value,
            'system access': RequestType.CLOUD_RESOURCE_ACCESS.value,
        }

        df['request_type'] = (
            df['request_type']
            .astype(str)
            .str.strip()
            .str.lower()
            .replace(mapping)
        )

        # Set invalid types to OTHER
        valid_types = {e.value for e in RequestType}
        df['request_type'] = df['request_type'].where(
            df['request_type'].isin(valid_types),
            other=RequestType.OTHER.value
        )

        return df

    def _normalize_outcomes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize outcome labels to match our enum.
        
        This method standardizes outcome labels by:
        1. Converting to uppercase and stripping whitespace
        2. Mapping legacy values to current enum values
        3. Handling invalid outcomes by setting them to NEEDS_MORE_INFO
        
        Args:
            df (pd.DataFrame): Input dataframe with outcome column
            
        Returns:
            pd.DataFrame: Dataframe with normalized outcomes
        """
        df['outcome'] = (
            df['outcome']
            .astype(str)
            .str.strip()
            .str.upper()
            .replace({
                'APPROVE': Outcome.APPROVED.value,
                'DENY': Outcome.REJECTED.value,
                'REJECT': Outcome.REJECTED.value,
                'MORE INFO': Outcome.NEEDS_MORE_INFO.value,
                'INFO NEEDED': Outcome.NEEDS_MORE_INFO.value,
                'PENDING': Outcome.NEEDS_MORE_INFO.value
            })
        )

        # Set invalid outcomes to NEEDS_MORE_INFO
        valid_outcomes = {e.value for e in Outcome}
        df['outcome'] = df['outcome'].where(
            df['outcome'].isin(valid_outcomes),
            other=Outcome.NEEDS_MORE_INFO.value
        )

        return df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate the dataset.
        
        This method performs data cleaning operations:
        1. Removes duplicate entries
        2. Handles missing values
        3. Validates data types
        4. Removes invalid entries
        
        Args:
            df (pd.DataFrame): Input dataframe to clean
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        # Remove duplicates
        df = df.drop_duplicates(subset=['request_summary', 'request_type', 'outcome'])
        
        # Handle missing values
        df['request_summary'] = df['request_summary'].fillna('')
        df['request_type'] = df['request_type'].fillna(RequestType.OTHER.value)
        df['outcome'] = df['outcome'].fillna(Outcome.NEEDS_MORE_INFO.value)
        df['security_risk_score'] = df['security_risk_score'].fillna(5.0)
        
        # Validate risk scores
        df['security_risk_score'] = df['security_risk_score'].clip(1.0, 10.0)
        
        return df

    def _create_comprehensive_sample_data(self) -> pd.DataFrame:
        """
        Create comprehensive sample data for testing and development.
        
        This method generates a diverse set of sample security requests
        covering all request types and outcomes with realistic examples.
        
        Returns:
            pd.DataFrame: Generated sample data
        """
        # Use a subset of synthetic examples as sample data
        sample_data = pd.DataFrame([
            # Include a few examples from each category
            ("Install Docker Desktop for container development", "devtool_install", "Approved", 3.0),
            ("Need VPN access for remote work", "network_access", "Approved", 5.5),
            ("SSH access to AWS EC2 production instance", "cloud_resource_access", "Needs More Info", 7.2),
            ("Onboard vendor Salesforce for CRM system", "vendor_approval", "Needs More Info", 6.0),
            ("Admin access to production database server", "permission_change", "Rejected", 8.5),
            ("Export customer data for compliance audit", "data_export", "Needs More Info", 7.0),
            ("Update firewall rules for new microservice", "firewall_change", "Approved", 5.5)
        ], columns=['request_summary', 'request_type', 'outcome', 'security_risk_score'])
        
        return sample_data