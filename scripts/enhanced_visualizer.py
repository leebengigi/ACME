#!/usr/bin/env python3
"""
Enhanced Visualizer for Classification Performance
"""

import sys
from pathlib import Path
import pandas as pd
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_comprehensive_test():
    """Run comprehensive classification test"""
    
    print("🚀 Enhanced Classification Performance Test")
    print("="*50)
    
    try:
        # Import the enhanced classifier
        from src.services.classification_service import ClassificationService
        from src.models.enums import RequestType
        
        # Create classifier
        classifier = ClassificationService()
        
        # Try to load and train with CSV data
        try:
            data = pd.read_csv("acme_security_tickets.csv")
            print(f"📚 Training with {len(data)} historical samples...")
            training_results = classifier.train(data)
            print(f"✅ Training complete: {training_results}")
        except FileNotFoundError:
            print("⚠️ acme_security_tickets.csv not found - using rule-based only")
        except Exception as e:
            print(f"⚠️ Training failed: {e} - using rule-based only")
        
        # Comprehensive test cases
        test_cases = [
            # DevTool Install
            ("Install Docker Desktop for development", RequestType.DEVTOOL_INSTALL),
            ("Setup Visual Studio Code with extensions", RequestType.DEVTOOL_INSTALL),
            ("Download Java JDK for programming", RequestType.DEVTOOL_INSTALL),
            
            # Permission Change
            ("Need admin access to production database", RequestType.PERMISSION_CHANGE),
            ("Temporary elevated permissions for maintenance", RequestType.PERMISSION_CHANGE),
            ("Root access for system updates", RequestType.PERMISSION_CHANGE),
            
            # Network Access
            ("Open port 443 for HTTPS traffic", RequestType.NETWORK_ACCESS),
            ("VPN access for remote work", RequestType.NETWORK_ACCESS),
            ("Network connectivity to partner systems", RequestType.NETWORK_ACCESS),
            
            # Cloud Resource Access
            ("SSH access to AWS EC2 instance", RequestType.CLOUD_RESOURCE_ACCESS),
            ("Access to Azure virtual machine", RequestType.CLOUD_RESOURCE_ACCESS),
            ("AWS S3 bucket permissions", RequestType.CLOUD_RESOURCE_ACCESS),
            
            # Data Export
            ("Export customer data for analysis", RequestType.DATA_EXPORT),
            ("Database access for reporting", RequestType.DATA_EXPORT),
            ("Query user logs for investigation", RequestType.DATA_EXPORT),
            
            # Vendor Approval
            ("Onboard vendor Microsoft for integration", RequestType.VENDOR_APPROVAL),
            ("Approve contractor for security audit", RequestType.VENDOR_APPROVAL),
            ("Third-party service approval", RequestType.VENDOR_APPROVAL),
            
            # Challenging cases
            ("Install Docker and get server access", RequestType.DEVTOOL_INSTALL),
            ("Urgent admin access needed ASAP", RequestType.PERMISSION_CHANGE),
        ]
        
        print(f"\n🧪 Running {len(test_cases)} comprehensive tests...")
        
        results = []
        start_time = time.time()
        
        for text, expected in test_cases:
            test_start = time.time()
            predicted, confidence = classifier.classify_request(text)
            test_end = time.time()
            
            is_correct = predicted == expected
            processing_time = (test_end - test_start) * 1000
            
            results.append({
                'text': text,
                'expected': expected.value,
                'predicted': predicted.value,
                'confidence': confidence,
                'correct': is_correct,
                'time_ms': processing_time
            })
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        correct = sum(1 for r in results if r['correct'])
        total = len(results)
        accuracy = correct / total
        avg_confidence = sum(r['confidence'] for r in results) / total
        avg_time = sum(r['time_ms'] for r in results) / total
        
        # Display detailed results
        print(f"\n📊 DETAILED RESULTS:")
        for i, result in enumerate(results, 1):
            status = "✅" if result['correct'] else "❌"
            print(f"{i:2d}. {status} {result['predicted']:20} (conf: {result['confidence']:.2f}) | \"{result['text'][:60]}...\"")
        
        # Summary
        print(f"\n📈 PERFORMANCE SUMMARY:")
        print(f"   Total Tests: {total}")
        print(f"   Correct: {correct}")
        print(f"   Accuracy: {accuracy:.1%}")
        print(f"   Average Confidence: {avg_confidence:.2f}")
        print(f"   Average Processing Time: {avg_time:.1f}ms")
        print(f"   Total Test Time: {total_time:.2f}s")
        
        # Target achievement
        target_achieved = accuracy >= 0.9
        print(f"\n🎯 TARGET ACHIEVEMENT: {'✅ SUCCESS' if target_achieved else '❌ NEEDS IMPROVEMENT'}")
        print(f"   Required: 90%+ accuracy")
        print(f"   Achieved: {accuracy:.1%}")
        
        # Error analysis
        errors = [r for r in results if not r['correct']]
        if errors:
            print(f"\n❌ ERRORS ({len(errors)} total):")
            for error in errors:
                print(f"   {error['expected']} → {error['predicted']}: \"{error['text'][:60]}...\"")
        else:
            print(f"\n🎉 PERFECT CLASSIFICATION: No errors detected!")
        
        return target_achieved
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    print("\n" + "="*50)
    if success:
        print("🏆 ENHANCED CLASSIFIER ACHIEVES 90%+ ACCURACY!")
    else:
        print("📊 Additional optimization recommended.")
    print("="*50)