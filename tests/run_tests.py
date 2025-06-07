import subprocess
import sys
import os
from pathlib import Path
import argparse


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'=' * 60}")
    print(f"🏃 {description}")
    print(f"{'=' * 60}")

    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout)

    if result.stderr and result.returncode != 0:
        print(f"❌ Error: {result.stderr}")
        return False

    if result.returncode == 0:
        print(f"✅ {description} completed successfully")
    else:
        print(f"❌ {description} failed")

    return result.returncode == 0


def setup_minimal_test_environment():
    """Setup minimal test environment"""
    print("🔧 Setting up minimal test environment...")

    # Create test directories
    os.makedirs("tests", exist_ok=True)
    os.makedirs("test_reports", exist_ok=True)

    # Install minimal test dependencies
    return run_command(
        "pip install pytest pytest-cov pytest-mock",
        "Installing minimal test dependencies"
    )


def run_unit_tests():
    """Run unit tests with basic coverage"""
    return run_command(
        "pytest tests/ -m 'not integration and not slow' --cov=src --cov-report=term",
        "Running unit tests with coverage"
    )


def run_integration_tests():
    """Run integration tests"""
    return run_command(
        "pytest tests/ -m integration --tb=long",
        "Running integration tests"
    )


def run_all_tests():
    """Run all tests with coverage"""
    return run_command(
        "pytest tests/ --cov=src --cov-report=term --cov-report=html",
        "Running all tests with coverage"
    )


def run_specific_test(test_path):
    """Run specific test file or function"""
    return run_command(
        f"pytest {test_path} -v",
        f"Running specific test: {test_path}"
    )


def check_pytest_installation():
    """Check if pytest is properly installed"""
    try:
        result = subprocess.run(["pytest", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Pytest installed: {result.stdout.strip()}")
            return True
        else:
            print("❌ Pytest not found")
            return False
    except FileNotFoundError:
        print("❌ Pytest not found")
        return False


def main():
    """Main test runner with error handling"""
    parser = argparse.ArgumentParser(description="ACME Security Bot Test Runner (Fixed)")
    parser.add_argument("--setup", action="store_true", help="Setup test environment")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--test", type=str, help="Run specific test file or function")
    parser.add_argument("--check", action="store_true", help="Check test environment")

    args = parser.parse_args()

    if not any(vars(args).values()):
        # Default: run all tests
        args.all = True

    print("🚀 ACME Security Bot Test Suite (Fixed)")
    print("=" * 60)

    # Check environment
    if args.check or not check_pytest_installation():
        if not setup_minimal_test_environment():
            print("❌ Failed to setup test environment")
            sys.exit(1)

        # Verify installation
        if not check_pytest_installation():
            print("❌ Pytest installation failed")
            sys.exit(1)

    success = True

    if args.setup:
        success &= setup_minimal_test_environment()

    if args.unit:
        success &= run_unit_tests()

    if args.integration:
        success &= run_integration_tests()

    if args.test:
        success &= run_specific_test(args.test)

    if args.all:
        success &= run_all_tests()

    print(f"\n{'=' * 60}")
    if success:
        print("✅ All tests completed successfully!")
        print("📊 Check htmlcov/index.html for detailed coverage report")
    else:
        print("❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
