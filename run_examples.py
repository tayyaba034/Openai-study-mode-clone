"""
Run Example Scripts for Study Mode Clone
This script demonstrates how to run the example files and test the application.
"""

import asyncio
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def run_basic_examples():
    """Run basic usage examples."""
    print("🚀 Running Basic Usage Examples...\n")
    
    try:
        from examples.basic_usage import main as basic_main
        await basic_main()
        print("✅ Basic examples completed successfully!\n")
        return True
    except Exception as e:
        print(f"❌ Error running basic examples: {e}\n")
        return False

async def run_advanced_examples():
    """Run advanced usage examples."""
    print("🚀 Running Advanced Usage Examples...\n")
    
    try:
        from examples.advanced_usage import main as advanced_main
        await advanced_main()
        print("✅ Advanced examples completed successfully!\n")
        return True
    except Exception as e:
        print(f"❌ Error running advanced examples: {e}\n")
        return False

def check_environment():
    """Check if the environment is properly configured."""
    print("🔍 Checking Environment Configuration...\n")
    
    # Check for .env file
    if not os.path.exists('.env'):
        print("⚠️  Warning: .env file not found. Please create one with your API keys.")
        return False
    
    # Check for required modules
    try:
        import google.generativeai
        import streamlit
        import pandas
        import numpy
        print("✅ All required packages are installed.")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please run: pip install -e .")
        return False

async def main():
    """Main function to run all examples."""
    print("=" * 60)
    print("📚 Study Mode Clone - Example Runner")
    print("=" * 60)
    print()
    
    # Check environment
    if not check_environment():
        print("\n❌ Environment check failed. Please fix the issues above.")
        return
    
    print()
    
    # Run examples
    basic_success = await run_basic_examples()
    advanced_success = await run_advanced_examples()
    
    # Summary
    print("=" * 60)
    print("📊 Summary:")
    print(f"Basic Examples: {'✅ Passed' if basic_success else '❌ Failed'}")
    print(f"Advanced Examples: {'✅ Passed' if advanced_success else '❌ Failed'}")
    
    if basic_success and advanced_success:
        print("\n🎉 All examples completed successfully!")
        print("You can now run the main application with: streamlit run app.py")
    else:
        print("\n⚠️  Some examples failed. Please check the errors above.")
    
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
