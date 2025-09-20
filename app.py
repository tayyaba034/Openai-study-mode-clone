"""
Study Mode Clone - Main Application
AI-powered study assistant with Mixture of Experts approach using Google AI Studio API.
"""

import asyncio
import json
import logging
import streamlit as st
from typing import Dict, Any, Optional
from datetime import datetime
import pandas as pd

# Import our modules
from config import settings
from agent_orchestrator import orchestrator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main function to run the Study Mode application."""
    try:
        # Validate configuration
        if not settings.google_api_key:
            st.error("❌ Google AI Studio API key not found. Please check your .env file.")
            st.stop()
        
        # Application header
        st.set_page_config(
            page_title="Study Mode Clone",
            page_icon="📚",
            layout="wide"
        )
        
        st.title("📚 Study Mode Clone")
        st.markdown("**AI-Powered Study Assistant with Mixture of Experts**")
        st.markdown("Powered by Google AI Studio API • Built with Python")
        
        # Chat interface
        st.header("💬 Chat Interface")
        
        # Initialize chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        # Chat input
        user_input = st.chat_input("Ask me anything about your studies...")
        
        if user_input:
            # Add user message
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now().isoformat()
            })
            
            # Process request
            with st.spinner("🤔 Thinking..."):
                try:
                    context = {
                        "user_id": "user_001",
                        "session_id": f"session_{int(datetime.now().timestamp())}",
                        "model": "gemini-pro",
                        "complexity": "moderate"
                    }
                    
                    result = asyncio.run(orchestrator.process_study_request(user_input, context))
                    
                    if result.get("success"):
                        response_content = result.get("response", "No response generated")
                    else:
                        response_content = f"Error: {result.get('error', 'Unknown error')}"
                    
                    # Add assistant response
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response_content,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                except Exception as e:
                    error_response = f"Sorry, I encountered an error: {str(e)}"
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": error_response,
                        "timestamp": datetime.now().isoformat()
                    })
            
            st.rerun()
        
        # Display chat history
        chat_history = st.session_state.chat_history
        
        if not chat_history:
            st.info("👋 Welcome! Ask me anything about your studies.")
            st.markdown("""
            **I can help with:**
            - 📊 Data Analysis and Processing
            - 🧹 Data Cleaning and Preprocessing
            - 🤖 Model Recommendations
            - ⚡ Parallel Model Inference
            - 📚 Study Questions and Explanations
            """)
        else:
            for message in chat_history:
                if message["role"] == "user":
                    with st.chat_message("user"):
                        st.write(message["content"])
                else:
                    with st.chat_message("assistant"):
                        st.write(message["content"])
        
        # Sidebar controls
        with st.sidebar:
            st.header("🎛️ Controls")
            
            # Agent status
            if st.button("🔄 Check Agent Status"):
                with st.spinner("Checking..."):
                    health_status = asyncio.run(orchestrator.health_check())
                    st.write("**Agent Status:**")
                    for agent_id, status in health_status.get("agents", {}).items():
                        if status.get("status") == "healthy":
                            st.success(f"✅ {agent_id}")
                        else:
                            st.error(f"❌ {agent_id}")
            
            # Clear chat
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()
            
            # Performance metrics
            st.header("📊 Performance")
            metrics = orchestrator.get_performance_metrics()
            st.write(f"**Total Tasks:** {metrics.get('total_tasks', 0)}")
            success_rate = metrics.get('overall_success_rate', 0) * 100
            st.write(f"**Success Rate:** {success_rate:.1f}%")
        
        # Footer
        st.markdown("---")
        st.markdown("Built with ❤️ using Google AI Studio API and Python")
        
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        st.error(f"Fatal error: {str(e)}")

if __name__ == "__main__":
    main()
