# Openai-study-mode-clone
Built a multiagent Study Mode AI platform using OpenAI SDK and Xero MCP Server, adapted to work with Google AI Studio API. Features a chatbot interface that intelligently handles study queries, performing tasks like data preprocessing, model recommendations, and parallel inference. Demonstrates integration of a Mixture-of-Experts framework.
# Study Mode Clone

An AI-powered study assistant application built with a Mixture of Experts (MoE) approach using Google AI Studio API. This application provides intelligent study support through multiple specialized AI agents working together.

## 🌟 Features

- **🤖 Multiple AI Agents**: Specialized agents for different tasks
  - Data Processing Agent: Convert text to structured formats
  - Data Preprocessing Agent: Clean and preprocess data
  - Model Suggestion Agent: Recommend appropriate AI models
  - Parallel Inference Agent: Run multiple models simultaneously

- **🧠 Mixture of Experts (MoE)**: Intelligent task routing and agent orchestration
- **💬 Chatbot Interface**: User-friendly Streamlit-based interface
- **📊 Data Analysis**: Comprehensive data processing and analysis capabilities
- **⚡ Parallel Processing**: Efficient multi-model inference
- **🔧 MCP Server**: Model Context Protocol for agent communication

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Google AI Studio API key
- Xero API credentials (optional)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd study-mode-clone
   ```

2. **Install dependencies:**
   ```bash
   pip install -e .
   ```

3. **Set up environment variables:**
   Create a `.env` file in the root directory:
   ```env
   # Google AI Studio API Configuration
   GOOGLE_API_KEY=your_google_ai_studio_api_key_here
   
   # Xero API Configuration (optional)
   XERO_CLIENT_ID=your_xero_client_id_here
   XERO_CLIENT_SECRET=your_xero_client_secret_here
   
   # Application Configuration
   APP_HOST=localhost
   APP_PORT=8000
   DEBUG=True
   ```

4. **Run the application:**
   ```bash
   streamlit run app.py
   ```

## 📖 Usage Examples

### Basic Usage

```python
import asyncio
from agent_orchestrator import orchestrator

async def ask_question():
    result = await orchestrator.process_study_request(
        "Explain machine learning concepts",
        context={"user_id": "user_001", "session_id": "session_001"}
    )
    
    print(result.get("response"))

asyncio.run(ask_question())
```

### Data Processing

```python
from agents.data_agent import DataProcessingAgent

async def process_data():
    agent = DataProcessingAgent()
    
    text_data = """
    Name: John Doe, Age: 25, Occupation: Engineer
    Name: Jane Smith, Age: 30, Occupation: Scientist
    """
    
    df = await agent.process_text_to_dataframe(text_data)
    print(df)

asyncio.run(process_data())
```

### Parallel Model Inference

```python
from agents.inference_agent import ParallelInferenceAgent

async def parallel_inference():
    agent = ParallelInferenceAgent()
    
    result = await agent.run_parallel_inference(
        "Explain quantum computing",
        models=["gemini-pro", "gemini-ultra"]
    )
    
    print(result)

asyncio.run(parallel_inference())
```

## 🏗️ Architecture

### Core Components

1. **Agent Orchestrator**: Routes tasks to appropriate agents using MoE approach
2. **MCP Server**: Handles agent communication and context management
3. **Specialized Agents**: 
   - Data Processing Agent
   - Data Preprocessing Agent
   - Model Suggestion Agent
   - Parallel Inference Agent
4. **Streamlit Interface**: User-friendly web interface

### Agent Capabilities

#### Data Processing Agent
- Convert unstructured text to DataFrames
- Extract tabular data from text
- Parse structured content (JSON, CSV, etc.)
- Validate data structures

#### Data Preprocessing Agent
- Analyze data quality
- Clean and preprocess data
- Handle missing values
- Normalize and scale data
- Feature engineering

#### Model Suggestion Agent
- Analyze task requirements
- Recommend appropriate models
- Optimize model parameters
- Compare model performance
- Estimate costs

#### Parallel Inference Agent
- Run multiple models in parallel
- Ensemble inference
- Load balancing
- Performance monitoring
- Error handling

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GOOGLE_API_KEY` | Google AI Studio API key | Yes |
| `XERO_CLIENT_ID` | Xero API client ID | No |
| `XERO_CLIENT_SECRET` | Xero API client secret | No |
| `APP_HOST` | Application host | No |
| `APP_PORT` | Application port | No |
| `DEBUG` | Debug mode | No |

### Model Configuration

The application supports multiple Google AI models:
- `gemini-pro`: General-purpose text generation
- `gemini-pro-vision`: Multimodal (text + images)
- `gemini-ultra`: Advanced reasoning and analysis

## 📊 Performance Monitoring

The application includes comprehensive performance monitoring:

- Task execution metrics
- Agent performance tracking
- Success rate monitoring
- Execution time analysis
- Health status checks

## 🧪 Testing

Run the example scripts to test functionality:

```bash
# Basic examples
python examples/basic_usage.py

# Advanced examples
python examples/advanced_usage.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the examples directory for usage patterns
- Review the configuration documentation

## 🔮 Future Enhancements

- [ ] Additional AI model integrations
- [ ] Enhanced visualization capabilities
- [ ] Real-time collaboration features
- [ ] Advanced analytics dashboard
- [ ] API endpoint development
- [ ] Docker containerization

---

Built with ❤️ using Google AI Studio API, Streamlit, and Python.
