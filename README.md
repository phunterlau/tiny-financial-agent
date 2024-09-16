# Tiny financial analysis agent
TL,DR A simple tool-using and orchestration financial analysis agent example, no Lllama-index no langchain no CrewAI.

This repository contains a flexible, lightweight framework for building AI agents, demonstrated through a financial analysis example. The framework is designed to be independent of complex libraries, fully customizable, and ready to use for various agent scenarios involving tool use and high-level functions.

## Key Features

- **Lightweight Design**: Built with minimal dependencies, focusing on core Python libraries and API interactions.
- **Flexible Architecture**: Separates atomic tools from orchestration functions, allowing for easy customization and extension.
- **Language Model Integration**: Leverages language models for intelligent function selection and complex reasoning.
- **Scalable Complexity**: Handles both simple queries and complex, multi-faceted analyses.

## Design Philosophy

1. **Independence**: Free from reliance on heavyweight agent frameworks like LangChain or LlamaIndex.
2. **Customization**: Easily adaptable to various domains beyond financial analysis.
3. **Separation of Concerns**:
   - Low-level functions (tools) for basic data retrieval and simple operations. Give this task to a small tool-using LLM like GPT-4o-mini.
   - High-level functions (orchestrations) for complex analyses and decision-making. Give this task to a large CoT LLM like OpenAI o1.
4. **Smart Orchestration**: Utilizes language models' chain-of-thought capabilities to select and combine tools effectively.

## Structure

- `atomic_tools.py`: Contains basic data retrieval and processing functions.
- `orchestration.py`: Defines high-level orchestration functions for complex analyses.
- `driver.py`: Implements the core agent logic, including the `FunctionCallingAgent`.

## Usage

1. Set up your API keys in environment variables.
```
pip install openai colorama
```

2. Customize atomic tools and orchestration functions as needed.
3. Run the agent through the `driver.py` script.

Example:
```python
export OPENAI_API_KEY=sk-proj-....
export FINANCIAL_MODELING_PREP_API_KEY=...
python driver.py
```

## Extensibility

- Add new atomic tools in `atomic_tools.py` for additional data sources or operations.
- Create new orchestration functions in `orchestration.py` for different types of analyses.
- Modify the `FunctionCallingAgent` in `driver.py` to alter the agent's decision-making process.

## Design Thoughts

This framework exemplifies a "smart orchestration, simple tools" approach:

- **Simple Tools for Efficiency**: Low-level functions (tools) are designed to be straightforward and efficient, suitable for execution by smaller, faster models like GPT-4o-mini.
- **Smart Orchestration for Complexity**: High-level functions (orchestrations) leverage the reasoning capabilities of more advanced models (like OpenAI o1 or Claude Opus) to handle complex queries and decision-making.
- **Flexible Integration**: The framework allows easy integration of different language models for various tasks, enabling a balance between efficiency and sophisticated reasoning.

By following this design, the framework remains lightweight and flexible, while still being capable of handling complex scenarios through intelligent orchestration of simple tools.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.