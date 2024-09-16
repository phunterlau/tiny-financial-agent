import os
from typing import List, Dict, Any
from pydantic import BaseModel
from openai import OpenAI
from atomic_tools import get_stock_price, get_company_financials, get_income_statement, compare_stocks
#from orchestration import SectorAnalysis, PortfolioRecommendation, FinancialHealthAssessment, StrategicInvestmentAnalysis, MarketTrendPrediction, CompanyCompetitiveAnalysis

from orchestration import get_all_orchestration_functions
import json

# Use system environment variable for OpenAI API key
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

class FunctionTool(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]
    
    model_config = {
        "extra": "forbid"
    }

class FunctionCallingAgent:
    def __init__(self, tools: List[FunctionTool], llm: Any):
        self.tools = tools
        self.llm = llm
        self.memory = []
        self.orchestration_functions = self._load_orchestration_functions()

    def _load_orchestration_functions(self):
        orchestration_classes = get_all_orchestration_functions()
        return {
            name: cls(self.llm)
            for name, cls in orchestration_classes.items()
        }

    def chat(self, query: str) -> str:
        print(f"Processing query: {query}")
        self.memory.append({"role": "user", "content": query})
        
        print("Determining appropriate function to call...")
        response = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.memory,
            functions=[tool.model_dump() for tool in self.tools],
            function_call="auto"
        )
        
        if response.choices[0].message.function_call:
            function_call = response.choices[0].message.function_call
            function_name = function_call.name
            function_args = json.loads(function_call.arguments)
            
            print(f"Calling function: {function_name}")
            if function_name in self.orchestration_functions:
                result = self.orchestration_functions[function_name].execute(**function_args)
            elif hasattr(self, function_name):
                result = getattr(self, function_name)(**function_args)
            else:
                result = f"Error: Function {function_name} not found."
            
            self.memory.append({"role": "function", "name": function_name, "content": str(result)})
        
        print("Generating final response...")
        final_response = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.memory
        )
        
        self.memory.append({"role": "assistant", "content": final_response.choices[0].message.content})
        return final_response.choices[0].message.content
 
# Define tools
tools = [
    FunctionTool(
        name="get_stock_price",
        description="Get current stock price and related information",
        parameters={"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}
    ),
    FunctionTool(
        name="get_company_financials",
        description="Get company financial information",
        parameters={"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}
    ),
    FunctionTool(
        name="get_income_statement",
        description="Get company income statement",
        parameters={"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}
    ),
    FunctionTool(
        name="compare_stocks",
        description="Compare two stocks",
        parameters={
            "type": "object",
            "properties": {
                "symbol1": {"type": "string"},
                "symbol2": {"type": "string"}
            },
            "required": ["symbol1", "symbol2"]
        }
    ),
    FunctionTool(
        name="sector_analysis",
        description="Analyze a specific sector",
        parameters={
            "type": "object",
            "properties": {
                "sector": {"type": "string"},
                "top_n": {"type": "integer", "default": 5}
            },
            "required": ["sector"]
        }
    ),
    FunctionTool(
        name="portfolio_recommendation",
        description="Get portfolio recommendations based on risk tolerance and sectors",
        parameters={
            "type": "object",
            "properties": {
                "risk_tolerance": {"type": "string", "enum": ["low", "medium", "high"]},
                "investment_amount": {"type": "number"},
                "sectors": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["risk_tolerance", "investment_amount", "sectors"]
        }
    ),
    FunctionTool(
        name="financial_health_assessment",
        description="Assess the financial health of a company",
        parameters={
            "type": "object",
            "properties": {
                "symbol": {"type": "string"}
            },
            "required": ["symbol"]
        }
    ),
    FunctionTool(
        name="strategic_investment_analysis",
        description="Perform a strategic investment analysis for a company",
        parameters={
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "time_horizon": {"type": "string", "enum": ["short-term", "medium-term", "long-term"]}
            },
            "required": ["symbol", "time_horizon"]
        }
    ),
    FunctionTool(
        name="market_trend_prediction",
        description="Predict market trends for a specific sector",
        parameters={
            "type": "object",
            "properties": {
                "sector": {"type": "string"},
                "timeframe": {"type": "string", "enum": ["6 months", "1 year", "3 years", "5 years"]}
            },
            "required": ["sector", "timeframe"]
        }
    ),
    FunctionTool(
        name="company_competitive_analysis",
        description="Perform a competitive analysis for a company",
        parameters={
            "type": "object",
            "properties": {
                "symbol": {"type": "string"}
            },
            "required": ["symbol"]
        }
    ),
    FunctionTool(
        name="company_comparative_analysis",
        description="Perform a comparative analysis of two companies, including competitive analysis and investment potential",
        parameters={
            "type": "object",
            "properties": {
                "symbol1": {"type": "string"},
                "symbol2": {"type": "string"},
                "time_horizon": {"type": "string"}
            },
            "required": ["symbol1", "symbol2", "time_horizon"]
        }
    ),
]

# Initialize agent
agent = FunctionCallingAgent(tools, client)

example_queries ="""
Example queries for the Financial Analysis Assistant:
Perform a strategic investment analysis for Tesla (TSLA) with a medium-term time horizon.
Conduct a strategic investment analysis for Amazon (AMZN) considering a long-term perspective.
Analyze Nvidia (NVDA) as a potential investment opportunity for the next 5 years.
Predict market trends for the renewable energy sector over the next 3 years.
What are the projected trends in the healthcare sector for the coming year?
Analyze and forecast the trends in the artificial intelligence industry for the next 5 years.
Conduct a competitive analysis for Apple (AAPL) in the consumer electronics market.
Perform a detailed competitive analysis of Coca-Cola (KO) versus its major rivals.
Analyze the competitive position of JPMorgan Chase (JPM) in the banking industry.
Compare the investment potential of Microsoft (MSFT) and Google (GOOGL) over the next 3 years, including a competitive analysis of both companies.
Analyze the renewable energy sector trends for the next 5 years and provide a strategic investment analysis for First Solar (FSLR) in this context.
Conduct a comprehensive analysis of the electric vehicle market, including a competitive analysis of Tesla (TSLA) and a strategic investment analysis for the company over a 5-year horizon.
Considering the current economic climate, analyze the banking sector trends for the next 2 years and provide a comparative strategic investment analysis for JPMorgan Chase (JPM) and Bank of America (BAC).
Evaluate the impact of artificial intelligence on the technology sector over the next 5 years. Based on this, perform a competitive analysis of NVIDIA (NVDA) and provide a strategic investment recommendation.
Analyze the healthcare sector in light of recent technological advancements and regulatory changes. Provide a market trend prediction for the next 3 years and a competitive analysis of UnitedHealth Group (UNH) within this context.
"""

from colorama import init, Fore, Style

# Initialize colorama for Windows compatibility
init()

if __name__ == "__main__":
    print("Financial Analysis Assistant. Type 'exit' or 'quit' to end the conversation.")
    print(example_queries)
    while True:
        user_input = input(Fore.RED + "\nYou: " + Style.RESET_ALL)
        if user_input.lower() in ["exit", "quit"]:
            print(Fore.GREEN + "Assistant: Goodbye!" + Style.RESET_ALL)
            break
        if user_input.strip() == "":
            continue
        response = agent.chat(user_input)
        print(Fore.GREEN + f"Assistant: {response}" + Style.RESET_ALL)