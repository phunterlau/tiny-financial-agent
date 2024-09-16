from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type, Callable
import inspect
import json
import requests
from atomic_tools import get_stock_price, get_company_financials, get_income_statement, FINANCIAL_MODELING_PREP_API_KEY
import sys

import functools

class OrchestrationFunction(ABC):
    def __init__(self, client: Any):
        self.client = client

    @abstractmethod
    def gather_data(self, **kwargs) -> Dict[str, Any]:
        pass

    @abstractmethod
    def prepare_prompt(self, data: Dict[str, Any]) -> str:
        pass

    @property
    @abstractmethod
    def system_message(self) -> str:
        pass

    def print_tool_usage(self, tool_name: str, result: Any):
        print(f"Tool used: {tool_name}")
        print(f"Result: {result}")
        print("-" * 50)

    def tool_use_decorator(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            self.print_tool_usage(func.__name__, result)
            return result
        return wrapper

    def __getattribute__(self, name: str):
        attr = super().__getattribute__(name)
        if callable(attr) and name.startswith('get_') and not name.startswith('__'):
            return self.tool_use_decorator(attr)
        return attr

    def execute(self, **kwargs) -> str:
        print(f"Executing {self.__class__.__name__}...")
        data = self.gather_data(**kwargs)
        prompt = self.prepare_prompt(data)
        
        print(f"Generating analysis for {self.__class__.__name__}...")
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content

class SectorAnalysis(OrchestrationFunction):
    def gather_data(self, sector: str, top_n: int = 5) -> Dict[str, Any]:
        url = f"https://financialmodelingprep.com/api/v3/stock-screener?sector={sector}&limit={top_n}&apikey={FINANCIAL_MODELING_PREP_API_KEY}"
        response = requests.get(url)
        companies = response.json()

        sector_data = []
        for company in companies:
            symbol = company['symbol']
            financials = self.get_company_financials(symbol)
            income = self.get_income_statement(symbol)
            stock_price = self.get_stock_price(symbol)
            
            sector_data.append({
                "symbol": symbol,
                "name": financials.companyName,
                "market_cap": financials.marketCap,
                "revenue": income.revenue,
                "net_income": income.net_income,
                "pe_ratio": stock_price.PE
            })

        return {"sector": sector, "top_n": top_n, "companies": sector_data}

    def prepare_prompt(self, data: Dict[str, Any]) -> str:
        return f"""
        Analyze the following top {data['top_n']} companies in the {data['sector']} sector:

        {json.dumps(data['companies'], indent=2)}

        Please provide a comprehensive sector analysis, including:
        1. Overall sector performance and trends
        2. Comparison of key players (market cap, revenue, profitability)
        3. Sector-specific metrics and their importance
        4. Future outlook and potential challenges for the sector

        Be sure to highlight any standout companies or notable trends.
        """

    @property
    def system_message(self) -> str:
        return "You are a financial analyst expert at sector analysis."

class PortfolioRecommendation(OrchestrationFunction):
    def gather_data(self, risk_tolerance: str, investment_amount: float, sectors: List[str]) -> Dict[str, Any]:
        portfolio_data = []
        for sector in sectors:
            url = f"https://financialmodelingprep.com/api/v3/stock-screener?sector={sector}&limit=3&apikey={FINANCIAL_MODELING_PREP_API_KEY}"
            response = requests.get(url)
            companies = response.json()
            for company in companies:
                symbol = company['symbol']
                financials = self.get_company_financials(symbol)
                stock_price = self.get_stock_price(symbol)
                portfolio_data.append({
                    "symbol": symbol,
                    "name": financials.companyName,
                    "sector": sector,
                    "price": stock_price.price,
                    "pe_ratio": stock_price.PE,
                    "beta": financials.beta
                })

        return {
            "risk_tolerance": risk_tolerance,
            "investment_amount": investment_amount,
            "sectors": sectors,
            "companies": portfolio_data
        }

    def prepare_prompt(self, data: Dict[str, Any]) -> str:
        return f"""
        Create a portfolio recommendation based on the following parameters:

        Risk Tolerance: {data['risk_tolerance']}
        Investment Amount: ${data['investment_amount']}
        Sectors of Interest: {', '.join(data['sectors'])}

        Consider the following companies:
        {json.dumps(data['companies'], indent=2)}

        Please provide a comprehensive portfolio recommendation, including:
        1. Asset allocation strategy based on risk tolerance
        2. Specific stock recommendations with rationale
        3. Diversification approach across sectors
        4. Potential risks and mitigation strategies

        Ensure the recommendation aligns with the given risk tolerance and investment amount.
        """

    @property
    def system_message(self) -> str:
        return "You are a financial advisor expert at creating portfolio recommendations."

class FinancialHealthAssessment(OrchestrationFunction):
    def gather_data(self, symbol: str) -> Dict[str, Any]:
        financials = self.get_company_financials(symbol)
        income = self.get_income_statement(symbol)
        stock_price = self.get_stock_price(symbol)

        url = f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{symbol}?limit=1&apikey={FINANCIAL_MODELING_PREP_API_KEY}"
        response = requests.get(url)
        balance_sheet = response.json()[0]

        url = f"https://financialmodelingprep.com/api/v3/cash-flow-statement/{symbol}?limit=1&apikey={FINANCIAL_MODELING_PREP_API_KEY}"
        response = requests.get(url)
        cash_flow = response.json()[0]

        current_ratio = balance_sheet['totalCurrentAssets'] / balance_sheet['totalCurrentLiabilities']
        debt_to_equity = balance_sheet['totalLiabilities'] / balance_sheet['totalStockholdersEquity']
        roe = income.net_income / balance_sheet['totalStockholdersEquity']
        free_cash_flow = cash_flow['operatingCashFlow'] - cash_flow['capitalExpenditure']

        return {
            "symbol": symbol,
            "company_name": financials.companyName,
            "stock_price": stock_price.price,
            "pe_ratio": stock_price.PE,
            "market_cap": financials.marketCap,
            "beta": financials.beta,
            "revenue": income.revenue,
            "net_income": income.net_income,
            "ebitda": income.ebitda,
            "current_ratio": current_ratio,
            "debt_to_equity": debt_to_equity,
            "roe": roe,
            "free_cash_flow": free_cash_flow
        }

    def prepare_prompt(self, data: Dict[str, Any]) -> str:
        return f"""
        Conduct a financial health assessment for {data['company_name']} ({data['symbol']}) based on the following data:

        Stock Price: ${data['stock_price']}
        P/E Ratio: {data['pe_ratio']}
        Market Cap: ${data['market_cap']}
        Beta: {data['beta']}

        Income Statement:
        Revenue: ${data['revenue']}
        Net Income: ${data['net_income']}
        EBITDA: ${data['ebitda']}

        Key Ratios:
        Current Ratio: {data['current_ratio']:.2f}
        Debt-to-Equity Ratio: {data['debt_to_equity']:.2f}
        Return on Equity: {data['roe']:.2f}
        Free Cash Flow: ${data['free_cash_flow']}

        Please provide a comprehensive financial health assessment, including:
        1. Profitability analysis
        2. Liquidity and solvency evaluation
        3. Efficiency and performance metrics
        4. Cash flow analysis
        5. Overall financial strength and potential red flags

        Consider industry standards and provide context for the financial metrics.
        """

    @property
    def system_message(self) -> str:
        return "You are a financial analyst expert at assessing company financial health."

class StrategicInvestmentAnalysis(OrchestrationFunction):
    def gather_data(self, symbol: str, time_horizon: str) -> Dict[str, Any]:
        financials = self.get_company_financials(symbol)
        income = self.get_income_statement(symbol)
        stock_price = self.get_stock_price(symbol)

        # Get historical price data
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?apikey={FINANCIAL_MODELING_PREP_API_KEY}"
        response = requests.get(url)
        historical_data = response.json()['historical'][:365]  # Last year's data

        # Get industry peers
        url = f"https://financialmodelingprep.com/api/v3/stock-screener?industry={financials.industry}&limit=5&apikey={FINANCIAL_MODELING_PREP_API_KEY}"
        response = requests.get(url)
        peers = [company['symbol'] for company in response.json() if company['symbol'] != symbol]

        return {
            "symbol": symbol,
            "company_name": financials.companyName,
            "industry": financials.industry,
            "current_price": stock_price.price,
            "pe_ratio": stock_price.PE,
            "market_cap": financials.marketCap,
            "revenue": income.revenue,
            "net_income": income.net_income,
            "historical_data": historical_data,
            "peers": peers,
            "time_horizon": time_horizon
        }

    def prepare_prompt(self, data: Dict[str, Any]) -> str:
        return f"""
        Conduct a strategic investment analysis for {data['company_name']} ({data['symbol']}) with a {data['time_horizon']} time horizon. Follow these steps:

        1. Company Overview:
           Analyze the company's current financial position, including market cap, P/E ratio, revenue, and net income.

        2. Historical Performance:
           Review the provided historical price data for the last year. Identify any significant trends or patterns.

        3. Industry Analysis:
           Consider the company's position within the {data['industry']} industry. Compare its performance to industry peers: {', '.join(data['peers'])}.

        4. SWOT Analysis:
           Based on the available data, perform a SWOT (Strengths, Weaknesses, Opportunities, Threats) analysis for the company.

        5. Future Outlook:
           Taking into account the {data['time_horizon']} time horizon, project potential scenarios for the company's future performance. Consider industry trends, economic factors, and company-specific elements.

        6. Risk Assessment:
           Identify and analyze potential risks associated with investing in this company over the specified time horizon.

        7. Investment Recommendation:
           Based on all the above analysis, provide a detailed investment recommendation. Include potential upsides, downsides, and any specific conditions or triggers that investors should watch for.

        Ensure that each step builds upon the previous ones, creating a coherent and comprehensive analysis. Provide specific data points and reasoning for each conclusion drawn.
        """

    @property
    def system_message(self) -> str:
        return "You are a senior investment strategist with expertise in long-term financial analysis and investment recommendations."

class MarketTrendPrediction(OrchestrationFunction):
    def gather_data(self, sector: str, timeframe: str) -> Dict[str, Any]:
        # Get top companies in the sector
        url = f"https://financialmodelingprep.com/api/v3/stock-screener?sector={sector}&limit=10&apikey={FINANCIAL_MODELING_PREP_API_KEY}"
        response = requests.get(url)
        companies = response.json()

        sector_data = []
        for company in companies:
            symbol = company['symbol']
            financials = self.get_company_financials(symbol)
            stock_price = self.get_stock_price(symbol)
            sector_data.append({
                "symbol": symbol,
                "name": financials.companyName,
                "market_cap": financials.marketCap,
                "price": stock_price.price,
                "pe_ratio": stock_price.PE,
                "beta": financials.beta
            })

        # Get sector ETF data (assuming there's an ETF for the sector)
        etf_symbol = f"XL{sector[:1]}"  # This is a simplification; in reality, you'd need to map sectors to actual ETF symbols
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{etf_symbol}?apikey={FINANCIAL_MODELING_PREP_API_KEY}"
        response = requests.get(url)
        etf_data = response.json()['historical'][:365]  # Last year's data

        return {
            "sector": sector,
            "timeframe": timeframe,
            "companies": sector_data,
            "etf_data": etf_data
        }

    def prepare_prompt(self, data: Dict[str, Any]) -> str:
        return f"""
        Conduct a market trend prediction analysis for the {data['sector']} sector over the next {data['timeframe']}. Follow these steps:

        1. Sector Overview:
           Analyze the current state of the {data['sector']} sector based on the provided data for the top 10 companies.

        2. Historical Trend Analysis:
           Review the historical price data for the sector ETF. Identify any significant trends, cycles, or patterns.

        3. Company Comparison:
           Compare the performance and metrics of the top companies in the sector. Identify any standout performers or laggards.

        4. Macroeconomic Factors:
           Consider relevant macroeconomic factors that could impact the {data['sector']} sector over the {data['timeframe']} timeframe.

        5. Technological and Regulatory Trends:
           Analyze any significant technological advancements or regulatory changes that could affect the sector.

        6. Market Sentiment Analysis:
           Based on the available data, gauge the current market sentiment towards the {data['sector']} sector and how it might evolve.

        7. Scenario Planning:
           Develop multiple scenarios for how the sector might perform over the {data['timeframe']}, including best-case, worst-case, and most likely scenarios.

        8. Key Indicators:
           Identify key indicators or events that investors should monitor to track the accuracy of these predictions over time.

        9. Investment Implications:
           Discuss the implications of these predictions for investors interested in the {data['sector']} sector.

        Ensure that each step of the analysis builds upon the previous ones, creating a coherent and comprehensive market trend prediction. Provide specific data points and reasoning for each conclusion drawn.
        """

    @property
    def system_message(self) -> str:
        return "You are a market analyst specializing in sector trends and long-term market predictions."

class CompanyCompetitiveAnalysis(OrchestrationFunction):
    def gather_data(self, symbol: str) -> Dict[str, Any]:
        financials = self.get_company_financials(symbol)
        income = self.get_income_statement(symbol)

        # Get competitors
        url = f"https://financialmodelingprep.com/api/v3/stock-screener?industry={financials.industry}&limit=5&apikey={FINANCIAL_MODELING_PREP_API_KEY}"
        response = requests.get(url)
        competitors = [company['symbol'] for company in response.json() if company['symbol'] != symbol]

        company_data = {
            symbol: {
                "name": financials.companyName,
                "market_cap": financials.marketCap,
                "revenue": income.revenue,
                "net_income": income.net_income,
                "pe_ratio": get_stock_price(symbol).PE,
                "beta": financials.beta
            }
        }

        for competitor in competitors:
            comp_financials = get_company_financials(competitor)
            comp_income = get_income_statement(competitor)
            company_data[competitor] = {
                "name": comp_financials.companyName,
                "market_cap": comp_financials.marketCap,
                "revenue": comp_income.revenue,
                "net_income": comp_income.net_income,
                "pe_ratio": get_stock_price(competitor).PE,
                "beta": comp_financials.beta
            }

        return {
            "main_company": symbol,
            "industry": financials.industry,
            "companies": company_data
        }

    def prepare_prompt(self, data: Dict[str, Any]) -> str:
        return f"""
        Conduct a comprehensive competitive analysis for {data['companies'][data['main_company']]['name']} ({data['main_company']}) in the {data['industry']} industry. Follow these steps:

        1. Company Overview:
           Provide an overview of {data['companies'][data['main_company']]['name']}, including its market position, key financials, and business model.

        2. Competitor Identification:
           Analyze the provided competitors: {', '.join([data['companies'][c]['name'] for c in data['companies'] if c != data['main_company']])}.

        3. Comparative Financial Analysis:
           Compare key financial metrics (market cap, revenue, net income, P/E ratio, beta) across all companies. Identify strengths and weaknesses of each.

        4. Market Share Analysis:
           Estimate market share for each company based on the available financial data.

        5. Product/Service Comparison:
           Based on public knowledge of these companies, compare their main products or services, identifying unique selling points and areas of overlap.

        6. SWOT Analysis:
           Perform a SWOT (Strengths, Weaknesses, Opportunities, Threats) analysis for {data['companies'][data['main_company']]['name']}, taking into account its competitive landscape.

        7. Competitive Strategies:
           Analyze potential competitive strategies that {data['companies'][data['main_company']]['name']} could employ to improve its market position.

        8. Future Outlook:
           Project how the competitive landscape might evolve in the next 3-5 years, considering current trends and potential disruptors in the {data['industry']} industry.

        9. Key Success Factors:
           Identify the key factors that will determine success in this competitive landscape.

        10. Strategic Recommendations:
            Provide strategic recommendations for {data['companies'][data['main_company']]['name']} to enhance its competitive position.

        Ensure that each step of the analysis builds upon the previous ones, creating a coherent and comprehensive competitive analysis. Use specific data points and provide reasoning for each conclusion drawn.
        """

    @property
    def system_message(self) -> str:
        return "You are a strategic consultant specializing in competitive analysis and corporate strategy."
    
class CompanyComparativeAnalysis(OrchestrationFunction):
    def gather_data(self, symbol1: str, symbol2: str, time_horizon: str) -> Dict[str, Any]:
        company1_data = self._gather_company_data(symbol1)
        company2_data = self._gather_company_data(symbol2)
        
        return {
            "company1": company1_data,
            "company2": company2_data,
            "time_horizon": time_horizon
        }
    
    def _gather_company_data(self, symbol: str) -> Dict[str, Any]:
        financials = self.use_atomic_function('get_company_financials', symbol)
        income = self.use_atomic_function('get_income_statement', symbol)
        stock_price = self.use_atomic_function('get_stock_price', symbol)
        historical_data = self.use_atomic_function('get_historical_price_data', symbol)
        
        return {
            "symbol": symbol,
            "financials": financials,
            "income": income,
            "stock_price": stock_price,
            "historical_data": historical_data
        }

    def prepare_prompt(self, data: Dict[str, Any]) -> str:
        return f"""
        Perform a comparative analysis of {data['company1']['symbol']} and {data['company2']['symbol']} over a {data['time_horizon']} time horizon.
        Include a competitive analysis and assessment of investment potential for both companies.
        
        Company 1 ({data['company1']['symbol']}) Data:
        {json.dumps(data['company1'], indent=2)}
        
        Company 2 ({data['company2']['symbol']}) Data:
        {json.dumps(data['company2'], indent=2)}
        
        Provide a comprehensive analysis covering:
        1. Competitive position of both companies
        2. Financial performance comparison
        3. Growth prospects over the {data['time_horizon']} time horizon
        4. Potential risks and opportunities
        5. Overall investment potential comparison
        """

    @property
    def system_message(self) -> str:
        return "You are a financial analyst expert in comparative company analysis and investment potential assessment."

def get_all_orchestration_functions() -> Dict[str, Type[OrchestrationFunction]]:
    """
    Dynamically retrieves all orchestration function classes defined in this module.
    """
    return {
        name.lower(): cls
        for name, cls in inspect.getmembers(
            sys.modules[__name__],
            lambda member: inspect.isclass(member) and issubclass(member, OrchestrationFunction) and member != OrchestrationFunction
        )
    }
