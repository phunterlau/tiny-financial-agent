import os
import requests
from typing import Dict, Any
from pydantic import BaseModel

FINANCIAL_MODELING_PREP_API_KEY = os.environ.get("FINANCIAL_MODELING_PREP_API_KEY")

class FinancialData(BaseModel):
    symbol: str
    price: float
    volume: int
    priceAvg50: float
    priceAvg200: float
    EPS: float
    PE: float
    earningsAnnouncement: str

class CompanyFinancials(BaseModel):
    symbol: str
    companyName: str
    marketCap: int
    industry: str
    sector: str
    website: str
    beta: float
    price: float

class IncomeStatement(BaseModel):
    date: str
    revenue: int
    gross_profit: int
    net_income: int
    ebitda: int
    EPS: float
    EPS_diluted: float

def get_stock_price(symbol: str) -> FinancialData:
    """
    Fetches current stock price and related information for a given symbol.
    
    Args:
    symbol (str): The stock symbol to fetch data for.
    
    Returns:
    FinancialData: An object containing various financial metrics for the stock.
    """
    print(f"Fetching stock price data for {symbol}...")
    url = f"https://financialmodelingprep.com/api/v3/quote-order/{symbol}?apikey={FINANCIAL_MODELING_PREP_API_KEY}"
    response = requests.get(url)
    data = response.json()[0]
    return FinancialData(**data)

def get_company_financials(symbol: str) -> CompanyFinancials:
    """
    Fetches company financial information for a given symbol.
    
    Args:
    symbol (str): The stock symbol to fetch data for.
    
    Returns:
    CompanyFinancials: An object containing various financial metrics for the company.
    """
    print(f"Fetching company financial data for {symbol}...")
    url = f"https://financialmodelingprep.com/api/v3/profile/{symbol}?apikey={FINANCIAL_MODELING_PREP_API_KEY}"
    response = requests.get(url)
    data = response.json()[0]
    return CompanyFinancials(**data)

def get_income_statement(symbol: str) -> IncomeStatement:
    """
    Fetches the income statement for a given symbol.
    
    Args:
    symbol (str): The stock symbol to fetch data for.
    
    Returns:
    IncomeStatement: An object containing various income statement metrics for the company.
    """
    print(f"Fetching income statement data for {symbol}...")
    url = f"https://financialmodelingprep.com/api/v3/income-statement/{symbol}?period=annual&apikey={FINANCIAL_MODELING_PREP_API_KEY}"
    response = requests.get(url)
    data = response.json()[0]
    return IncomeStatement(**data)

def compare_stocks(symbol1: str, symbol2: str) -> Dict[str, Any]:
    """
    Compares two stocks based on their financial metrics.
    
    Args:
    symbol1 (str): The first stock symbol to compare.
    symbol2 (str): The second stock symbol to compare.
    
    Returns:
    Dict[str, Any]: A dictionary containing comparison data for both stocks.
    """
    print(f"Comparing stocks {symbol1} and {symbol2}...")
    stock1_price = get_stock_price(symbol1)
    stock1_financials = get_company_financials(symbol1)
    stock1_income = get_income_statement(symbol1)

    stock2_price = get_stock_price(symbol2)
    stock2_financials = get_company_financials(symbol2)
    stock2_income = get_income_statement(symbol2)

    return {
        "stock1": {
            "symbol": symbol1,
            "price": stock1_price.price,
            "PE": stock1_price.PE,
            "EPS": stock1_price.EPS,
            "marketCap": stock1_financials.marketCap,
            "sector": stock1_financials.sector,
            "industry": stock1_financials.industry,
            "revenue": stock1_income.revenue,
            "net_income": stock1_income.net_income
        },
        "stock2": {
            "symbol": symbol2,
            "price": stock2_price.price,
            "PE": stock2_price.PE,
            "EPS": stock2_price.EPS,
            "marketCap": stock2_financials.marketCap,
            "sector": stock2_financials.sector,
            "industry": stock2_financials.industry,
            "revenue": stock2_income.revenue,
            "net_income": stock2_income.net_income
        }
    }