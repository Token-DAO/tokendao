import os
import sys
import pandas as pd

ROOT_DIR = os.path.abspath(os.curdir)
ROOT_DIR = ROOT_DIR.replace("\\", "/").replace("tests", "")
sys.path.insert(0, ROOT_DIR)

from tokendao import liquidity


# Ethereum Token Contract Addresses
ethereum_addresses = [
    "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",  # WETH
    "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",  # WBTC
]


def test_api_response():
    responses = liquidity.api_response(ethereum_addresses)
    assert type(responses) == dict, "Data type should be a dictionary."


def test_get_liquidity():
    responses = liquidity.api_response(ethereum_addresses)
    df = liquidity.get_liquidity(responses, 0)
    assert isinstance(df, pd.DataFrame), "Data type should be DataFrame."


def test_compile_liquidity():
    responses = liquidity.api_response(ethereum_addresses)
    df = liquidity.compile_liquidity(ethereum_addresses, responses)
    assert isinstance(df, pd.DataFrame), "Data type should be DataFrame."


if __name__ == "__main__":
    test_api_response()
    test_get_liquidity()
    test_compile_liquidity()
    print("Everything passed")
