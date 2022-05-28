# -*- coding: utf-8 -*-
"""API client for obtaining liquidity data."""

import requests
import pandas as pd
import numpy as np
from tqdm import tqdm


def api_response(token_addresses):
    """Get token metadata from dexscreener API.

    Args:
        token_addresses (list): List of token addresses.

    Returns:
        dict: dictionary containing data.

    """
    d = {}
    for i in range(len(token_addresses)):
        r = requests.get("https://api.dexscreener.com/latest/dex/tokens/{}".format(token_addresses[i]))
        try:
            d[i] = r.json()
            print("Status: {}".format(r.status_code))
        except ValueError:
            print("Token address returned no data: {}".format(token_addresses[i]))
            continue
    return d


def get_liquidity(data, token_address):
    """Get liquidity data from dictionary containing data.

    Args:
        data (dict): Dictionary containing data.
        token_address (str): Token contract address.

    Returns:
        DataFrame: DataFrame of liquidity data for token address.

    """
    dex = {}
    liq = {}
    for i in range(0, len(data[token_address]["pairs"])):
        try:
            liq[i] = data[token_address]["pairs"][i]["liquidity"]["usd"]
            dex[i] = data[token_address]["pairs"][i]["dexId"]
        except KeyError:
            continue
    df = pd.DataFrame({
        "DexId": np.array(list(dex.values())),
        "Liquidity": np.array(list(liq.values())),
    }).groupby(["DexId"]).sum().sort_values("Liquidity", ascending=False)
    max_liquidity = df.iloc[0].squeeze()
    top_dex = df.index[0]
    return pd.DataFrame({
        "Top DEX": top_dex,
        "Liquidity": max_liquidity
    }, index=[0])


def compile_liquidity(token_addresses, data):
    """Combines liquidity data into single DataFrame.

    Args:
        token_addresses (list): List of token addresses.
        data (dict): Dictionary containing data.

    Returns:
        DataFrame: DataFrame of liquidity data for token address.

    """
    symbols = []
    liquidity = {}
    for token_address in tqdm(range(len(token_addresses))):
        try:
            symbols.append(data[token_address]["pairs"][0]["baseToken"]["symbol"])
        except IndexError:
            print("This address is invalid or has no liquidity: {}".format(token_addresses[token_address]))
            continue
        try:
            liquidity[data[token_address]["pairs"][0]["baseToken"]["symbol"]] = get_liquidity(data, token_address)
        except KeyError:
            continue
    df = pd.concat(liquidity).reset_index().rename(columns={"level_0": "Symbol"}).drop(["level_1"], axis=1).set_index(
        "Symbol").sort_values(by="Liquidity", ascending=False)
    df["Liquidity"] = df["Liquidity"].map("${0:,.0f}".format)
    return df
