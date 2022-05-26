import requests
import pandas as pd
import numpy as np


def api_response(token_addresses):
    responses = {}
    for i in range(0, len(token_addresses)):
        r = requests.get("https://api.dexscreener.com/latest/dex/tokens/{}".format(token_addresses[i]))
        try:
            responses[i] = r.json()
            print("Status: {}".format(r.status_code))
        except ValueError:
            print("Token address returned no data: {}".format(token_addresses[i]))
            continue
    return responses


def get_liquidity(responses, address_num):
    dex = {}
    liq = {}
    for i in range(0, len(responses[address_num]["pairs"])):
        try:
            liq[i] = responses[address_num]["pairs"][i]["liquidity"]["usd"]
            dex[i] = responses[address_num]["pairs"][i]["dexId"]
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


def compile_liquidity(token_addresses, responses):
    symbols = []
    liquidity = {}
    for address_num in range(0, len(token_addresses)):
        try:
            symbols.append(responses[address_num]["pairs"][0]["baseToken"]["symbol"])
        except IndexError:
            print("This address is invalid or has no liquidity: {}".format(token_addresses[address_num]))
            continue
        try:
            liquidity[responses[address_num]["pairs"][0]["baseToken"]["symbol"]] = get_liquidity(responses, address_num)
        except KeyError:
            continue
    df = pd.concat(liquidity).reset_index().rename(columns={"level_0": "Symbol"}).drop(["level_1"], axis=1).set_index(
        "Symbol").sort_values(by="Liquidity", ascending=False)
    df["Liquidity"] = df["Liquidity"].map("${0:,.0f}".format)
    return df
