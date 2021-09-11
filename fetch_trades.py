#!/usr/bin/env python3

from brownie import Contract, web3
import json


START_BLOCK = 11_500_000
D_BLOCKS = 1000
END_BLOCK = web3.eth.blockNumber

timestamps = {}


def get_timestamp(b):
    if b in timestamps:
        return timestamps[b]
    else:
        t = web3.eth.getBlock(b).timestamp
        timestamps[b] = t
        return t


def fetch_sushi(addr):
    b_contract = Contract.from_explorer(addr)
    w_contract = web3.eth.contract(b_contract.address, abi=b_contract.abi)
    token0 = Contract(b_contract.token0())
    token1 = Contract(b_contract.token1())
    s0 = token0.symbol()
    s1 = token1.symbol()
    print('Fetchning information for %s-%s...' % (s0, s1))
    events = []
    for b in range(START_BLOCK, (END_BLOCK // D_BLOCKS + 1) * D_BLOCKS, D_BLOCKS):
        print('  fetching %s... for %s-%s' % (b, s0, s1))
        events += [{'t': get_timestamp(ev.blockNumber),
                    'block': ev.blockNumber,
                    'in': (ev.args.amount0In, ev.args.amount1In),
                    'out': (ev.args.amount0Out, ev.args.amount1Out)}
                   for ev in w_contract.events.Swap.getLogs(fromBlock=b, toBlock=b + D_BLOCKS)]
    return events


def get_price_vol(data, digits0, digits1, reverse=False):
    digits0 = 18 - digits0
    digits1 = 18 - digits1
    result = []
    for d in data:
        r = {'t': d['t'], 'block': d['block']}
        try:
            if d['in'][0] > 0:
                r['p'] = d['out'][1] * 10**digits1 / (d['in'][0] * 10**digits0)
            else:
                r['p'] = d['in'][1] * 10**digits1 / (d['out'][0] * 10**digits0)
        except ZeroDivisionError:
            continue
        if r['p'] == 0:
            continue
        if reverse:
            r['p'] = 1 / r['p']
            r['volume'] = max(d['in'][0], d['out'][0]) * 10**digits0 / 1e18
        else:
            r['volume'] = max(d['in'][1], d['out'][1]) * 10**digits1 / 1e18
        result.append(r)
    return result


def get_balances(addr, digits0, digits1, reverse = False):
    digits0 = 18 - digits0
    digits1 = 18 - digits1
    b_contract = Contract.from_explorer(addr)
    token0 = Contract(b_contract.token0())
    token1 = Contract(b_contract.token1())
    s0 = token0.symbol()
    s1 = token1.symbol()
    result = {}
    for b, t in timestamps.items():
        if b not in result:
            print('  fetching balances %s... for %s-%s' % (b, s0, s1))
            b0 = Contract(b_contract.token0()).balanceOf(b_contract) * 10**digits0 / 1e18
            b1 = Contract(b_contract.token1()).balanceOf(b_contract) * 10**digits1 / 1e18
            if reverse:
                result[b] = {'t': t, 'balance0': b1, 'balance1': b0}
            else:
                result[b] = {'t': t, 'balance0': b0, 'balance1': b1}
    return result


def main():
    eth_usdt = get_price_vol(fetch_sushi("0x06da0fd433C1A5d7a4faa01111c044910A184553"), 18, 6)
    wbtc_eth = get_price_vol(fetch_sushi("0xceff51756c56ceffca006cd410b03ffc46dd3a58"), 8, 18, reverse=True)
    wbtc_usdc = get_price_vol(fetch_sushi("0x004375dff511095cc5a197a54140a24efef3a416"), 8, 6)
    eth_usdt_b = get_balances("0x06da0fd433C1A5d7a4faa01111c044910A184553", 18, 6)
    wbtc_eth_b = get_balances("0xceff51756c56ceffca006cd410b03ffc46dd3a58", 8, 18, reverse=True)
    wbtc_usdc_b = get_balances("0x004375dff511095cc5a197a54140a24efef3a416", 8, 6)

    data = {'pricevol': {'eth-usdt': eth_usdt, 'eth-wbtc': wbtc_eth, 'wbtc-usdc': wbtc_usdc},
            'balances': {'eth-usdt': eth_usdt_b, 'eth-wbtc': wbtc_eth_b, 'wbtc-usdc': wbtc_usdc_b}}

    with open('trades.json', 'w') as f:
        json.dump(data, f)
