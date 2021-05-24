from data_client import DataClient
pair_list = DataClient().get_binance_pairs(
    base_currencies=['USDT'], quote_currencies=['BTC'])
print(pair_list)
