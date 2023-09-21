const FMP_KEY = "4e832008a436bc85dfa8fc47c797d59f";
const FUGLE_KEY =
  "YTQ2NzliMDgtOGU4OC00NDQyLTk1YzUtMzQ0ZWJhYzA3NTliIDQzM2VhYWU1LWMzYTctNGE1Yy05ZjhhLWMyYjVmNmRlNjE2OQ==";

function cachedFetch(url, options) {
  let cache = CacheService.getScriptCache();
  if (!(text = cache.get(url))) {
    text = UrlFetchApp.fetch(url, options).getContentText();
    cache.put(url, text, 60);
  }
  return text;
}

function getPrice(symbol) {
  let text = cachedFetch(
    `https://financialmodelingprep.com/api/v3/quote-short/${symbol}?apikey=${FMP_KEY}`
  );
  return JSON.parse(text)[0].price;
}

function getCryptoPrice(symbol) {
  let url = `https://api1.binance.com/api/v3/ticker/price?symbol=${symbol}USDT`;
  let text = cachedFetch(
    `http://52.198.155.160:8080/content?url=${encodeURIComponent(url)}`
  );
  return parseFloat(JSON.parse(text).price);
}

function getTwPrice(symbol) {
  let text = cachedFetch(
    `https://api.fugle.tw/marketdata/v1.0/stock/intraday/quote/${symbol}`,
    { headers: { "X-API-KEY": FUGLE_KEY } }
  );
  return JSON.parse(text).lastTrade.price;
}

function getScore(market, symbol, scale) {
  let text = cachedFetch(
    `http://52.198.155.160:8080/score?market=${market}&symbol=${symbol}&scale=${scale}`
  );
  return parseFloat(text);
}

function getSignals(market, symbol) {
  return cachedFetch(
    `http://52.198.155.160:8080/signals?market=${market}&symbol=${symbol}`
  );
}
