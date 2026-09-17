"""China ADR sleeve universe (US-listed Alpaca symbols). Edit here; ``deepvibe_hedge.config`` imports this tuple.

Theme    : long-term-bullish China, expressed via US-listed Chinese ADRs.
Proxy ETF: ``PGJ`` (Invesco Golden Dragon China) drives the sleeve's 200D trend
           gate + MRAT — it tracks the NASDAQ Golden Dragon index of US-listed
           Chinese ADRs, the closest ETF match to this basket. ``PGJ`` is the
           slot's signal ETF only; it is intentionally NOT a basket member
           (this sleeve holds individual ADRs, not the ETF). Swap to ``KWEB``
           (China internet) or ``FXI`` (large-cap H-shares) in ``MAD_INDEX_SLOTS``
           if you prefer a different sleeve-selection signal.

Conventions (same as ``sp500.py`` / ``space_sector.py``):
  * Only US-exchange (NYSE/Nasdaq) symbols — names delisted to OTC (e.g. DIDI →
    DIDIY, Luckin → LKNCY, China Mobile/Telecom/Unicom, PetroChina) are omitted.
  * The runtime data-completeness filter (``MAD_MIN_DATA_COMPLETENESS`` over the
    MRAT long window) and Alpaca availability prune any name lacking history, so
    a freshly-IPO'd or thinly-traded ADR simply drops out until it seasons.
  * Run the OHLCV fetcher after enabling this slot so the new symbols download.

Ticker count: ~50 US-listed Chinese ADRs.
"""

china_adr = (
    # --- Internet / e-commerce / media ---
    "BABA",   # Alibaba
    "PDD",    # PDD Holdings (Pinduoduo / Temu)
    "JD",     # JD.com
    "BIDU",   # Baidu
    "NTES",   # NetEase
    "TCOM",   # Trip.com
    "BILI",   # Bilibili
    "TME",    # Tencent Music
    "IQ",     # iQIYI
    "WB",     # Weibo
    "VIPS",   # Vipshop
    "MOMO",   # Hello Group (Momo)
    "YY",     # JOYY
    "BZUN",   # Baozun
    "KC",     # Kingsoft Cloud
    "DAO",    # Youdao
    "GOTU",   # Gaotu Techedu
    "DOYU",   # DouYu
    "HUYA",   # Huya
    "ZH",     # Zhihu
    "API",    # Agora
    "TUYA",   # Tuya
    "MNSO",   # Miniso
    "RLX",    # RLX Technology
    # --- EV / mobility ---
    "NIO",    # NIO
    "LI",     # Li Auto
    "XPEV",   # XPeng
    "ZK",     # Zeekr
    # --- Fintech / financials ---
    "FUTU",   # Futu Holdings
    "TIGR",   # UP Fintech (Tiger Brokers)
    "QFIN",   # Qifu Technology (360 DigiTech)
    "FINV",   # FinVolution
    "LU",     # Lufax
    "LX",     # LexinFintech
    # --- Consumer / services / logistics / real estate ---
    "YUMC",   # Yum China
    "ATHM",   # Autohome
    "HTHT",   # H World (Huazhu hotels)
    "ZTO",    # ZTO Express
    "BEKE",   # KE Holdings (Beike)
    "ATAT",   # Atour Lifestyle
    "NOAH",   # Noah Holdings
    "EH",     # EHang
    # --- Cloud / data-center infrastructure ---
    "GDS",    # GDS Holdings
    "VNET",   # VNET Group (21Vianet)
    # --- Education ---
    "TAL",    # TAL Education
    "EDU",    # New Oriental Education
    # --- Clean energy / solar ---
    "DQ",     # Daqo New Energy
    "JKS",    # JinkoSolar
    "CSIQ",   # Canadian Solar (China ops)
    "SOL",    # Emeren (ReneSola)
    # --- Biotech / pharma ---
    "ZLAB",   # Zai Lab
    "LEGN",   # Legend Biotech
)
