BASELINE_START = 2000
BASELINE_END = 2020
FUTURE_START = 2020
FUTURE_END = 2040

DEFAULT_SCENARIO = "ssp245"
VALID_SCENARIOS = ("ssp126", "ssp245", "ssp585")

CACHE_DB = "baselines.db"
# Baselines are 20-year averages; refresh monthly is plenty
CACHE_TTL_DAYS = 30
