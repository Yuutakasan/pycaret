# External Factors Analysis Module

## Overview

The `external_factors.py` module provides comprehensive integration of external data sources to enhance sales forecasting accuracy for Japanese convenience stores. It integrates weather data, calendar events, economic indicators, and provides advanced correlation and impact quantification analysis.

## Features

### 1. Weather Data Integration

**Supported APIs:**
- **OpenWeatherMap** - Global coverage including Japan (requires API key)
- **Open-Meteo JMA API** - Free Japan-specific weather data from JMA models
- **JMA Weather API** - Unofficial scraper for JMA data (requires additional setup)

**Weather Features:**
- Temperature (°C)
- Precipitation (mm)
- Humidity (%)
- Wind speed (m/s)
- Weather condition (clear, cloudy, rain, snow, etc.)
- Atmospheric pressure (hPa)
- Visibility (km)

**Example:**
```python
from src.analysis.external_factors import WeatherAPIClient

# Initialize client (Open-Meteo is free)
weather_client = WeatherAPIClient(provider="open-meteo")

# Get current weather for Tokyo
tokyo_lat, tokyo_lon = 35.6762, 139.6503
current = weather_client.get_current_weather(tokyo_lat, tokyo_lon)

print(f"Temperature: {current.temperature}°C")
print(f"Condition: {current.condition}")

# Get historical weather
from datetime import datetime
start = datetime(2024, 1, 1)
end = datetime(2024, 1, 31)
historical = weather_client.get_historical_weather(tokyo_lat, tokyo_lon, start, end)
```

### 2. Calendar Events Integration

**Supported APIs:**
- **Nager.Date** - Free public holidays API (recommended)
- **Calendarific** - Commercial API with free tier
- **Custom Events** - Add promotions, local events, etc.

**Event Types:**
- National holidays
- Promotional events
- Local events
- Sports events
- Custom business events

**Impact Levels:**
- Low: Minor impact on sales
- Medium: Moderate impact
- High: Significant impact (major holidays)
- Critical: Extreme impact (Golden Week, New Year)

**Example:**
```python
from src.analysis.external_factors import CalendarAPIClient, CalendarEvent

# Initialize client (Nager.Date is free)
calendar_client = CalendarAPIClient(provider="nager")

# Get 2025 holidays
holidays = calendar_client.get_holidays(2025)

# Add custom promotional event
from datetime import datetime
custom_events = []
promo_event = CalendarEvent(
    date=datetime(2025, 3, 15),
    event_type='promotion',
    name='Spring Sale',
    impact_level='high'
)
custom_events.append(promo_event)
```

### 3. Economic Indicators

**Supported APIs:**
- **World Bank Data API** - Free access to global economic data
- **Trading Economics API** - Real-time economic indicators (requires subscription)
- **FRED API** - Federal Reserve economic data

**Available Indicators:**
- GDP growth rate (%)
- Inflation rate (%)
- Unemployment rate (%)
- Consumer confidence index
- Retail sales index

**Example:**
```python
from src.analysis.external_factors import EconomicDataClient

# Initialize client (World Bank is free)
econ_client = EconomicDataClient(provider="worldbank")

# Get economic indicators
from datetime import datetime
indicators = econ_client.get_indicators(
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2025, 1, 1)
)

for ind in indicators:
    print(f"{ind.date.year}: GDP={ind.gdp_growth}%, Inflation={ind.inflation_rate}%")
```

### 4. Complete Analysis Pipeline

**ExternalFactorsAnalyzer** provides a unified interface for:
- Data integration from multiple sources
- Correlation analysis
- Impact quantification
- Feature engineering

**Example:**
```python
from src.analysis.external_factors import (
    ExternalFactorsAnalyzer,
    WeatherAPIClient,
    CalendarAPIClient,
    EconomicDataClient
)
import pandas as pd

# Initialize clients
weather_client = WeatherAPIClient(provider="open-meteo")
calendar_client = CalendarAPIClient(provider="nager")
econ_client = EconomicDataClient(provider="worldbank")

# Create analyzer
analyzer = ExternalFactorsAnalyzer(
    weather_client=weather_client,
    calendar_client=calendar_client,
    economic_client=econ_client
)

# Load your sales data
sales_df = pd.read_csv('sales_data.csv')

# Store location (Tokyo example)
latitude, longitude = 35.6762, 139.6503

# Integrate weather data
sales_with_weather = analyzer.integrate_weather_data(
    sales_df,
    latitude=latitude,
    longitude=longitude,
    date_column='date'
)

# Integrate calendar events
sales_enriched = analyzer.integrate_calendar_events(
    sales_with_weather,
    date_column='date'
)

# Integrate economic indicators
sales_complete = analyzer.integrate_economic_indicators(
    sales_enriched,
    date_column='date'
)

# Analyze correlations
correlations = analyzer.calculate_correlations(
    sales_complete,
    target_column='sales'
)

print("Top correlated features:")
print(correlations.head(10))

# Quantify impact
impact = analyzer.quantify_impact(
    sales_complete,
    target_column='sales',
    method='regression'
)

print("\nFeature importance:")
for feature, importance in list(impact.items())[:10]:
    print(f"  {feature}: {importance:.4f}")

# Create comprehensive summary
summary = analyzer.create_feature_summary(sales_complete, target_column='sales')
summary.to_csv('external_factors_summary.csv', index=False)
```

## Correlation Analysis

The module provides two correlation methods:

1. **Pearson Correlation**: Measures linear relationships
   - Best for: Continuous variables with linear relationships
   - Range: -1 to +1

2. **Spearman Correlation**: Measures monotonic relationships
   - Best for: Non-linear but monotonic relationships
   - Range: -1 to +1

**Statistical Significance:**
- p-value < 0.05: Statistically significant correlation
- p-value ≥ 0.05: Not statistically significant

## Impact Quantification

Two methods are available for quantifying the impact of external factors:

### 1. Random Forest Regression
- Uses ensemble learning to determine feature importance
- Handles non-linear relationships
- Robust to outliers
- Recommended for complex datasets

### 2. Elasticity Analysis
- Measures percentage change in sales per percentage change in factor
- Useful for economic interpretation
- Best for business stakeholders

## API Configuration

### Required API Keys

1. **OpenWeatherMap** (if using this provider):
   - Sign up: https://openweathermap.org/api
   - Free tier: 1,000 calls/day
   - Pricing: https://openweathermap.org/price

2. **Calendarific** (if using this provider):
   - Sign up: https://calendarific.com
   - Free tier: 1,000 calls/month
   - Recommended: Use Nager.Date instead (no key required)

### Free Alternatives (No API Key Required)

- **Weather**: Open-Meteo JMA API
- **Holidays**: Nager.Date API
- **Economic**: World Bank Data API

## Best Practices

### 1. API Rate Limiting
```python
import time

# Add delays between requests
weather_data = []
for location in locations:
    data = weather_client.get_current_weather(lat, lon)
    weather_data.append(data)
    time.sleep(1)  # 1 second delay
```

### 2. Caching
```python
# Weather client has built-in caching
weather_client = WeatherAPIClient(
    provider="open-meteo",
    cache_ttl=3600  # Cache for 1 hour
)
```

### 3. Error Handling
```python
try:
    weather_data = weather_client.get_historical_weather(lat, lon, start, end)
except requests.exceptions.RequestException as e:
    logger.error(f"Weather API failed: {e}")
    # Use fallback or cached data
    weather_data = load_cached_weather_data()
```

### 4. Data Quality Checks
```python
# Check for missing values
print(f"Missing weather data: {df['temperature'].isna().sum()}")

# Check for outliers
from scipy import stats
z_scores = np.abs(stats.zscore(df['temperature'].dropna()))
outliers = df[z_scores > 3]
print(f"Temperature outliers: {len(outliers)}")
```

## Feature Engineering Examples

### Weather-Based Features
```python
# Temperature categories
df['temp_category'] = pd.cut(
    df['temperature'],
    bins=[-np.inf, 10, 20, 30, np.inf],
    labels=['cold', 'cool', 'warm', 'hot']
)

# Rainy day flag
df['is_rainy'] = (df['precipitation'] > 1.0).astype(int)

# Extreme weather flag
df['extreme_weather'] = (
    (df['temperature'] < 5) |
    (df['temperature'] > 35) |
    (df['precipitation'] > 50)
).astype(int)
```

### Calendar-Based Features
```python
# Weekend flag
df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)

# Month-end flag
df['is_month_end'] = (df['date'].dt.day >= 25).astype(int)

# Payday flag (typically 25th of month)
df['is_payday'] = (df['date'].dt.day == 25).astype(int)

# Holiday week
df['in_holiday_week'] = (df['days_to_holiday'].abs() <= 3).astype(int)
```

### Economic-Based Features
```python
# Economic trend
df['gdp_trend'] = df['gdp_growth'].rolling(window=4).mean()

# Inflation impact
df['real_purchasing_power'] = 100 - df['inflation_rate']

# Economic sentiment
df['economic_sentiment'] = (
    df['gdp_growth'].rank(pct=True) * 0.5 +
    (100 - df['unemployment_rate']).rank(pct=True) * 0.5
)
```

## Performance Optimization

### Batch Processing
```python
# Process multiple locations in parallel
from concurrent.futures import ThreadPoolExecutor

def fetch_weather(location):
    lat, lon = location
    return weather_client.get_current_weather(lat, lon)

with ThreadPoolExecutor(max_workers=5) as executor:
    weather_results = list(executor.map(fetch_weather, locations))
```

### Memory Efficiency
```python
# Use chunking for large datasets
chunk_size = 10000
for chunk in pd.read_csv('large_sales_data.csv', chunksize=chunk_size):
    enriched_chunk = analyzer.integrate_weather_data(chunk, lat, lon)
    # Process chunk...
```

## Troubleshooting

### Common Issues

1. **API Timeout**
   - Solution: Increase timeout parameter
   ```python
   weather_client = WeatherAPIClient(provider="open-meteo")
   # Modify requests timeout in source code or use retry logic
   ```

2. **Missing Data**
   - Solution: Use forward/backward fill
   ```python
   df['temperature'].fillna(method='ffill', inplace=True)
   ```

3. **Date Parsing Errors**
   - Solution: Explicit date format
   ```python
   df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
   ```

## Advanced Topics

### Custom Weather Conditions
```python
# Define custom weather severity
def weather_severity(row):
    score = 0
    if row['precipitation'] > 20: score += 3
    if row['temperature'] < 5 or row['temperature'] > 35: score += 2
    if row['wind_speed'] > 10: score += 1
    return score

df['weather_severity'] = df.apply(weather_severity, axis=1)
```

### Multi-Location Analysis
```python
# Analyze multiple store locations
stores = [
    {'name': 'Tokyo Store', 'lat': 35.6762, 'lon': 139.6503},
    {'name': 'Osaka Store', 'lat': 34.6937, 'lon': 135.5023},
    {'name': 'Nagoya Store', 'lat': 35.1815, 'lon': 136.9066}
]

for store in stores:
    print(f"Analyzing {store['name']}...")
    enriched_data = analyzer.integrate_weather_data(
        sales_df,
        latitude=store['lat'],
        longitude=store['lon']
    )
    # Analyze each store...
```

### Composite External Factors Index
```python
# Create weighted composite index
def create_external_index(row):
    weather_score = (
        (row['temperature'] - 20) / 10 * 0.3 +  # Optimal temp = 20°C
        -row['precipitation'] / 50 * 0.2 +       # Less rain = better
        (80 - row['humidity']) / 20 * 0.1        # Lower humidity = better
    )

    calendar_score = (
        row['is_holiday'] * 0.3 +
        row['is_promotion'] * 0.2 +
        (1 if abs(row['days_to_holiday']) <= 3 else 0) * 0.1
    )

    return weather_score + calendar_score

df['external_factor_index'] = df.apply(create_external_index, axis=1)
```

## References

### Weather APIs
- OpenWeatherMap: https://openweathermap.org/api
- Open-Meteo JMA: https://open-meteo.com/en/docs/jma-api
- JMA Official: https://www.jma.go.jp/jma/indexe.html

### Calendar APIs
- Nager.Date: https://date.nager.at
- Calendarific: https://calendarific.com

### Economic Data
- World Bank: https://data.worldbank.org
- Trading Economics: https://tradingeconomics.com/japan/indicators
- FRED: https://fred.stlouisfed.org

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

## Support

For issues and questions:
- GitHub Issues: https://github.com/pycaret/pycaret/issues
- Documentation: https://pycaret.gitbook.io/
