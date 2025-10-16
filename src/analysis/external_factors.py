"""
External Factors Analysis Module for Japanese Convenience Store Forecasting

This module integrates external data sources to enhance sales forecasting accuracy:
1. Weather data (temperature, precipitation, conditions)
2. Calendar events (holidays, promotions, local events)
3. Competitor activity tracking
4. Foot traffic patterns
5. Economic indicators
6. Correlation analysis with sales
7. Impact quantification

Author: PyCaret Development Team
License: MIT
"""

import warnings
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
import requests
import json
from functools import lru_cache
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class WeatherData:
    """Weather observation data structure"""
    timestamp: datetime
    temperature: float  # Celsius
    precipitation: float  # mm
    humidity: float  # percentage
    wind_speed: float  # m/s
    condition: str  # clear, cloudy, rain, snow, etc.
    pressure: float  # hPa
    visibility: float  # km

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp,
            'temperature': self.temperature,
            'precipitation': self.precipitation,
            'humidity': self.humidity,
            'wind_speed': self.wind_speed,
            'condition': self.condition,
            'pressure': self.pressure,
            'visibility': self.visibility
        }


@dataclass
class CalendarEvent:
    """Calendar event data structure"""
    date: datetime
    event_type: str  # holiday, promotion, local_event, sports_event
    name: str
    description: str = ""
    impact_level: str = "medium"  # low, medium, high, critical
    is_national: bool = False
    is_recurring: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'date': self.date,
            'event_type': self.event_type,
            'name': self.name,
            'description': self.description,
            'impact_level': self.impact_level,
            'is_national': self.is_national,
            'is_recurring': self.is_recurring
        }


@dataclass
class EconomicIndicator:
    """Economic indicator data structure"""
    date: datetime
    gdp_growth: Optional[float] = None
    inflation_rate: Optional[float] = None
    unemployment_rate: Optional[float] = None
    consumer_confidence: Optional[float] = None
    retail_sales_index: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'date': self.date,
            'gdp_growth': self.gdp_growth,
            'inflation_rate': self.inflation_rate,
            'unemployment_rate': self.unemployment_rate,
            'consumer_confidence': self.consumer_confidence,
            'retail_sales_index': self.retail_sales_index
        }


class WeatherAPIClient:
    """
    Weather API client supporting multiple providers

    Supported APIs:
    - OpenWeatherMap (global, Japan supported)
    - Open-Meteo JMA API (Japan-specific)
    - JMA Weather API (unofficial scraper)
    """

    def __init__(
        self,
        provider: str = "openweathermap",
        api_key: Optional[str] = None,
        cache_ttl: int = 3600
    ):
        """
        Initialize weather API client

        Parameters:
        -----------
        provider : str
            Weather API provider ('openweathermap', 'open-meteo', 'jma')
        api_key : str, optional
            API key for providers that require authentication
        cache_ttl : int
            Cache time-to-live in seconds (default: 1 hour)
        """
        self.provider = provider.lower()
        self.api_key = api_key
        self.cache_ttl = cache_ttl
        self._validate_config()

    def _validate_config(self):
        """Validate API configuration"""
        if self.provider == "openweathermap" and not self.api_key:
            raise ValueError("OpenWeatherMap requires an API key")

    @lru_cache(maxsize=128)
    def get_current_weather(
        self,
        latitude: float,
        longitude: float
    ) -> WeatherData:
        """
        Get current weather for location

        Parameters:
        -----------
        latitude : float
            Location latitude
        longitude : float
            Location longitude

        Returns:
        --------
        WeatherData
            Current weather observation
        """
        if self.provider == "openweathermap":
            return self._get_openweathermap_current(latitude, longitude)
        elif self.provider == "open-meteo":
            return self._get_open_meteo_current(latitude, longitude)
        elif self.provider == "jma":
            return self._get_jma_current(latitude, longitude)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def get_historical_weather(
        self,
        latitude: float,
        longitude: float,
        start_date: datetime,
        end_date: datetime
    ) -> List[WeatherData]:
        """
        Get historical weather data

        Parameters:
        -----------
        latitude : float
            Location latitude
        longitude : float
            Location longitude
        start_date : datetime
            Start date for historical data
        end_date : datetime
            End date for historical data

        Returns:
        --------
        List[WeatherData]
            Historical weather observations
        """
        if self.provider == "openweathermap":
            return self._get_openweathermap_historical(
                latitude, longitude, start_date, end_date
            )
        elif self.provider == "open-meteo":
            return self._get_open_meteo_historical(
                latitude, longitude, start_date, end_date
            )
        else:
            raise NotImplementedError(
                f"Historical data not available for {self.provider}"
            )

    def _get_openweathermap_current(
        self,
        latitude: float,
        longitude: float
    ) -> WeatherData:
        """Get current weather from OpenWeatherMap"""
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            'lat': latitude,
            'lon': longitude,
            'appid': self.api_key,
            'units': 'metric'
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            return WeatherData(
                timestamp=datetime.fromtimestamp(data['dt']),
                temperature=data['main']['temp'],
                precipitation=data.get('rain', {}).get('1h', 0.0),
                humidity=data['main']['humidity'],
                wind_speed=data['wind']['speed'],
                condition=data['weather'][0]['main'].lower(),
                pressure=data['main']['pressure'],
                visibility=data.get('visibility', 10000) / 1000  # Convert to km
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenWeatherMap API error: {e}")
            raise

    def _get_open_meteo_current(
        self,
        latitude: float,
        longitude: float
    ) -> WeatherData:
        """Get current weather from Open-Meteo JMA API"""
        url = "https://api.open-meteo.com/v1/jma"
        params = {
            'latitude': latitude,
            'longitude': longitude,
            'current': 'temperature_2m,precipitation,relative_humidity_2m,'
                      'wind_speed_10m,weather_code,pressure_msl,visibility',
            'timezone': 'Asia/Tokyo'
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            current = data['current']

            # Map weather code to condition
            weather_code = current.get('weather_code', 0)
            condition = self._map_weather_code(weather_code)

            return WeatherData(
                timestamp=datetime.fromisoformat(current['time']),
                temperature=current['temperature_2m'],
                precipitation=current.get('precipitation', 0.0),
                humidity=current['relative_humidity_2m'],
                wind_speed=current['wind_speed_10m'],
                condition=condition,
                pressure=current['pressure_msl'],
                visibility=current.get('visibility', 10000) / 1000
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"Open-Meteo API error: {e}")
            raise

    def _get_jma_current(
        self,
        latitude: float,
        longitude: float
    ) -> WeatherData:
        """
        Get current weather from JMA Weather API (GitHub scraper)
        Note: This is an unofficial API that scrapes JMA data
        """
        # This would integrate with github.com/weather-jp/jma-weather-api
        # For now, return mock data structure
        raise NotImplementedError(
            "JMA Weather API integration requires additional setup"
        )

    def _get_openweathermap_historical(
        self,
        latitude: float,
        longitude: float,
        start_date: datetime,
        end_date: datetime
    ) -> List[WeatherData]:
        """Get historical weather from OpenWeatherMap One Call API"""
        weather_data = []
        current_date = start_date

        while current_date <= end_date:
            timestamp = int(current_date.timestamp())
            url = "https://api.openweathermap.org/data/3.0/onecall/timemachine"
            params = {
                'lat': latitude,
                'lon': longitude,
                'dt': timestamp,
                'appid': self.api_key,
                'units': 'metric'
            }

            try:
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                for hourly in data.get('data', []):
                    weather_data.append(WeatherData(
                        timestamp=datetime.fromtimestamp(hourly['dt']),
                        temperature=hourly['temp'],
                        precipitation=hourly.get('rain', {}).get('1h', 0.0),
                        humidity=hourly['humidity'],
                        wind_speed=hourly['wind_speed'],
                        condition=hourly['weather'][0]['main'].lower(),
                        pressure=hourly['pressure'],
                        visibility=hourly.get('visibility', 10000) / 1000
                    ))
            except requests.exceptions.RequestException as e:
                logger.warning(f"Failed to fetch data for {current_date}: {e}")

            current_date += timedelta(days=1)

        return weather_data

    def _get_open_meteo_historical(
        self,
        latitude: float,
        longitude: float,
        start_date: datetime,
        end_date: datetime
    ) -> List[WeatherData]:
        """Get historical weather from Open-Meteo Archive API"""
        url = "https://archive-api.open-meteo.com/v1/jma"
        params = {
            'latitude': latitude,
            'longitude': longitude,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'hourly': 'temperature_2m,precipitation,relative_humidity_2m,'
                     'wind_speed_10m,weather_code,pressure_msl',
            'timezone': 'Asia/Tokyo'
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            weather_data = []
            hourly = data['hourly']

            for i in range(len(hourly['time'])):
                weather_code = hourly.get('weather_code', [0] * len(hourly['time']))[i]
                condition = self._map_weather_code(weather_code)

                weather_data.append(WeatherData(
                    timestamp=datetime.fromisoformat(hourly['time'][i]),
                    temperature=hourly['temperature_2m'][i],
                    precipitation=hourly.get('precipitation', [0.0] * len(hourly['time']))[i],
                    humidity=hourly['relative_humidity_2m'][i],
                    wind_speed=hourly['wind_speed_10m'][i],
                    condition=condition,
                    pressure=hourly['pressure_msl'][i],
                    visibility=10.0  # Default visibility
                ))

            return weather_data
        except requests.exceptions.RequestException as e:
            logger.error(f"Open-Meteo Archive API error: {e}")
            raise

    @staticmethod
    def _map_weather_code(code: int) -> str:
        """Map WMO weather code to condition string"""
        if code == 0:
            return 'clear'
        elif code in [1, 2, 3]:
            return 'cloudy'
        elif code in [45, 48]:
            return 'fog'
        elif code in [51, 53, 55, 61, 63, 65, 80, 81, 82]:
            return 'rain'
        elif code in [71, 73, 75, 77, 85, 86]:
            return 'snow'
        elif code in [95, 96, 99]:
            return 'thunderstorm'
        else:
            return 'unknown'


class CalendarAPIClient:
    """
    Calendar and holiday API client for Japan

    Supported APIs:
    - Calendarific (commercial, free tier available)
    - Nager.Date (free public holidays API)
    - Custom calendar integration
    """

    def __init__(
        self,
        provider: str = "nager",
        api_key: Optional[str] = None
    ):
        """
        Initialize calendar API client

        Parameters:
        -----------
        provider : str
            Calendar API provider ('nager', 'calendarific', 'custom')
        api_key : str, optional
            API key for providers that require authentication
        """
        self.provider = provider.lower()
        self.api_key = api_key
        self.country_code = "JP"  # Japan

    def get_holidays(
        self,
        year: int
    ) -> List[CalendarEvent]:
        """
        Get national holidays for a specific year

        Parameters:
        -----------
        year : int
            Year to retrieve holidays for

        Returns:
        --------
        List[CalendarEvent]
            List of holiday events
        """
        if self.provider == "nager":
            return self._get_nager_holidays(year)
        elif self.provider == "calendarific":
            return self._get_calendarific_holidays(year)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _get_nager_holidays(self, year: int) -> List[CalendarEvent]:
        """Get holidays from Nager.Date API"""
        url = f"https://date.nager.at/api/v3/PublicHolidays/{year}/{self.country_code}"

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            holidays = response.json()

            events = []
            for holiday in holidays:
                events.append(CalendarEvent(
                    date=datetime.fromisoformat(holiday['date']),
                    event_type='holiday',
                    name=holiday['name'],
                    description=holiday.get('localName', ''),
                    impact_level='high' if holiday.get('global', False) else 'medium',
                    is_national=True,
                    is_recurring=True
                ))

            return events
        except requests.exceptions.RequestException as e:
            logger.error(f"Nager.Date API error: {e}")
            raise

    def _get_calendarific_holidays(self, year: int) -> List[CalendarEvent]:
        """Get holidays from Calendarific API"""
        if not self.api_key:
            raise ValueError("Calendarific requires an API key")

        url = "https://calendarific.com/api/v2/holidays"
        params = {
            'api_key': self.api_key,
            'country': self.country_code,
            'year': year
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            events = []
            for holiday in data['response']['holidays']:
                events.append(CalendarEvent(
                    date=datetime.fromisoformat(holiday['date']['iso']),
                    event_type='holiday',
                    name=holiday['name'],
                    description=holiday.get('description', ''),
                    impact_level=self._determine_impact_level(holiday),
                    is_national=holiday.get('type', ['national'])[0] == 'national',
                    is_recurring=True
                ))

            return events
        except requests.exceptions.RequestException as e:
            logger.error(f"Calendarific API error: {e}")
            raise

    @staticmethod
    def _determine_impact_level(holiday: Dict) -> str:
        """Determine impact level based on holiday type"""
        holiday_types = holiday.get('type', [])
        if 'national' in holiday_types:
            return 'high'
        elif 'observance' in holiday_types:
            return 'medium'
        else:
            return 'low'

    def add_custom_event(
        self,
        events_list: List[CalendarEvent],
        date: datetime,
        event_type: str,
        name: str,
        **kwargs
    ) -> List[CalendarEvent]:
        """
        Add custom event to events list

        Parameters:
        -----------
        events_list : List[CalendarEvent]
            Existing events list
        date : datetime
            Event date
        event_type : str
            Type of event (promotion, local_event, sports_event)
        name : str
            Event name
        **kwargs : dict
            Additional event parameters

        Returns:
        --------
        List[CalendarEvent]
            Updated events list
        """
        event = CalendarEvent(
            date=date,
            event_type=event_type,
            name=name,
            **kwargs
        )
        events_list.append(event)
        return events_list


class EconomicDataClient:
    """
    Economic indicators API client

    Supported APIs:
    - World Bank Data API
    - Trading Economics API
    - FRED API (Federal Reserve Economic Data)
    """

    def __init__(
        self,
        provider: str = "worldbank",
        api_key: Optional[str] = None
    ):
        """
        Initialize economic data API client

        Parameters:
        -----------
        provider : str
            Data provider ('worldbank', 'tradingeconomics', 'fred')
        api_key : str, optional
            API key for providers that require authentication
        """
        self.provider = provider.lower()
        self.api_key = api_key
        self.country_code = "JPN"  # Japan

    def get_indicators(
        self,
        start_date: datetime,
        end_date: datetime,
        indicators: Optional[List[str]] = None
    ) -> List[EconomicIndicator]:
        """
        Get economic indicators for date range

        Parameters:
        -----------
        start_date : datetime
            Start date
        end_date : datetime
            End date
        indicators : List[str], optional
            Specific indicators to retrieve

        Returns:
        --------
        List[EconomicIndicator]
            Economic indicator data
        """
        if self.provider == "worldbank":
            return self._get_worldbank_indicators(start_date, end_date, indicators)
        else:
            raise NotImplementedError(f"Provider {self.provider} not yet implemented")

    def _get_worldbank_indicators(
        self,
        start_date: datetime,
        end_date: datetime,
        indicators: Optional[List[str]] = None
    ) -> List[EconomicIndicator]:
        """Get indicators from World Bank API"""
        # Default indicators
        if indicators is None:
            indicators = [
                'NY.GDP.MKTP.KD.ZG',  # GDP growth
                'FP.CPI.TOTL.ZG',     # Inflation
                'SL.UEM.TOTL.ZS'      # Unemployment
            ]

        base_url = "https://api.worldbank.org/v2/country"
        economic_data = {}

        for indicator_code in indicators:
            url = f"{base_url}/{self.country_code}/indicator/{indicator_code}"
            params = {
                'format': 'json',
                'date': f"{start_date.year}:{end_date.year}",
                'per_page': 1000
            }

            try:
                response = requests.get(url, params=params, timeout=15)
                response.raise_for_status()
                data = response.json()

                if len(data) > 1:
                    for record in data[1]:
                        year = int(record['date'])
                        value = record['value']

                        if year not in economic_data:
                            economic_data[year] = {'year': year}

                        # Map indicator codes to fields
                        if indicator_code == 'NY.GDP.MKTP.KD.ZG':
                            economic_data[year]['gdp_growth'] = value
                        elif indicator_code == 'FP.CPI.TOTL.ZG':
                            economic_data[year]['inflation_rate'] = value
                        elif indicator_code == 'SL.UEM.TOTL.ZS':
                            economic_data[year]['unemployment_rate'] = value

            except requests.exceptions.RequestException as e:
                logger.warning(f"Failed to fetch {indicator_code}: {e}")

        # Convert to EconomicIndicator objects
        result = []
        for year, data in sorted(economic_data.items()):
            result.append(EconomicIndicator(
                date=datetime(year, 1, 1),
                gdp_growth=data.get('gdp_growth'),
                inflation_rate=data.get('inflation_rate'),
                unemployment_rate=data.get('unemployment_rate')
            ))

        return result


class ExternalFactorsAnalyzer:
    """
    Comprehensive external factors analysis for sales forecasting

    Features:
    - Multi-source data integration
    - Correlation analysis
    - Impact quantification
    - Feature engineering
    """

    def __init__(
        self,
        weather_client: Optional[WeatherAPIClient] = None,
        calendar_client: Optional[CalendarAPIClient] = None,
        economic_client: Optional[EconomicDataClient] = None
    ):
        """
        Initialize external factors analyzer

        Parameters:
        -----------
        weather_client : WeatherAPIClient, optional
            Weather data client
        calendar_client : CalendarAPIClient, optional
            Calendar events client
        economic_client : EconomicDataClient, optional
            Economic indicators client
        """
        self.weather_client = weather_client
        self.calendar_client = calendar_client
        self.economic_client = economic_client
        self.scaler = StandardScaler()

    def integrate_weather_data(
        self,
        sales_df: pd.DataFrame,
        latitude: float,
        longitude: float,
        date_column: str = 'date'
    ) -> pd.DataFrame:
        """
        Integrate weather data with sales data

        Parameters:
        -----------
        sales_df : pd.DataFrame
            Sales data with date column
        latitude : float
            Store location latitude
        longitude : float
            Store location longitude
        date_column : str
            Name of date column

        Returns:
        --------
        pd.DataFrame
            Sales data enriched with weather features
        """
        if self.weather_client is None:
            raise ValueError("Weather client not configured")

        df = sales_df.copy()
        df[date_column] = pd.to_datetime(df[date_column])

        # Get date range
        start_date = df[date_column].min()
        end_date = df[date_column].max()

        # Fetch weather data
        logger.info(f"Fetching weather data from {start_date} to {end_date}")
        weather_data = self.weather_client.get_historical_weather(
            latitude, longitude, start_date, end_date
        )

        # Convert to DataFrame
        weather_df = pd.DataFrame([w.to_dict() for w in weather_data])
        weather_df['date'] = pd.to_datetime(weather_df['timestamp']).dt.date

        # Aggregate to daily level
        daily_weather = weather_df.groupby('date').agg({
            'temperature': 'mean',
            'precipitation': 'sum',
            'humidity': 'mean',
            'wind_speed': 'mean',
            'pressure': 'mean',
            'visibility': 'mean'
        }).reset_index()

        # Add weather condition (most frequent)
        condition_mode = weather_df.groupby('date')['condition'].agg(
            lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown'
        ).reset_index()
        daily_weather = daily_weather.merge(condition_mode, on='date')

        # Merge with sales data
        df['date_only'] = df[date_column].dt.date
        df = df.merge(daily_weather, left_on='date_only', right_on='date', how='left')
        df.drop(['date_only', 'date'], axis=1, inplace=True)

        # One-hot encode weather condition
        if 'condition' in df.columns:
            condition_dummies = pd.get_dummies(df['condition'], prefix='weather')
            df = pd.concat([df, condition_dummies], axis=1)

        logger.info(f"Added {len(daily_weather.columns)} weather features")
        return df

    def integrate_calendar_events(
        self,
        sales_df: pd.DataFrame,
        date_column: str = 'date',
        custom_events: Optional[List[CalendarEvent]] = None
    ) -> pd.DataFrame:
        """
        Integrate calendar events with sales data

        Parameters:
        -----------
        sales_df : pd.DataFrame
            Sales data with date column
        date_column : str
            Name of date column
        custom_events : List[CalendarEvent], optional
            Additional custom events

        Returns:
        --------
        pd.DataFrame
            Sales data enriched with calendar features
        """
        if self.calendar_client is None:
            raise ValueError("Calendar client not configured")

        df = sales_df.copy()
        df[date_column] = pd.to_datetime(df[date_column])

        # Get unique years
        years = df[date_column].dt.year.unique()

        # Fetch holidays for all years
        all_events = []
        for year in years:
            logger.info(f"Fetching holidays for year {year}")
            holidays = self.calendar_client.get_holidays(year)
            all_events.extend(holidays)

        # Add custom events
        if custom_events:
            all_events.extend(custom_events)

        # Convert to DataFrame
        events_df = pd.DataFrame([e.to_dict() for e in all_events])
        events_df['date'] = pd.to_datetime(events_df['date']).dt.date

        # Create event features
        df['date_only'] = df[date_column].dt.date

        # Is holiday flag
        holiday_dates = events_df[events_df['event_type'] == 'holiday']['date'].unique()
        df['is_holiday'] = df['date_only'].isin(holiday_dates).astype(int)

        # Is promotion flag
        promotion_dates = events_df[events_df['event_type'] == 'promotion']['date'].unique()
        df['is_promotion'] = df['date_only'].isin(promotion_dates).astype(int)

        # Days to/from nearest holiday
        df['days_to_holiday'] = df['date_only'].apply(
            lambda x: self._days_to_nearest_event(x, holiday_dates)
        )

        # Event impact level
        impact_map = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
        df = df.merge(
            events_df[['date', 'impact_level']],
            left_on='date_only',
            right_on='date',
            how='left'
        )
        df['event_impact'] = df['impact_level'].map(impact_map).fillna(0)
        df.drop(['date', 'impact_level'], axis=1, inplace=True)

        df.drop(['date_only'], axis=1, inplace=True)

        logger.info(f"Added calendar event features")
        return df

    def integrate_economic_indicators(
        self,
        sales_df: pd.DataFrame,
        date_column: str = 'date'
    ) -> pd.DataFrame:
        """
        Integrate economic indicators with sales data

        Parameters:
        -----------
        sales_df : pd.DataFrame
            Sales data with date column
        date_column : str
            Name of date column

        Returns:
        --------
        pd.DataFrame
            Sales data enriched with economic indicators
        """
        if self.economic_client is None:
            raise ValueError("Economic client not configured")

        df = sales_df.copy()
        df[date_column] = pd.to_datetime(df[date_column])

        # Get date range
        start_date = df[date_column].min()
        end_date = df[date_column].max()

        # Fetch economic data
        logger.info(f"Fetching economic indicators")
        economic_data = self.economic_client.get_indicators(start_date, end_date)

        # Convert to DataFrame
        econ_df = pd.DataFrame([e.to_dict() for e in economic_data])
        econ_df['year'] = pd.to_datetime(econ_df['date']).dt.year

        # Merge with sales data (by year, as economic data is annual)
        df['year'] = df[date_column].dt.year
        df = df.merge(
            econ_df[['year', 'gdp_growth', 'inflation_rate', 'unemployment_rate']],
            on='year',
            how='left'
        )

        # Forward fill missing values
        df[['gdp_growth', 'inflation_rate', 'unemployment_rate']] = \
            df[['gdp_growth', 'inflation_rate', 'unemployment_rate']].fillna(method='ffill')

        logger.info(f"Added economic indicator features")
        return df

    def calculate_correlations(
        self,
        df: pd.DataFrame,
        target_column: str = 'sales',
        feature_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Calculate correlations between external factors and sales

        Parameters:
        -----------
        df : pd.DataFrame
            Data with external factors and sales
        target_column : str
            Sales/target column name
        feature_columns : List[str], optional
            Specific features to analyze (if None, uses all numeric)

        Returns:
        --------
        pd.DataFrame
            Correlation analysis results
        """
        if feature_columns is None:
            # Use all numeric columns except target
            feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_column in feature_columns:
                feature_columns.remove(target_column)

        correlations = []

        for feature in feature_columns:
            if feature in df.columns and target_column in df.columns:
                # Remove NaN values
                valid_data = df[[feature, target_column]].dropna()

                if len(valid_data) > 2:
                    # Pearson correlation
                    pearson_corr, pearson_p = stats.pearsonr(
                        valid_data[feature],
                        valid_data[target_column]
                    )

                    # Spearman correlation (for non-linear relationships)
                    spearman_corr, spearman_p = stats.spearmanr(
                        valid_data[feature],
                        valid_data[target_column]
                    )

                    correlations.append({
                        'feature': feature,
                        'pearson_correlation': pearson_corr,
                        'pearson_pvalue': pearson_p,
                        'spearman_correlation': spearman_corr,
                        'spearman_pvalue': spearman_p,
                        'abs_pearson': abs(pearson_corr),
                        'is_significant': pearson_p < 0.05
                    })

        corr_df = pd.DataFrame(correlations)
        corr_df = corr_df.sort_values('abs_pearson', ascending=False)

        logger.info(f"Calculated correlations for {len(correlations)} features")
        return corr_df

    def quantify_impact(
        self,
        df: pd.DataFrame,
        target_column: str = 'sales',
        factor_columns: Optional[List[str]] = None,
        method: str = 'regression'
    ) -> Dict[str, float]:
        """
        Quantify impact of external factors on sales

        Parameters:
        -----------
        df : pd.DataFrame
            Data with external factors and sales
        target_column : str
            Sales/target column name
        factor_columns : List[str], optional
            External factor columns to analyze
        method : str
            Impact quantification method ('regression', 'elasticity')

        Returns:
        --------
        Dict[str, float]
            Feature importance/impact scores
        """
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor

        if factor_columns is None:
            factor_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_column in factor_columns:
                factor_columns.remove(target_column)

        # Prepare data
        X = df[factor_columns].fillna(0)
        y = df[target_column]

        # Remove rows with NaN in target
        valid_idx = ~y.isna()
        X = X[valid_idx]
        y = y[valid_idx]

        if method == 'regression':
            # Use Random Forest for feature importance
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)

            importance = dict(zip(factor_columns, model.feature_importances_))

        elif method == 'elasticity':
            # Calculate elasticity (% change in sales / % change in factor)
            importance = {}

            for feature in factor_columns:
                # Avoid division by zero
                if X[feature].std() > 0 and y.std() > 0:
                    # Normalized elasticity
                    elasticity = (X[feature].std() / X[feature].mean()) / \
                                (y.std() / y.mean()) if X[feature].mean() != 0 else 0
                    importance[feature] = abs(elasticity)
                else:
                    importance[feature] = 0.0

        else:
            raise ValueError(f"Unknown method: {method}")

        # Sort by importance
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

        logger.info(f"Quantified impact for {len(importance)} factors using {method}")
        return importance

    @staticmethod
    def _days_to_nearest_event(date, event_dates) -> int:
        """Calculate days to nearest event (negative if past)"""
        if len(event_dates) == 0:
            return 999

        event_dates = pd.to_datetime(event_dates)
        date = pd.to_datetime(date)

        future_events = event_dates[event_dates >= date]
        past_events = event_dates[event_dates < date]

        if len(future_events) > 0:
            days_ahead = (future_events.min() - date).days
        else:
            days_ahead = 999

        if len(past_events) > 0:
            days_behind = -(date - past_events.max()).days
        else:
            days_behind = -999

        # Return the closest event (positive for future, negative for past)
        if abs(days_ahead) <= abs(days_behind):
            return days_ahead
        else:
            return days_behind

    def create_feature_summary(
        self,
        df: pd.DataFrame,
        target_column: str = 'sales'
    ) -> pd.DataFrame:
        """
        Create comprehensive summary of external factors analysis

        Parameters:
        -----------
        df : pd.DataFrame
            Data with external factors
        target_column : str
            Sales/target column name

        Returns:
        --------
        pd.DataFrame
            Feature analysis summary
        """
        # Get correlations
        correlations = self.calculate_correlations(df, target_column)

        # Get impact scores
        impact_scores = self.quantify_impact(df, target_column, method='regression')

        # Merge results
        summary = correlations.copy()
        summary['rf_importance'] = summary['feature'].map(impact_scores)

        # Add feature statistics
        for feature in summary['feature']:
            if feature in df.columns:
                summary.loc[summary['feature'] == feature, 'mean_value'] = df[feature].mean()
                summary.loc[summary['feature'] == feature, 'std_value'] = df[feature].std()
                summary.loc[summary['feature'] == feature, 'missing_pct'] = \
                    (df[feature].isna().sum() / len(df)) * 100

        return summary


# Example usage and testing
if __name__ == "__main__":
    # This section demonstrates how to use the external factors module

    print("=" * 80)
    print("External Factors Analysis Module - Example Usage")
    print("=" * 80)

    # Example 1: Weather Data Integration
    print("\n1. Weather Data Integration")
    print("-" * 80)

    # Initialize weather client (use Open-Meteo for free access)
    weather_client = WeatherAPIClient(provider="open-meteo")

    # Tokyo coordinates (example convenience store location)
    tokyo_lat, tokyo_lon = 35.6762, 139.6503

    # Get current weather
    try:
        current_weather = weather_client.get_current_weather(tokyo_lat, tokyo_lon)
        print(f"Current weather in Tokyo:")
        print(f"  Temperature: {current_weather.temperature}°C")
        print(f"  Condition: {current_weather.condition}")
        print(f"  Humidity: {current_weather.humidity}%")
        print(f"  Precipitation: {current_weather.precipitation}mm")
    except Exception as e:
        print(f"  Error fetching weather: {e}")

    # Example 2: Calendar Events
    print("\n2. Calendar Events Integration")
    print("-" * 80)

    calendar_client = CalendarAPIClient(provider="nager")

    try:
        holidays_2025 = calendar_client.get_holidays(2025)
        print(f"Found {len(holidays_2025)} holidays in 2025:")
        for holiday in holidays_2025[:5]:  # Show first 5
            print(f"  {holiday.date.strftime('%Y-%m-%d')}: {holiday.name}")
    except Exception as e:
        print(f"  Error fetching holidays: {e}")

    # Example 3: Complete Analysis Pipeline
    print("\n3. Complete Analysis Pipeline")
    print("-" * 80)

    # Create sample sales data
    date_range = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
    sample_sales_df = pd.DataFrame({
        'date': date_range,
        'sales': np.random.normal(100000, 20000, len(date_range))
    })

    print(f"Sample sales data shape: {sample_sales_df.shape}")

    # Initialize analyzer
    analyzer = ExternalFactorsAnalyzer(
        weather_client=weather_client,
        calendar_client=calendar_client
    )

    # Integrate calendar events
    try:
        enriched_df = analyzer.integrate_calendar_events(sample_sales_df)
        print(f"After calendar integration: {enriched_df.shape}")
        print(f"New features: {set(enriched_df.columns) - set(sample_sales_df.columns)}")
    except Exception as e:
        print(f"  Error in calendar integration: {e}")

    print("\n" + "=" * 80)
    print("Module loaded successfully!")
    print("=" * 80)
