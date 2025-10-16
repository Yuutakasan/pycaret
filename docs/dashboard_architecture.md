# Store Manager Dashboard Architecture

## Architecture Decision Record (ADR)

**Date:** 2025-10-08
**Status:** Proposed
**Context:** Design scalable, modular dashboard for store managers with multi-store analysis capabilities
**Decision:** Microservices-based architecture with event-driven data pipeline and layered visualization

---

## 1. Executive Summary

This document outlines the architecture for a comprehensive Store Manager Dashboard supporting:
- Multi-store management and benchmarking
- Five core analysis modules (時系列分析、ABC分析、発注最適化、外部要因、需要予測)
- Multi-granularity time analysis (monthly/weekly/daily)
- Real-time alerts and AI-powered recommendations
- Interactive visualizations with performance optimization

### Architecture Principles
1. **Modularity**: Each analysis type is an independent, pluggable module
2. **Scalability**: Horizontal scaling for multi-store deployments
3. **Performance**: Sub-second response times with intelligent caching
4. **Extensibility**: Easy addition of new analysis types or stores
5. **Reliability**: 99.9% uptime with graceful degradation

---

## 2. System Architecture Overview

### 2.1 C4 Model - Context Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     Store Manager Dashboard                      │
│                        (System Context)                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  External Users:                    External Systems:            │
│  ┌──────────────┐                  ┌──────────────┐            │
│  │ Store Manager│◄────────────────►│ POS System   │            │
│  └──────────────┘                  └──────────────┘            │
│  ┌──────────────┐                  ┌──────────────┐            │
│  │ Regional Mgr │◄────────────────►│ Inventory DB │            │
│  └──────────────┘                  └──────────────┘            │
│  ┌──────────────┐                  ┌──────────────┐            │
│  │   Analyst    │◄────────────────►│ Weather API  │            │
│  └──────────────┘                  └──────────────┘            │
│                                     ┌──────────────┐            │
│                                     │ Calendar API │            │
│                                     └──────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 C4 Model - Container Diagram

```
┌────────────────────────────────────────────────────────────────────┐
│                    Store Manager Dashboard System                   │
├────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │               Frontend Layer (React/Next.js)                  │ │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐              │ │
│  │  │  Dashboard │ │   Charts   │ │   Alerts   │              │ │
│  │  │  Shell     │ │  Component │ │   Panel    │              │ │
│  │  └────────────┘ └────────────┘ └────────────┘              │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                              │                                      │
│                              ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │                API Gateway (GraphQL/REST)                     │ │
│  │              Authentication & Authorization                   │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                              │                                      │
│         ┌────────────────────┼────────────────────┐               │
│         ▼                    ▼                    ▼               │
│  ┌─────────────┐      ┌─────────────┐     ┌─────────────┐       │
│  │  Analysis   │      │   Store     │     │   Alert &   │       │
│  │  Service    │      │   Service   │     │  Recommend  │       │
│  │  Layer      │      │   Layer     │     │   Engine    │       │
│  └─────────────┘      └─────────────┘     └─────────────┘       │
│         │                    │                    │               │
│         └────────────────────┼────────────────────┘               │
│                              ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │              Data Access Layer (ORM/Query Builder)            │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                              │                                      │
│         ┌────────────────────┼────────────────────┐               │
│         ▼                    ▼                    ▼               │
│  ┌─────────────┐      ┌─────────────┐     ┌─────────────┐       │
│  │ PostgreSQL  │      │    Redis    │     │  Time-Series│       │
│  │  Database   │      │    Cache    │     │  DB (Influx)│       │
│  └─────────────┘      └─────────────┘     └─────────────┘       │
│                                                                      │
└────────────────────────────────────────────────────────────────────┘
```

---

## 3. Modular Analysis Structure

### 3.1 Analysis Module Architecture

Each analysis type follows a consistent module pattern:

```
┌─────────────────────────────────────────────────────────────┐
│                    Analysis Module Pattern                   │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  1. Data Ingestion Layer                            │   │
│  │     - Store filter application                       │   │
│  │     - Time range selection                           │   │
│  │     - Data validation & normalization                │   │
│  └─────────────────────────────────────────────────────┘   │
│                        │                                     │
│                        ▼                                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  2. Processing Engine                                │   │
│  │     - Analysis algorithm execution                   │   │
│  │     - Multi-granularity computation                  │   │
│  │     - Cross-store comparison logic                   │   │
│  └─────────────────────────────────────────────────────┘   │
│                        │                                     │
│                        ▼                                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  3. Results Aggregation                              │   │
│  │     - Metrics calculation                            │   │
│  │     - Trend identification                           │   │
│  │     - Anomaly detection                              │   │
│  └─────────────────────────────────────────────────────┘   │
│                        │                                     │
│                        ▼                                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  4. Visualization Preparation                        │   │
│  │     - Data transformation for charts                 │   │
│  │     - UI state generation                            │   │
│  │     - Export formatting                              │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Five Core Analysis Modules

#### Module 1: 時系列分析 (Time Series Analysis)

```yaml
Module: TimeSeriesAnalysisModule
Purpose: Track and forecast sales/inventory trends over time

Components:
  - DataCollector:
      Responsibilities:
        - Aggregate sales data by time granularity
        - Handle missing data imputation
        - Seasonal adjustment
      Inputs: [store_id, product_id, date_range, granularity]
      Outputs: TimeSeries data structure

  - TrendAnalyzer:
      Algorithms:
        - Moving Average (MA)
        - Exponential Smoothing (ES)
        - ARIMA modeling
        - Prophet for multi-seasonality
      Outputs: Trend components, seasonality patterns

  - VisualizationEngine:
      Chart Types:
        - Line charts (multi-store overlay)
        - Area charts (stacked for categories)
        - Candlestick charts (for volatility)
        - Heatmaps (time × store matrix)

  - AlertTriggers:
      Conditions:
        - Trend reversal detection
        - Seasonal deviation > 2σ
        - Growth rate change > threshold

Performance Targets:
  - Response time: < 500ms for 1-year daily data
  - Cache hit rate: > 85%
  - Concurrent users: 100+ per store
```

#### Module 2: ABC分析 (ABC Analysis)

```yaml
Module: ABCAnalysisModule
Purpose: Categorize products by revenue contribution

Components:
  - RevenueCalculator:
      Metrics:
        - Total revenue per SKU
        - Cumulative revenue percentage
        - Transaction frequency
      Classification: Pareto principle (80-20 rule)

  - CategoryAssigner:
      Categories:
        - A: Top 20% (80% revenue)
        - B: Next 30% (15% revenue)
        - C: Remaining 50% (5% revenue)
      Dynamic thresholds per store/region

  - ComparisonEngine:
      Cross-Store Metrics:
        - Category distribution variance
        - A-product overlap rate
        - Revenue concentration index

  - VisualizationEngine:
      Chart Types:
        - Pareto charts
        - Pie charts (category distribution)
        - Scatter plots (quantity vs revenue)
        - Matrix view (store × category)

Performance Targets:
  - Calculation time: < 200ms for 10K SKUs
  - Real-time updates: Every 15 minutes
  - Historical comparison: 12-month lookback
```

#### Module 3: 発注最適化 (Order Optimization)

```yaml
Module: OrderOptimizationModule
Purpose: Optimize inventory ordering and reduce waste

Components:
  - InventoryAnalyzer:
      Metrics:
        - Current stock levels
        - Days of inventory (DOI)
        - Stockout frequency
        - Overstock percentage

  - DemandForecaster:
      Algorithms:
        - Lead time demand calculation
        - Safety stock optimization
        - Economic Order Quantity (EOQ)
        - Reorder Point (ROP) determination

  - OptimizationEngine:
      Constraints:
        - Budget limits
        - Storage capacity
        - Supplier MOQ (Minimum Order Quantity)
        - Shelf life considerations
      Objectives:
        - Minimize total cost
        - Maximize service level (95%+)

  - RecommendationGenerator:
      Output:
        - Suggested order quantities
        - Optimal order timing
        - Alternative suppliers
        - Bulk discount opportunities

  - VisualizationEngine:
      Chart Types:
        - Inventory level timeline
        - Order simulation scenarios
        - Cost breakdown waterfall
        - What-if analysis tables

Performance Targets:
  - Optimization solve time: < 2 seconds
  - Forecast accuracy: MAPE < 15%
  - Cost reduction: 10-15% target
```

#### Module 4: 外部要因 (External Factors Analysis)

```yaml
Module: ExternalFactorsModule
Purpose: Correlate sales with weather, events, holidays

Components:
  - ExternalDataIntegrator:
      Data Sources:
        - Weather API (temperature, precipitation, conditions)
        - Calendar API (holidays, events)
        - Economic indicators (optional)
        - Local events database

  - CorrelationAnalyzer:
      Algorithms:
        - Pearson/Spearman correlation
        - Multivariate regression
        - Lag analysis (delayed effects)
        - Interaction effects

  - ImpactQuantifier:
      Metrics:
        - Sales lift percentage per factor
        - Attribution modeling
        - Incremental revenue calculation

  - PredictiveModeler:
      Capabilities:
        - Factor-adjusted forecasting
        - Scenario planning (what-if)
        - Event impact prediction

  - VisualizationEngine:
      Chart Types:
        - Correlation heatmaps
        - Dual-axis line charts (sales vs weather)
        - Event markers on timelines
        - Attribution waterfall

Performance Targets:
  - API response time: < 1 second
  - Correlation update: Daily at 2 AM
  - Historical depth: 24 months
```

#### Module 5: 需要予測 (Demand Forecasting)

```yaml
Module: DemandForecastingModule
Purpose: Predict future demand with high accuracy

Components:
  - FeatureEngineer:
      Features:
        - Historical sales patterns
        - Trend and seasonality
        - Promotional calendar
        - External factors (from Module 4)
        - Product lifecycle stage

  - ModelEnsemble:
      Models:
        - Prophet (baseline)
        - LSTM neural networks
        - XGBoost (feature-rich)
        - Exponential smoothing
      Ensemble Method: Weighted average by recent accuracy

  - AccuracyMonitor:
      Metrics:
        - MAPE (Mean Absolute Percentage Error)
        - RMSE (Root Mean Square Error)
        - Forecast bias
        - Coverage probability
      Real-time validation against actuals

  - ScenarioSimulator:
      Capabilities:
        - Promotional impact modeling
        - New product introduction
        - Competitive response
        - Supply chain disruption

  - VisualizationEngine:
      Chart Types:
        - Forecast fan charts (confidence intervals)
        - Actual vs predicted comparison
        - Accuracy trend over time
        - Scenario comparison tables

Performance Targets:
  - Forecast horizon: 4-12 weeks
  - Update frequency: Daily
  - Accuracy: MAPE < 12% for A-items
  - Model training: Weekly refresh
```

---

## 4. Store-Level Filtering & Comparison Framework

### 4.1 Store Hierarchy Model

```
┌────────────────────────────────────────────────────────┐
│              Store Hierarchy Structure                  │
├────────────────────────────────────────────────────────┤
│                                                          │
│  Organization (企業)                                    │
│    │                                                     │
│    ├─ Region (地域)                                     │
│    │    │                                               │
│    │    ├─ District (地区)                              │
│    │    │    │                                          │
│    │    │    ├─ Store Format (店舗形態)                │
│    │    │    │    │                                     │
│    │    │    │    ├─ Individual Store (店舗)           │
│    │    │    │    │    │                                │
│    │    │    │    │    ├─ Department (部門)            │
│    │    │    │    │    │    │                           │
│    │    │    │    │    │    ├─ Category (カテゴリ)      │
│    │    │    │    │    │    │    │                      │
│    │    │    │    │    │    │    └─ SKU (商品)         │
│                                                          │
└────────────────────────────────────────────────────────┘
```

### 4.2 Filter System Architecture

```yaml
FilteringFramework:
  Components:
    - FilterBuilder:
        UI Elements:
          - Multi-select dropdown (stores)
          - Cascading filters (region → district → store)
          - Search autocomplete
          - Tag-based selection
        Persistence: User preferences in localStorage

    - FilterEngine:
        Capabilities:
          - Dynamic query generation
          - Filter validation
          - Performance optimization (index hints)
        Query Patterns:
          - Single store: WHERE store_id = ?
          - Multi-store: WHERE store_id IN (?)
          - Region/district: JOIN via hierarchy table

    - ComparisonCoordinator:
        Modes:
          - Side-by-side comparison (2-4 stores)
          - Normalized comparison (percentage basis)
          - Benchmark comparison (vs average/best)
        Sync Controls: Linked time ranges, zoom levels

    - AggregationService:
        Rollup Logic:
          - SUM for sales, quantity
          - AVG for prices, margins
          - WEIGHTED_AVG for store-level metrics
          - PERCENTILE for distribution analysis

Database Schema:
  Tables:
    - stores:
        Columns: [id, name, region_id, district_id, format, size_tier]
        Indexes: [region_id, district_id], [format]

    - store_hierarchies:
        Columns: [child_id, parent_id, hierarchy_level]
        Indexes: [child_id], [parent_id]
        Purpose: Fast ancestor/descendant queries

    - store_metrics_cache:
        Columns: [store_id, metric_type, time_period, value, updated_at]
        Indexes: [store_id, metric_type, time_period]
        TTL: 1 hour for real-time, 24 hours for historical
```

### 4.3 Cross-Store Benchmarking

```yaml
BenchmarkingSystem:
  Peer Group Selection:
    Criteria:
      - Store format similarity
      - Size tier (revenue/sqft bands)
      - Regional clustering
      - Customer demographic match
    Algorithm: K-means clustering + manual override

  Benchmark Metrics:
    Sales Performance:
      - Sales per square foot
      - Average transaction value
      - Items per transaction
      - Conversion rate

    Inventory Efficiency:
      - Inventory turnover ratio
      - Days of inventory
      - Stockout rate
      - Shrinkage percentage

    Product Mix:
      - ABC distribution similarity
      - Category contribution variance
      - New product adoption rate

  Visualization:
    - Percentile ranking (P25, P50, P75, P90)
    - Gap analysis (vs top quartile)
    - Radar charts (multi-metric comparison)
    - League tables (sortable rankings)

  Performance Targets:
    - Peer group refresh: Monthly
    - Benchmark calculation: Daily
    - Historical depth: 24 months
```

---

## 5. Multi-Granularity Time Analysis

### 5.1 Time Dimension Architecture

```yaml
TimeGranularityFramework:
  Supported Granularities:
    - 月次 (Monthly):
        Grain: Month start date
        Use Cases: Strategic planning, trend analysis
        Aggregation: SUM over month

    - 週次 (Weekly):
        Grain: Week start (Monday/Sunday configurable)
        Use Cases: Operational planning, promotional analysis
        Aggregation: SUM over ISO week

    - 日次 (Daily):
        Grain: Calendar date
        Use Cases: Tactical decisions, anomaly detection
        Aggregation: Raw daily values

    - 時間帯 (Hourly - Optional):
        Grain: Hour of day
        Use Cases: Staffing optimization, peak analysis
        Aggregation: AVG by hour across days

  Implementation:
    Database Layer:
      - time_dimensions table:
          Columns:
            - date_key (YYYYMMDD integer)
            - date_actual (DATE)
            - month_key (YYYYMM)
            - week_key (YYYYWW)
            - day_of_week
            - is_holiday
            - fiscal_period
          Indexes: All key columns

      - Materialized views per granularity:
          - sales_monthly_mv
          - sales_weekly_mv
          - sales_daily_mv
        Refresh: Incremental (last 7 days daily)

    API Layer:
      - Endpoint: /api/analytics/{module}?granularity={monthly|weekly|daily}
      - Query builder: Dynamic GROUP BY based on granularity
      - Cache strategy: Separate cache keys per granularity

    Frontend Layer:
      - Granularity selector component
      - Dynamic chart re-rendering
      - Drill-down capability (month → week → day)
      - Drill-up aggregation (day → week → month)

  Performance Optimization:
    - Pre-aggregated summaries for common queries
    - Lazy loading for drill-down (fetch on demand)
    - Client-side caching with stale-while-revalidate
    - Backend query result caching (Redis)

  User Experience:
    - Default view: Weekly (balance detail vs overview)
    - Quick toggle buttons (月/週/日)
    - Automatic granularity suggestion based on date range:
        - < 7 days: Daily
        - 7-90 days: Weekly
        - > 90 days: Monthly
```

### 5.2 Time Comparison Modes

```yaml
ComparisonModes:
  - Year-over-Year (YoY):
      Calculation: Current period vs same period last year
      Alignment: Same day-of-week, holiday-adjusted
      Use Cases: Growth tracking, seasonality validation

  - Period-over-Period (PoP):
      Calculation: Current vs previous period (month, week, day)
      Use Cases: Short-term trend analysis

  - Moving Average Comparison:
      Window: 4-week, 13-week, 52-week
      Use Cases: Smoothing volatility, trend identification

  - Cumulative Comparison:
      Calculation: Year-to-date, quarter-to-date
      Use Cases: Progress tracking against targets

  Visualization:
    - Dual-axis charts (current vs prior)
    - Variance bridges (waterfall charts)
    - Percentage change indicators
    - Color coding (green/red for positive/negative)
```

---

## 6. Alert & Recommendation Engine

### 6.1 Alert System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Alert Engine Pipeline                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  [1] Event Detection                                         │
│      ├─ Continuous monitoring (stream processing)            │
│      ├─ Anomaly detection algorithms                         │
│      ├─ Threshold breach detection                           │
│      └─ Pattern recognition (ML models)                      │
│                      │                                        │
│                      ▼                                        │
│  [2] Alert Classification                                    │
│      ├─ Severity scoring (Critical/High/Medium/Low)          │
│      ├─ Impact assessment (revenue/customer/operations)      │
│      ├─ Urgency determination (immediate/today/this week)    │
│      └─ Actionability check (can user respond?)              │
│                      │                                        │
│                      ▼                                        │
│  [3] Alert Filtering & Deduplication                         │
│      ├─ Duplicate suppression (same alert, 24hr window)      │
│      ├─ Alert fatigue prevention (max 5 critical/day)        │
│      ├─ User preference filtering                            │
│      └─ Context-aware grouping                               │
│                      │                                        │
│                      ▼                                        │
│  [4] Notification Routing                                    │
│      ├─ In-app notification center                           │
│      ├─ Email digest (daily summary)                         │
│      ├─ SMS for critical alerts (opt-in)                     │
│      └─ Webhook for integrations (Slack, Teams)              │
│                      │                                        │
│                      ▼                                        │
│  [5] Alert Tracking & Resolution                             │
│      ├─ Acknowledgment tracking                              │
│      ├─ Action logging                                       │
│      ├─ Auto-resolution (conditions met)                     │
│      └─ Feedback loop (was alert useful?)                    │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Alert Types by Module

```yaml
TimeSeriesAlerts:
  - Trend Reversal:
      Trigger: Trend direction change sustained > 3 periods
      Severity: Medium
      Action: Review cause, adjust forecasts

  - Seasonal Anomaly:
      Trigger: Actual deviates > 2σ from seasonal expectation
      Severity: High if revenue impact > $X
      Action: Investigate local factors, supply issues

  - Rapid Decline:
      Trigger: Sales drop > 20% WoW for 2+ weeks
      Severity: Critical
      Action: Emergency review, promotional response

ABCAlerts:
  - Category Shift:
      Trigger: A-product drops to B or B to C
      Severity: High
      Action: Root cause analysis, marketing review

  - Concentration Risk:
      Trigger: Top 10 SKUs represent > 50% revenue
      Severity: Medium
      Action: Diversification strategy

  - Cross-Store Divergence:
      Trigger: ABC distribution variance > threshold vs peers
      Severity: Low
      Action: Best practice sharing

OrderOptimizationAlerts:
  - Stockout Imminent:
      Trigger: Projected stockout within lead time
      Severity: Critical
      Action: Expedite order, find alternatives

  - Overstock Warning:
      Trigger: DOI > 60 days for perishables, > 90 for others
      Severity: Medium
      Action: Markdown planning, transfer to other stores

  - Order Recommendation:
      Trigger: Optimal order point reached
      Severity: Low (informational)
      Action: Review and approve suggested order

ExternalFactorsAlerts:
  - Weather Impact:
      Trigger: Forecast conditions correlate with sales change
      Severity: Medium
      Action: Adjust staffing, inventory positioning

  - Event Opportunity:
      Trigger: Upcoming event with historical sales lift
      Severity: Low
      Action: Prepare promotional plans, ensure stock

DemandForecastAlerts:
  - Forecast Accuracy Degradation:
      Trigger: MAPE increases > 5% vs baseline
      Severity: Medium
      Action: Model retraining, feature review

  - Demand Spike Predicted:
      Trigger: Forecast shows > 30% increase vs normal
      Severity: High
      Action: Capacity planning, supply chain prep
```

### 6.3 Recommendation Engine

```yaml
RecommendationEngine:
  Architecture:
    - Rule-Based System:
        Use: Well-defined business logic
        Examples:
          - "Reorder when inventory < ROP"
          - "Mark down slow-movers after 30 days"
        Pros: Explainable, reliable

    - Machine Learning Models:
        Use: Complex pattern recognition
        Models:
          - Collaborative filtering (similar stores)
          - Reinforcement learning (action optimization)
          - Neural networks (demand patterns)
        Pros: Adaptive, discovers hidden patterns

    - Hybrid Approach:
        Combination: Rules provide guardrails, ML optimizes within bounds
        Example: ML suggests order qty, rules enforce MOQ/budget

  Recommendation Types:
    - Inventory Actions:
        - "Order 50 units of SKU-123 by tomorrow"
        - "Transfer 20 units to Store B (high demand)"
        - "Mark down Product X by 15% to clear"

    - Pricing Suggestions:
        - "Increase price 5% (low price elasticity)"
        - "Match competitor price on Product Y"
        - "Bundle Products A+B for 10% discount"

    - Promotional Opportunities:
        - "Promote umbrellas (rain forecast next week)"
        - "Feature holiday items starting Oct 15"
        - "Cross-sell Product Z with Product A"

    - Operational Improvements:
        - "Add staff on Fridays 5-7 PM (peak traffic)"
        - "Expand shelf space for Category X (underperforming)"
        - "Benchmark Store C for best practices (top quartile)"

  Prioritization:
    Scoring Criteria:
      - Expected ROI (revenue impact × probability)
      - Urgency (time sensitivity)
      - Ease of implementation (low/medium/high effort)
      - Strategic alignment (fits business goals)

    Display:
      - Top 5 recommendations per module
      - "Quick wins" section (high impact, low effort)
      - Action buttons (approve, dismiss, snooze)
      - Success tracking (track outcomes)

  Personalization:
    - User role-based (store manager vs analyst)
    - Store context (size, format, performance tier)
    - Learning from past actions (accept/reject history)
    - Seasonal adjustment (holiday periods, etc.)
```

---

## 7. Interactive Visualization Layer

### 7.1 Visualization Technology Stack

```yaml
TechnologyStack:
  Frontend Framework:
    - React 18+ (concurrent features)
    - Next.js 14 (App Router)
    - TypeScript (type safety)

  Charting Libraries:
    Primary: Recharts (declarative, responsive)
    Advanced: D3.js (custom visualizations)
    Mapping: Mapbox GL JS (geographic analysis)

  State Management:
    - Zustand (lightweight, performant)
    - TanStack Query (server state, caching)

  Styling:
    - Tailwind CSS (utility-first)
    - Shadcn/ui (component library)
    - Framer Motion (animations)

  Data Virtualization:
    - TanStack Virtual (large lists/tables)
    - React Window (scrolling performance)
```

### 7.2 Chart Component Library

```yaml
ChartComponents:
  TimeSeries:
    - LineChart:
        Use: Trend visualization
        Features:
          - Multi-series (up to 10 stores)
          - Zoom/pan interaction
          - Crosshair with tooltip
          - Annotations (events, alerts)
        Performance: Virtualized rendering for > 1000 points

    - AreaChart:
        Use: Volume/magnitude with trends
        Features:
          - Stacked or overlapping
          - Gradient fills
          - Negative value support (variance)

    - CandlestickChart:
        Use: High/low/open/close analysis
        Features:
          - OHLC display
          - Volume bars
          - Moving average overlays

  Categorical:
    - BarChart:
        Use: Comparisons across categories
        Variants:
          - Grouped (multi-store comparison)
          - Stacked (part-to-whole)
          - Horizontal (long labels)
        Features:
          - Sort/filter inline
          - Drill-down on click

    - ParetoChart:
        Use: ABC analysis
        Features:
          - Combined bar + line (cumulative %)
          - Category threshold lines
          - Dynamic classification

  Distribution:
    - Histogram:
        Use: Frequency distribution
        Features:
          - Adjustable bin size
          - Overlay normal curve
          - Outlier highlighting

    - BoxPlot:
        Use: Statistical dispersion
        Features:
          - Multi-store comparison
          - Outlier detection
          - Quartile visualization

  Correlation:
    - ScatterPlot:
        Use: Relationship analysis
        Features:
          - Regression line
          - Cluster highlighting
          - Size/color encoding (3rd/4th dimension)

    - Heatmap:
        Use: Matrix correlation, time patterns
        Features:
          - Diverging color scales
          - Cell value display
          - Row/column sorting

  Geographic:
    - ChoroplethMap:
        Use: Regional performance
        Features:
          - Color intensity by metric
          - Interactive tooltips
          - Store location markers

  Specialty:
    - WaterfallChart:
        Use: Variance analysis, contribution
        Features:
          - Sequential breakdown
          - Positive/negative encoding
          - Running total line

    - RadarChart:
        Use: Multi-metric comparison
        Features:
          - Overlay multiple stores
          - Normalized scales
          - Area fill

    - SankeyDiagram:
        Use: Flow analysis (customer journey, etc.)
        Features:
          - Interactive node dragging
          - Highlight path on hover
```

### 7.3 Interactive Features

```yaml
InteractivityDesign:
  User Actions:
    - Hover:
        Triggers:
          - Tooltip display (data point details)
          - Highlight related elements
          - Preview associated data
        Performance: Debounced to 50ms

    - Click:
        Actions:
          - Drill-down (navigate to detail view)
          - Filter (isolate data subset)
          - Select (multi-select with Ctrl/Cmd)
        Feedback: Visual selection state, loading indicator

    - Drag:
        Uses:
          - Time range selection (brush)
          - Reorder dashboard widgets
          - Adjust threshold lines
        Constraints: Snap to valid values, boundary checks

    - Zoom/Pan:
        Implementation:
          - Mouse wheel zoom
          - Pinch gesture (touch devices)
          - Zoom controls (buttons)
        Features:
          - Synchronized across linked charts
          - Reset button
          - Mini-map navigator

  Filtering & Slicing:
    - Global Filters:
        Scope: Apply to all modules
        Types:
          - Date range picker (preset + custom)
          - Store multi-select
          - Product category selector
        Persistence: URL query params (shareable links)

    - Local Filters:
        Scope: Single chart/module
        Types:
          - Legend toggle (hide/show series)
          - Data table column filters
          - Chart-specific controls
        State: Component-local (not shared)

  Export & Sharing:
    - Export Formats:
        - PNG/SVG (charts)
        - Excel/CSV (data tables)
        - PDF (full dashboard report)
        - PowerPoint (selected charts)

    - Sharing:
        - Permalink (captures all filters/state)
        - Email report (scheduled/on-demand)
        - Embed code (iframe for other systems)
        - API endpoint (programmatic access)
```

### 7.4 Dashboard Layout System

```yaml
LayoutArchitecture:
  Layout Engine:
    - Library: React Grid Layout
    - Features:
      - Drag-and-drop rearrangement
      - Resize widgets
      - Responsive breakpoints (desktop/tablet/mobile)
      - Layout persistence (user preferences)

  Widget System:
    - Widget Types:
        - Chart Widget (any chart type)
        - Metric Card (KPI display)
        - Table Widget (data grid)
        - Alert List (notification panel)
        - Text/Markdown (annotations)

    - Widget Configuration:
        - Data source selection
        - Visual customization (colors, labels)
        - Refresh rate (real-time, 5min, 1hr, manual)
        - Size constraints (min/max width/height)

  Preset Layouts:
    - Executive Dashboard:
        Widgets:
          - Sales YoY comparison (line chart)
          - Top 10 products (bar chart)
          - Alert summary (card)
          - Store performance map (geo chart)
        Target User: Regional managers

    - Inventory Manager Dashboard:
        Widgets:
          - Stockout alerts (list)
          - DOI distribution (histogram)
          - ABC classification (Pareto)
          - Order recommendations (table)
        Target User: Store managers

    - Analyst Dashboard:
        Widgets:
          - Time series with seasonality (line + area)
          - Correlation matrix (heatmap)
          - Forecast accuracy (scatter)
          - External factors impact (dual-axis)
        Target User: Business analysts

  Customization:
    - User can:
        - Create new dashboards
        - Clone/modify presets
        - Save multiple layouts
        - Share layouts with team
    - Admin can:
        - Set default layouts by role
        - Lock certain widgets (mandated KPIs)
        - Define widget catalog (available types)
```

---

## 8. Performance Optimization Strategy

### 8.1 Database Optimization

```yaml
DatabaseLayer:
  Indexing Strategy:
    - Primary Indexes:
        - store_id (most queries filter by store)
        - date_key (time-based partitioning)
        - product_id (SKU-level analysis)
        - Composite: (store_id, date_key) for common queries

    - Covering Indexes:
        - Include frequently selected columns
        - Avoid index-only scans where possible
        - Example: CREATE INDEX idx_sales_metrics ON sales(store_id, date_key) INCLUDE (revenue, quantity)

    - Partial Indexes:
        - For specific query patterns
        - Example: CREATE INDEX idx_high_value ON sales(store_id, date_key) WHERE revenue > 1000

  Partitioning:
    - Time-Based Partitioning:
        - Partition sales table by month
        - Retention: Keep 36 months online, archive older
        - Benefits: Query pruning, faster aggregations

    - Store-Based Partitioning (optional):
        - For very large multi-tenant deployments
        - Partition by region or store tier

  Materialized Views:
    - Daily Sales Summary:
        - Aggregates: SUM(revenue), COUNT(transactions), AVG(basket_size)
        - Refresh: Incremental (last 2 days) at 1 AM
        - Usage: Fast dashboard loading, trend charts

    - Monthly KPIs:
        - Pre-calculated metrics for executive dashboards
        - Refresh: Monthly on 1st day

    - ABC Classification Cache:
        - Store current ABC assignments
        - Refresh: Weekly or on-demand

  Query Optimization:
    - Use EXPLAIN ANALYZE to profile slow queries
    - Rewrite correlated subqueries as JOINs
    - Limit result sets (pagination)
    - Avoid SELECT * (specify columns)

  Connection Pooling:
    - Library: PgBouncer or pgpool
    - Pool size: 20-50 connections (tune based on load)
    - Timeout: 30 seconds idle, 5 minutes transaction
```

### 8.2 Caching Strategy

```yaml
CachingLayers:
  L1 - Browser Cache:
    - Static Assets:
        - JS/CSS bundles (1 year TTL with content hash)
        - Images, fonts (immutable)
        - Service Worker caching (offline support)

    - Application State:
        - localStorage for user preferences
        - sessionStorage for temporary filters

  L2 - CDN Cache:
    - Provider: CloudFlare or AWS CloudFront
    - Cached:
        - API responses for public/shared data
        - Static dashboard configurations
    - TTL: 5-60 minutes (based on data freshness)
    - Purge: On-demand when data updates

  L3 - Application Cache (Redis):
    - Hot Data:
        - Active user sessions
        - Recent query results (keyed by hash of query)
        - Computed aggregations (daily summaries)
    - TTL Strategy:
        - Real-time data: 5 minutes
        - Historical data: 1 hour
        - Aggregated metrics: 24 hours
    - Eviction: LRU (Least Recently Used)
    - Cluster: Redis Cluster for high availability

  L4 - Database Cache:
    - Query result cache (PostgreSQL shared buffers)
    - Prepared statement cache

  Cache Invalidation:
    - Strategies:
        - TTL-based (time-to-live expiration)
        - Event-based (new data loaded → purge related caches)
        - Manual purge (admin tool for emergencies)
    - Cache keys:
        - Include version identifier
        - Hash of query + filters
        - Example: "sales_ts_v2:store_123:2025-01:weekly"

  Cache Warming:
    - Pre-populate cache during off-peak hours
    - Most common queries (80/20 rule)
    - New day data (pre-compute at midnight)
```

### 8.3 Frontend Performance

```yaml
FrontendOptimization:
  Code Splitting:
    - Route-based splitting (Next.js automatic)
    - Component-level lazy loading
    - Dynamic imports for heavy libraries (D3.js)
    - Example: const HeavyChart = lazy(() => import('./HeavyChart'))

  Bundle Optimization:
    - Tree shaking (remove unused code)
    - Minification + compression (gzip/brotli)
    - Target bundle size: < 200KB initial, < 50KB per route
    - Analyze with webpack-bundle-analyzer

  Rendering Performance:
    - Virtual Scrolling:
        - Use @tanstack/react-virtual for long lists
        - Render only visible rows (windowing)

    - Memoization:
        - React.memo for expensive components
        - useMemo for computed values
        - useCallback for event handlers

    - Web Workers:
        - Offload heavy computations (data transformations)
        - Background processing for exports

    - RequestIdleCallback:
        - Low-priority tasks during browser idle time
        - Non-critical analytics, preloading

  Image Optimization:
    - Next.js Image component (automatic optimization)
    - Responsive images (srcset)
    - Lazy loading (below-the-fold)
    - WebP format with PNG fallback

  Data Fetching:
    - Prefetching:
        - Link hover → prefetch next page data
        - Predictive loading (likely next action)

    - Deduplication:
        - TanStack Query dedupes identical requests
        - Request coalescing (batch multiple requests)

    - Optimistic Updates:
        - Update UI immediately, reconcile later
        - Rollback on error

    - Stale-While-Revalidate:
        - Show cached data instantly
        - Fetch fresh data in background
        - Update when ready

  Performance Budgets:
    - Metrics:
        - First Contentful Paint (FCP): < 1.5s
        - Largest Contentful Paint (LCP): < 2.5s
        - Time to Interactive (TTI): < 3.5s
        - Cumulative Layout Shift (CLS): < 0.1
    - Monitoring: Lighthouse CI, Web Vitals tracking
```

### 8.4 Backend Performance

```yaml
BackendOptimization:
  API Design:
    - GraphQL:
        - Pros: Client specifies exact data needs, no over-fetching
        - DataLoader for batching/caching
        - Query complexity limits (prevent abuse)

    - REST:
        - Field filtering (?fields=id,name,revenue)
        - Pagination (cursor-based for consistency)
        - Bulk endpoints (reduce round-trips)

  Asynchronous Processing:
    - Message Queue (RabbitMQ or AWS SQS):
        - Long-running tasks (forecast model training)
        - Batch exports (large Excel files)
        - Email notifications

    - Background Jobs:
        - Cron-based (daily data refresh at 2 AM)
        - Event-driven (new data arrives → trigger processing)

  Horizontal Scaling:
    - Stateless API servers (scale out easily)
    - Load balancer (NGINX or AWS ALB)
    - Auto-scaling policies:
        - CPU > 70% → add instance
        - Request queue depth > 100 → add instance

  Database Scaling:
    - Read Replicas:
        - Separate read-heavy queries (analytics)
        - Write to primary, read from replicas
        - Lag monitoring (keep < 5 seconds)

    - Connection Pooling:
        - PgBouncer to manage connections
        - Limit per-service connections

    - Query Optimization:
        - Slow query log analysis (> 1 second)
        - Index recommendations (pg_stat_statements)

  Resource Monitoring:
    - Metrics Collection:
        - Application: Prometheus + Grafana
        - Infrastructure: CloudWatch or Datadog
        - Logs: ELK Stack (Elasticsearch, Logstash, Kibana)

    - Alerts:
        - API latency > 500ms (P95)
        - Error rate > 1%
        - Database CPU > 80%
        - Cache hit rate < 70%
```

### 8.5 Network Optimization

```yaml
NetworkOptimization:
  Compression:
    - HTTP Compression:
        - Brotli (better than gzip, 20-30% smaller)
        - Enable for text assets (JSON, HTML, CSS, JS)

    - API Response Compression:
        - Compress JSON responses > 1KB
        - Server-side compression middleware

  HTTP/2 & HTTP/3:
    - Multiplexing (parallel requests over single connection)
    - Server push (proactively send resources)
    - Header compression (HPACK)

  Protocol Optimization:
    - WebSocket for real-time updates:
        - Alert notifications
        - Live dashboard updates
        - Collaborative features

    - gRPC for service-to-service:
        - Binary protocol (smaller, faster)
        - Streaming support

  Content Delivery:
    - CDN Edge Locations:
        - Serve from nearest location
        - Reduce latency (50-200ms improvement)

    - Asset Optimization:
        - Inline critical CSS
        - Defer non-critical JS
        - Preload key resources

  Request Batching:
    - Combine multiple API calls:
        - GraphQL queries (single request, multiple resources)
        - Batch REST endpoints (/api/batch with JSON array)

    - Debouncing:
        - User typing → wait 300ms before search
        - Scroll events → throttle to 60fps
```

---

## 9. Security & Data Privacy

### 9.1 Authentication & Authorization

```yaml
SecurityArchitecture:
  Authentication:
    - Method: OAuth 2.0 + OpenID Connect
    - Provider: Auth0 or AWS Cognito
    - MFA: TOTP (Time-based One-Time Password)
    - Session: JWT (JSON Web Tokens)
        - Access token: 15-minute expiry
        - Refresh token: 7-day expiry
        - Secure cookies (httpOnly, sameSite, secure)

  Authorization:
    - Model: RBAC (Role-Based Access Control)
    - Roles:
        - Admin: Full system access
        - Regional Manager: All stores in region
        - Store Manager: Assigned stores only
        - Analyst: Read-only, all stores
        - Viewer: Read-only, limited modules

    - Permissions:
        - Module-level (can access ABC analysis?)
        - Store-level (can view Store X data?)
        - Action-level (can export data?)

    - Implementation:
        - Database: store_user_permissions table
        - Middleware: Check permissions before query execution
        - Frontend: Conditional rendering (hide unauthorized features)

  Data Isolation:
    - Row-Level Security (RLS):
        - PostgreSQL RLS policies
        - Automatically filter queries by user's store access
        - Example: user can only see WHERE store_id IN (user.assigned_stores)

    - API Gateway Filtering:
        - Validate store_id in request against user permissions
        - Reject unauthorized requests (403 Forbidden)
```

### 9.2 Data Protection

```yaml
DataProtection:
  Encryption:
    - In Transit:
        - TLS 1.3 (all API calls)
        - Certificate management (Let's Encrypt auto-renewal)

    - At Rest:
        - Database: AES-256 encryption
        - Backups: Encrypted storage (AWS S3 SSE)
        - Secrets: AWS Secrets Manager or HashiCorp Vault

  PII Handling:
    - Minimize Collection:
        - No customer PII in analytics (anonymized IDs only)
        - Store employee data only as needed

    - Data Masking:
        - Mask sensitive fields in logs
        - Tokenization for identifiers

  Audit Logging:
    - Log All Access:
        - User ID, timestamp, action, resource
        - IP address, user agent

    - Retention:
        - 90 days online, 7 years archived (compliance)
        - Immutable logs (append-only)

    - Monitoring:
        - Anomaly detection (unusual access patterns)
        - Alerts for suspicious activity
```

---

## 10. Deployment & Operations

### 10.1 Infrastructure Architecture

```yaml
Infrastructure:
  Cloud Provider: AWS (or Azure/GCP alternative)

  Architecture Pattern: Microservices on Kubernetes

  Components:
    - Frontend:
        - Next.js app deployed to Vercel or AWS Amplify
        - CDN: CloudFront

    - API Gateway:
        - AWS API Gateway or Kong
        - Rate limiting, throttling

    - Backend Services:
        - EKS (Elastic Kubernetes Service) cluster
        - Services:
            - TimeSeriesService (Node.js)
            - ABCAnalysisService (Python)
            - OrderOptimizationService (Python + OR-Tools)
            - ExternalFactorsService (Node.js)
            - ForecastService (Python + ML libraries)
            - AlertService (Node.js)
            - RecommendationService (Python)
        - Auto-scaling: HPA (Horizontal Pod Autoscaler)

    - Databases:
        - PostgreSQL on RDS (Multi-AZ)
        - InfluxDB on EC2 (time-series optimization)
        - Redis on ElastiCache (caching)

    - Storage:
        - S3 for data exports, backups
        - EBS for database volumes

    - Monitoring:
        - CloudWatch (metrics, logs)
        - Prometheus + Grafana (custom dashboards)
        - Sentry (error tracking)
```

### 10.2 CI/CD Pipeline

```yaml
CICD:
  Source Control: GitHub

  Pipeline Stages:
    1. Code Push:
        - Trigger: Git push to branch
        - Actions:
            - Lint (ESLint, Pylint)
            - Unit tests
            - Type checking (TypeScript)

    2. Build:
        - Docker image creation
        - Tag with commit SHA
        - Push to ECR (Elastic Container Registry)

    3. Test:
        - Integration tests (Playwright)
        - API contract tests (Pact)
        - Performance tests (k6)
        - Security scans (Snyk, OWASP)

    4. Staging Deployment:
        - Deploy to staging environment
        - Smoke tests
        - Manual QA approval

    5. Production Deployment:
        - Blue-green deployment (zero downtime)
        - Gradual rollout (10% → 50% → 100%)
        - Automated rollback on error rate spike

  Tools:
    - GitHub Actions (CI/CD orchestration)
    - Terraform (Infrastructure as Code)
    - Helm (Kubernetes package management)
    - ArgoCD (GitOps deployment)
```

### 10.3 Monitoring & Alerting

```yaml
Monitoring:
  Application Metrics:
    - API Latency (P50, P95, P99)
    - Error rates (by endpoint)
    - Request throughput (req/sec)
    - Active users (concurrent sessions)

  Infrastructure Metrics:
    - CPU, Memory utilization (by pod)
    - Network I/O
    - Disk usage, IOPS
    - Database connections, query latency

  Business Metrics:
    - Dashboard views (by module)
    - Alert acknowledgment rate
    - Export downloads
    - Feature adoption (% users using each module)

  Alerting:
    - PagerDuty integration (on-call rotation)
    - Severity-based escalation:
        - Critical: Immediate page
        - High: Alert within 15 minutes
        - Medium: Email digest
    - Auto-remediation (restart unhealthy pods)

  Logging:
    - Centralized: ELK Stack or AWS CloudWatch Logs
    - Structured logging (JSON format)
    - Log levels: ERROR, WARN, INFO, DEBUG
    - Correlation IDs (trace requests across services)
```

---

## 11. Technology Evaluation Matrix

| Component | Option 1 | Option 2 | Option 3 | Recommendation | Rationale |
|-----------|----------|----------|----------|----------------|-----------|
| **Frontend Framework** | React | Vue.js | Angular | **React** | Largest ecosystem, Next.js integration, team expertise |
| **Backend Language** | Node.js | Python | Go | **Node.js (API), Python (ML)** | Unified JS stack for API, Python for ML/optimization |
| **Database** | PostgreSQL | MySQL | MongoDB | **PostgreSQL** | Mature, JSON support, excellent analytics performance |
| **Time-Series DB** | InfluxDB | TimescaleDB | Prometheus | **InfluxDB** | Purpose-built, query language, retention policies |
| **Cache** | Redis | Memcached | Hazelcast | **Redis** | Rich data structures, pub/sub, persistence |
| **Charting Library** | Recharts | Chart.js | D3.js | **Recharts (primary), D3.js (advanced)** | React-native, declarative, D3 for custom viz |
| **State Management** | Zustand | Redux | Jotai | **Zustand** | Minimal boilerplate, TypeScript-first, small bundle |
| **API Pattern** | GraphQL | REST | gRPC | **GraphQL** | Flexible queries, avoid over-fetching, great DX |
| **Container Orchestration** | Kubernetes | Docker Swarm | ECS | **Kubernetes** | Industry standard, rich ecosystem, portability |
| **Cloud Provider** | AWS | Azure | GCP | **AWS** | Market leader, mature services, extensive documentation |

---

## 12. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-4)
- Set up infrastructure (AWS, Kubernetes cluster)
- Database schema design and migration
- Authentication/authorization system
- Basic dashboard shell with layout system
- Store filtering framework

**Deliverables:**
- Infrastructure as Code (Terraform)
- Database schema v1.0
- User authentication working
- Empty dashboard with navigation

### Phase 2: Core Modules (Weeks 5-12)
- Implement 時系列分析 module (Weeks 5-6)
- Implement ABC分析 module (Weeks 7-8)
- Implement 発注最適化 module (Weeks 9-10)
- Implement 外部要因 module (Week 11)
- Implement 需要予測 module (Week 12)

**Deliverables:**
- All 5 modules functional
- 15+ chart types implemented
- Multi-granularity time analysis working
- Basic caching layer

### Phase 3: Intelligence Layer (Weeks 13-16)
- Alert engine implementation (Week 13)
- Recommendation engine (Weeks 14-15)
- Cross-store benchmarking (Week 16)

**Deliverables:**
- Real-time alert system
- ML-powered recommendations
- Peer group analysis

### Phase 4: Optimization & Polish (Weeks 17-20)
- Performance optimization (Week 17)
- Advanced caching strategies (Week 18)
- Mobile responsiveness (Week 19)
- User testing and refinement (Week 20)

**Deliverables:**
- Sub-second response times
- 90+ Lighthouse score
- Mobile-optimized UI
- User acceptance sign-off

### Phase 5: Launch Preparation (Weeks 21-24)
- Security audit and penetration testing (Week 21)
- Load testing and scaling validation (Week 22)
- Documentation and training materials (Week 23)
- Staged rollout to pilot stores (Week 24)

**Deliverables:**
- Security certification
- Performance benchmarks
- Training videos and guides
- Production deployment

---

## 13. Risk Assessment & Mitigation

| Risk | Probability | Impact | Mitigation Strategy |
|------|-------------|--------|---------------------|
| **Data quality issues** | High | High | Implement data validation pipelines, anomaly detection, manual review process |
| **Performance degradation at scale** | Medium | High | Load testing, caching strategy, horizontal scaling, database optimization |
| **User adoption resistance** | Medium | Medium | Phased rollout, comprehensive training, collect feedback, iterate on UX |
| **ML model accuracy drift** | Medium | Medium | Continuous monitoring, automated retraining, human-in-the-loop validation |
| **Integration failures (POS, ERP)** | Low | High | Robust error handling, fallback mechanisms, detailed logging, vendor SLAs |
| **Security breach** | Low | Critical | Security audits, penetration testing, encryption, access controls, incident response plan |
| **Vendor/cloud outage** | Low | High | Multi-AZ deployment, backup systems, disaster recovery plan, SLA monitoring |
| **Scope creep** | High | Medium | Clear requirements, change control process, stakeholder alignment, phased delivery |

---

## 14. Success Metrics

### Technical KPIs
- **Availability**: 99.9% uptime
- **Performance**: P95 API response < 500ms
- **Scalability**: Support 1000+ concurrent users
- **Data Freshness**: < 5 minutes lag for real-time data

### Business KPIs
- **User Adoption**: 80% of store managers log in weekly
- **Feature Usage**: All 5 modules used by 60%+ of users
- **Actionability**: 70% of recommendations acknowledged
- **ROI**: 10% inventory cost reduction within 6 months

### User Satisfaction
- **NPS Score**: > 50 (promoters - detractors)
- **CSAT**: > 4.0/5.0 average rating
- **Support Tickets**: < 5% of users submit tickets monthly

---

## 15. Appendices

### A. Glossary

- **ABC Analysis**: Inventory categorization by revenue contribution (Pareto principle)
- **DOI**: Days of Inventory (average days to sell current stock)
- **EOQ**: Economic Order Quantity (optimal order size minimizing total cost)
- **MAPE**: Mean Absolute Percentage Error (forecast accuracy metric)
- **ROP**: Reorder Point (inventory level triggering new order)
- **SKU**: Stock Keeping Unit (unique product identifier)
- **YoY**: Year-over-Year (comparison to same period last year)

### B. API Endpoint Examples

```
GET  /api/stores                          # List all accessible stores
GET  /api/stores/{id}                     # Get store details
GET  /api/analytics/timeseries            # Time series analysis
     ?store_ids=1,2,3
     &start_date=2025-01-01
     &end_date=2025-10-08
     &granularity=weekly
     &metrics=revenue,transactions

GET  /api/analytics/abc                   # ABC analysis
     ?store_id=1
     &period=monthly
     &month=2025-09

GET  /api/analytics/forecast              # Demand forecast
     ?store_id=1
     &product_ids=100,200
     &horizon=4weeks

POST /api/orders/optimize                 # Order optimization
     Body: { store_id, constraints, objectives }

GET  /api/alerts                          # List active alerts
     ?severity=high
     &acknowledged=false

GET  /api/recommendations                 # Get recommendations
     ?store_id=1
     &module=inventory
     &top_n=5
```

### C. Database Schema (Simplified)

```sql
-- Core tables
CREATE TABLE stores (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    region_id INT,
    district_id INT,
    format VARCHAR(50),
    size_sqft INT,
    opened_date DATE
);

CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    sku VARCHAR(100) UNIQUE,
    name VARCHAR(255),
    category_id INT,
    unit_cost DECIMAL(10,2),
    shelf_life_days INT
);

CREATE TABLE sales (
    id BIGSERIAL PRIMARY KEY,
    store_id INT REFERENCES stores(id),
    product_id INT REFERENCES products(id),
    date_key INT,  -- YYYYMMDD for partitioning
    transaction_id VARCHAR(100),
    quantity INT,
    revenue DECIMAL(10,2),
    discount DECIMAL(10,2)
) PARTITION BY RANGE (date_key);

CREATE TABLE inventory (
    id BIGSERIAL PRIMARY KEY,
    store_id INT REFERENCES stores(id),
    product_id INT REFERENCES products(id),
    snapshot_date DATE,
    quantity_on_hand INT,
    quantity_on_order INT,
    last_updated TIMESTAMP
);

-- Analysis cache tables
CREATE TABLE abc_classifications (
    id SERIAL PRIMARY KEY,
    store_id INT,
    product_id INT,
    period_month INT,  -- YYYYMM
    category CHAR(1),  -- A, B, or C
    revenue_contribution DECIMAL(5,2),
    cumulative_contribution DECIMAL(5,2),
    calculated_at TIMESTAMP
);

CREATE TABLE forecasts (
    id SERIAL PRIMARY KEY,
    store_id INT,
    product_id INT,
    forecast_date DATE,
    predicted_quantity DECIMAL(10,2),
    confidence_lower DECIMAL(10,2),
    confidence_upper DECIMAL(10,2),
    model_version VARCHAR(50),
    created_at TIMESTAMP
);
```

---

## Conclusion

This architecture provides a comprehensive, scalable foundation for a store manager dashboard with:

1. **Modularity**: Each analysis type is independently deployable and maintainable
2. **Performance**: Sub-second response times through intelligent caching and optimization
3. **Scalability**: Horizontal scaling supports growth from 10 to 1000+ stores
4. **Intelligence**: ML-powered alerts and recommendations drive actionable insights
5. **User Experience**: Interactive visualizations with multi-granularity analysis
6. **Reliability**: 99.9% uptime with graceful degradation and disaster recovery

The phased implementation approach minimizes risk while delivering value incrementally. Success metrics ensure alignment with business objectives, and the technology stack leverages industry-standard tools for long-term maintainability.

**Next Steps:**
1. Stakeholder review and approval of architecture
2. Detailed technical design for Phase 1 components
3. Team formation and sprint planning
4. Infrastructure provisioning and initial development

---

**Document Version**: 1.0
**Last Updated**: 2025-10-08
**Author**: System Architecture Designer
**Review Status**: Pending Stakeholder Approval
