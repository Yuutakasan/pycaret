# Dashboard Architecture - Visual Diagrams

## Component Interaction Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Store Manager Dashboard                              │
│                        Component Interaction Flow                            │
└─────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────┐
│  PRESENTATION LAYER                                                            │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │  Dashboard   │  │   Chart      │  │    Alert     │  │   Export     │    │
│  │   Shell      │  │  Components  │  │    Panel     │  │   Module     │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
│         │                  │                  │                  │             │
│         └──────────────────┼──────────────────┼──────────────────┘            │
│                            │                  │                                │
└────────────────────────────┼──────────────────┼────────────────────────────────┘
                             │                  │
                             ▼                  ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│  STATE MANAGEMENT LAYER                                                        │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  ┌─────────────────────┐         ┌─────────────────────┐                    │
│  │  Zustand Store      │         │   TanStack Query    │                    │
│  │  - UI State         │◄────────┤   - Server State    │                    │
│  │  - User Preferences │         │   - Cache Manager   │                    │
│  └─────────────────────┘         └──────────┬──────────┘                    │
│                                              │                                │
└──────────────────────────────────────────────┼────────────────────────────────┘
                                               │
                                               ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│  API GATEWAY LAYER                                                             │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  ┌──────────────────────────────────────────────────────────────────┐        │
│  │  GraphQL API Gateway                                              │        │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐│        │
│  │  │   Auth     │  │  Rate      │  │  Request   │  │  Response  ││        │
│  │  │ Middleware │→ │  Limiter   │→ │  Validator │→ │   Cache    ││        │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘│        │
│  └──────────────────────────────────────────────────────────────────┘        │
│                            │                                                   │
│         ┌──────────────────┼──────────────────┬──────────────────┐           │
│         │                  │                  │                  │            │
└─────────┼──────────────────┼──────────────────┼──────────────────┼────────────┘
          ▼                  ▼                  ▼                  ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│  BUSINESS LOGIC LAYER - Analysis Services                                     │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐       │
│  │ TimeSeries   │ │     ABC      │ │    Order     │ │  External    │       │
│  │   Service    │ │   Service    │ │ Optimization │ │   Factors    │       │
│  │              │ │              │ │   Service    │ │   Service    │       │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘       │
│         │                │                │                │                  │
│         └────────────────┼────────────────┼────────────────┘                 │
│                          │                │                                   │
│                          ▼                ▼                                   │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │            Demand Forecasting Service                             │       │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐│       │
│  │  │  Prophet   │  │    LSTM    │  │  XGBoost   │  │  Ensemble  ││       │
│  │  │   Model    │  │   Model    │  │   Model    │  │  Combiner  ││       │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘│       │
│  └──────────────────────────────────────────────────────────────────┘       │
│                                                                                │
└────────────────────────────────────┬───────────────────────────────────────────┘
                                     │
                                     ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│  INTELLIGENCE LAYER                                                            │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  ┌─────────────────────────────────────────────────────────────┐             │
│  │  Alert Engine                                                │             │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌──────────┐│             │
│  │  │  Event    │→ │ Classify  │→ │  Filter   │→ │  Notify  ││             │
│  │  │ Detection │  │ & Score   │  │   & Dedupe│  │ & Route  ││             │
│  │  └───────────┘  └───────────┘  └───────────┘  └──────────┘│             │
│  └─────────────────────────────────────────────────────────────┘             │
│                                                                                │
│  ┌─────────────────────────────────────────────────────────────┐             │
│  │  Recommendation Engine                                       │             │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌──────────┐│             │
│  │  │   Rules   │  │    ML     │  │  Hybrid   │  │  Priority││             │
│  │  │   Engine  │  │  Models   │  │ Combiner  │  │  Ranker  ││             │
│  │  └───────────┘  └───────────┘  └───────────┘  └──────────┘│             │
│  └─────────────────────────────────────────────────────────────┘             │
│                                                                                │
└────────────────────────────────────┬───────────────────────────────────────────┘
                                     │
                                     ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│  DATA ACCESS LAYER                                                             │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  ┌──────────────────────────────────────────────────────────────────┐        │
│  │  Data Access Service (ORM / Query Builder)                        │        │
│  │  - Store filtering logic                                          │        │
│  │  - Row-level security enforcement                                │        │
│  │  - Query optimization & caching                                  │        │
│  └────────────┬─────────────────────────┬───────────────────────────┘        │
│               │                         │                                     │
└───────────────┼─────────────────────────┼─────────────────────────────────────┘
                │                         │
                ▼                         ▼
┌─────────────────────────┐   ┌─────────────────────────┐   ┌──────────────┐
│  PostgreSQL Database    │   │   Redis Cache Layer     │   │  InfluxDB    │
│  - Transactional data   │   │   - Query results       │   │  - Time-     │
│  - Aggregated metrics   │   │   - Session data        │   │    series    │
│  - User data            │   │   - Hot computations    │   │    metrics   │
│  - Materialized views   │   │   - Pub/sub messaging   │   │              │
└─────────────────────────┘   └─────────────────────────┘   └──────────────┘

┌───────────────────────────────────────────────────────────────────────────────┐
│  EXTERNAL INTEGRATIONS                                                         │
├───────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │  POS System  │  │  Inventory   │  │  Weather API │  │  Calendar    │    │
│  │   (Sales)    │  │   Database   │  │  (External)  │  │  API (Events)│    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Diagram - Time Series Analysis

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Time Series Analysis - Data Flow                                       │
└─────────────────────────────────────────────────────────────────────────┘

User Request (UI)
    │
    │ 1. User selects:
    │    - Stores: [Store A, Store B, Store C]
    │    - Date Range: 2025-01-01 to 2025-10-08
    │    - Granularity: Weekly
    │    - Metrics: [Revenue, Transactions]
    │
    ▼
┌─────────────────────────────────────────┐
│  Frontend State Management               │
│  - Validate selections                  │
│  - Build query parameters               │
│  - Check local cache                    │
└─────────────────┬───────────────────────┘
                  │
                  │ 2. API Request:
                  │    GET /api/analytics/timeseries
                  │    ?store_ids=1,2,3
                  │    &start_date=2025-01-01
                  │    &end_date=2025-10-08
                  │    &granularity=weekly
                  │    &metrics=revenue,transactions
                  │
                  ▼
┌─────────────────────────────────────────┐
│  API Gateway                            │
│  - Authenticate user (JWT)              │
│  - Check authorization (can view stores)│
│  - Rate limit check                     │
│  - Log request                          │
└─────────────────┬───────────────────────┘
                  │
                  │ 3. Authorized request
                  │
                  ▼
┌─────────────────────────────────────────┐
│  Time Series Service                    │
│                                         │
│  Step 1: Parse & Validate               │
│    - Validate date range                │
│    - Check store access                 │
│    - Normalize granularity              │
│                                         │
│  Step 2: Check Cache                    │
│    - Build cache key                    │
│    - Query Redis for cached result      │
│    - If HIT → return cached data (10ms) │
│    - If MISS → proceed to database      │
│                                         │
│  Step 3: Query Builder                  │
│    - Determine optimal query path       │
│    - Use materialized view if available │
│    - Build SQL with filters:            │
│      * WHERE store_id IN (1,2,3)        │
│      * AND date >= '2025-01-01'         │
│      * GROUP BY week_key, store_id      │
│    - Apply indexes for performance      │
│                                         │
└─────────────────┬───────────────────────┘
                  │
                  │ 4. SQL Query
                  │
                  ▼
┌─────────────────────────────────────────┐
│  PostgreSQL Database                    │
│                                         │
│  Query Execution:                       │
│    - Partition pruning (by date range)  │
│    - Index scan on (store_id, date_key) │
│    - Aggregate by week_key              │
│    - Return ~120 rows (3 stores × 40wks)│
│                                         │
│  Execution time: ~150ms                 │
└─────────────────┬───────────────────────┘
                  │
                  │ 5. Raw data (JSON)
                  │
                  ▼
┌─────────────────────────────────────────┐
│  Time Series Service (Processing)       │
│                                         │
│  Step 4: Data Transformation            │
│    - Pivot data by store                │
│    - Calculate derived metrics:         │
│      * Week-over-week % change          │
│      * Moving averages (4-week, 13-week)│
│      * Trend line (linear regression)   │
│                                         │
│  Step 5: Trend Analysis                 │
│    - Apply Prophet model for seasonality│
│    - Detect anomalies (> 2σ deviation)  │
│    - Identify trend reversals           │
│                                         │
│  Step 6: Result Formatting              │
│    - Structure for chart library        │
│    - Add metadata (calculations, alerts)│
│    - Compress large responses           │
│                                         │
│  Step 7: Cache Result                   │
│    - Store in Redis (TTL: 1 hour)       │
│    - Tag for invalidation               │
│                                         │
│  Processing time: ~200ms                │
└─────────────────┬───────────────────────┘
                  │
                  │ 6. Formatted response (JSON)
                  │
                  ▼
┌─────────────────────────────────────────┐
│  API Gateway (Response)                 │
│  - Compress with Brotli                 │
│  - Set cache headers (CDN)              │
│  - Log response time                    │
└─────────────────┬───────────────────────┘
                  │
                  │ 7. HTTP Response
                  │    Status: 200 OK
                  │    Content-Type: application/json
                  │    X-Response-Time: 350ms
                  │
                  ▼
┌─────────────────────────────────────────┐
│  Frontend (Visualization)               │
│                                         │
│  Step 1: Parse Response                 │
│    - Validate data structure            │
│    - Extract chart data                 │
│    - Extract alerts/anomalies           │
│                                         │
│  Step 2: State Update                   │
│    - Update Zustand store               │
│    - Cache in TanStack Query            │
│    - Trigger re-render                  │
│                                         │
│  Step 3: Chart Rendering                │
│    - Recharts line chart component      │
│    - Multiple series (one per store)    │
│    - Annotations (trend lines, alerts)  │
│    - Interactive tooltips               │
│                                         │
│  Step 4: Alert Display                  │
│    - Show detected anomalies            │
│    - Highlight trend reversals          │
│    - Suggest next actions               │
│                                         │
│  Render time: ~100ms                    │
└─────────────────────────────────────────┘
                  │
                  │ 8. User sees chart
                  │    Total time: 450ms
                  │    (350ms backend + 100ms frontend)
                  │
                  ▼
              [Dashboard Display]
    ┌────────────────────────────────────────┐
    │  Weekly Sales Trends                   │
    │  ┌──────────────────────────────────┐ │
    │  │        📈 Line Chart              │ │
    │  │                                   │ │
    │  │  Revenue ($)                      │ │
    │  │    ▲                               │ │
    │  │    │      ⚠️ Anomaly               │ │
    │  │    │    /\    detected            │ │
    │  │    │   /  \  /                    │ │
    │  │    │  /    \/  [Store A]          │ │
    │  │    │ /      \  [Store B]          │ │
    │  │    │/        \ [Store C]          │ │
    │  │    └──────────────────────► Time  │ │
    │  │     Jan  Mar  May  Jul  Sep      │ │
    │  └──────────────────────────────────┘ │
    │                                        │
    │  🔔 Alerts:                            │
    │  • Store B: Sales declined 22% WoW    │
    │  • Store A: Unusual spike on Sep 15   │
    │                                        │
    │  💡 Recommendations:                   │
    │  • Investigate Store B drop (high)    │
    │  • Review Store A promo success (med) │
    └────────────────────────────────────────┘
```

---

## System Architecture - Deployment View

```
┌─────────────────────────────────────────────────────────────────────────┐
│  AWS Cloud Infrastructure - Production Environment                      │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  CloudFront CDN (Global Edge Locations)                                 │
│  - Static assets (JS, CSS, images)                                     │
│  - Cached API responses (public data)                                  │
│  - DDoS protection                                                      │
└───────────────────────────┬─────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Route 53 (DNS)                                                         │
│  - Health checks                                                        │
│  - Failover routing                                                     │
└───────────────────────────┬─────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Region:   │    │   Region:   │    │   Region:   │
│  us-east-1  │    │  eu-west-1  │    │  ap-south-1 │
│  (Primary)  │    │  (DR Site)  │    │  (Optional) │
└─────────────┘    └─────────────┘    └─────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  us-east-1 Region (Primary)                                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │  VPC: 10.0.0.0/16                                               │   │
│  │                                                                  │   │
│  │  ┌─────────────────────────────────────────────────────────┐  │   │
│  │  │  Public Subnet: 10.0.1.0/24 (AZ-1a)                     │  │   │
│  │  │  ┌──────────────┐  ┌──────────────┐                     │  │   │
│  │  │  │     ALB      │  │   NAT GW     │                     │  │   │
│  │  │  │ (Load Balancer)│ │              │                     │  │   │
│  │  │  └──────┬───────┘  └──────────────┘                     │  │   │
│  │  └─────────┼──────────────────────────────────────────────┘  │   │
│  │            │                                                   │   │
│  │  ┌─────────┼──────────────────────────────────────────────┐  │   │
│  │  │  Private Subnet: 10.0.10.0/24 (AZ-1a)                  │  │   │
│  │  │         │                                                │  │   │
│  │  │  ┌──────▼────────────────────────────────────────────┐ │  │   │
│  │  │  │  EKS Cluster: dashboard-prod                      │ │  │   │
│  │  │  │                                                    │ │  │   │
│  │  │  │  ┌────────────────────────────────────────────┐  │ │  │   │
│  │  │  │  │  Node Group: api-servers (3 nodes)         │  │ │  │   │
│  │  │  │  │  - Instance: t3.large                       │  │ │  │   │
│  │  │  │  │  - Auto-scaling: 3-10 nodes                │  │ │  │   │
│  │  │  │  │                                             │  │ │  │   │
│  │  │  │  │  Pods:                                      │  │ │  │   │
│  │  │  │  │  ┌─────────────────┐  ┌─────────────────┐ │  │ │  │   │
│  │  │  │  │  │  API Gateway    │  │  Auth Service   │ │  │ │  │   │
│  │  │  │  │  │  (GraphQL)      │  │                 │ │  │ │  │   │
│  │  │  │  │  │  Replicas: 3    │  │  Replicas: 2    │ │  │ │  │   │
│  │  │  │  │  └─────────────────┘  └─────────────────┘ │  │ │  │   │
│  │  │  │  └────────────────────────────────────────────┘  │ │  │   │
│  │  │  │                                                    │ │  │   │
│  │  │  │  ┌────────────────────────────────────────────┐  │ │  │   │
│  │  │  │  │  Node Group: analytics (5 nodes)           │  │ │  │   │
│  │  │  │  │  - Instance: c5.xlarge (compute-optimized) │  │ │  │   │
│  │  │  │  │  - Auto-scaling: 3-15 nodes                │  │ │  │   │
│  │  │  │  │                                             │  │ │  │   │
│  │  │  │  │  Pods:                                      │  │ │  │   │
│  │  │  │  │  ┌─────────────┐  ┌─────────────┐         │  │ │  │   │
│  │  │  │  │  │ TimeSeries  │  │    ABC      │         │  │ │  │   │
│  │  │  │  │  │  Service    │  │  Service    │  ...    │  │ │  │   │
│  │  │  │  │  │ Replicas: 3 │  │ Replicas: 2 │         │  │ │  │   │
│  │  │  │  │  └─────────────┘  └─────────────┘         │  │ │  │   │
│  │  │  │  └────────────────────────────────────────────┘  │ │  │   │
│  │  │  │                                                    │ │  │   │
│  │  │  │  ┌────────────────────────────────────────────┐  │ │  │   │
│  │  │  │  │  Node Group: ml-workers (2 nodes)          │  │ │  │   │
│  │  │  │  │  - Instance: p3.2xlarge (GPU, optional)    │  │ │  │   │
│  │  │  │  │  - Auto-scaling: 1-5 nodes                 │  │ │  │   │
│  │  │  │  │                                             │  │ │  │   │
│  │  │  │  │  Pods:                                      │  │ │  │   │
│  │  │  │  │  ┌─────────────┐  ┌─────────────┐         │  │ │  │   │
│  │  │  │  │  │  Forecast   │  │ Recommend   │         │  │ │  │   │
│  │  │  │  │  │  Service    │  │  Engine     │         │  │ │  │   │
│  │  │  │  │  │ Replicas: 2 │  │ Replicas: 2 │         │  │ │  │   │
│  │  │  │  │  └─────────────┘  └─────────────┘         │  │ │  │   │
│  │  │  │  └────────────────────────────────────────────┘  │ │  │   │
│  │  │  └────────────────────────────────────────────────────┘ │   │
│  │  └──────────────────────────────────────────────────────────┘   │
│  │                                                                   │
│  │  ┌─────────────────────────────────────────────────────────┐   │
│  │  │  Private Subnet: 10.0.20.0/24 (AZ-1a) - Data Tier       │   │
│  │  │                                                           │   │
│  │  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │  │  RDS PostgreSQL (Multi-AZ)                       │   │   │
│  │  │  │  - Instance: db.r5.2xlarge                       │   │   │
│  │  │  │  - Storage: 1TB SSD (auto-scaling)               │   │   │
│  │  │  │  - Read Replicas: 2 (for analytics queries)      │   │   │
│  │  │  │  - Automated backups: 7 days retention           │   │   │
│  │  │  └─────────────────────────────────────────────────┘   │   │
│  │  │                                                           │   │
│  │  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │  │  ElastiCache Redis (Cluster Mode)                │   │   │
│  │  │  │  - Node: cache.r5.large                          │   │   │
│  │  │  │  - Shards: 3 (for horizontal scaling)            │   │   │
│  │  │  │  - Replicas: 1 per shard (6 nodes total)         │   │   │
│  │  │  │  - Automatic failover enabled                    │   │   │
│  │  │  └─────────────────────────────────────────────────┘   │   │
│  │  │                                                           │   │
│  │  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │  │  EC2: InfluxDB (Time-Series)                     │   │   │
│  │  │  │  - Instance: r5.xlarge                           │   │   │
│  │  │  │  - EBS: 500GB SSD                                │   │   │
│  │  │  │  - Backup to S3 daily                            │   │   │
│  │  │  └─────────────────────────────────────────────────┘   │   │
│  │  └──────────────────────────────────────────────────────────┘   │
│  │                                                                   │
│  │  ┌─────────────────────────────────────────────────────────┐   │
│  │  │  Public Subnet: 10.0.2.0/24 (AZ-1b) - High Availability│   │
│  │  │  [Mirror of AZ-1a for redundancy]                       │   │
│  │  └─────────────────────────────────────────────────────────┘   │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │  Managed Services (Regional)                                      │ │
│  │                                                                    │ │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐       │ │
│  │  │      S3       │  │   CloudWatch  │  │  Secrets Mgr  │       │ │
│  │  │  (Data Lake)  │  │  (Monitoring) │  │  (API Keys)   │       │ │
│  │  └───────────────┘  └───────────────┘  └───────────────┘       │ │
│  │                                                                    │ │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐       │ │
│  │  │      SQS      │  │      SNS      │  │   Lambda      │       │ │
│  │  │ (Job Queue)   │  │ (Notifications)│  │ (Serverless)  │       │ │
│  │  └───────────────┘  └───────────────┘  └───────────────┘       │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│  Frontend Hosting (Separate)                                             │
├──────────────────────────────────────────────────────────────────────────┤
│  Vercel / AWS Amplify                                                    │
│  - Next.js app (SSR + static generation)                                │
│  - Automatic deployments from GitHub                                    │
│  - Edge functions for API routes                                        │
│  - Preview environments for pull requests                               │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│  Monitoring & Observability                                              │
├──────────────────────────────────────────────────────────────────────────┤
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐              │
│  │   Prometheus  │  │    Grafana    │  │     Sentry    │              │
│  │  (Metrics)    │  │  (Dashboards) │  │ (Error Track) │              │
│  └───────────────┘  └───────────────┘  └───────────────┘              │
│                                                                           │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐              │
│  │   ELK Stack   │  │   PagerDuty   │  │   Datadog     │              │
│  │    (Logs)     │  │    (Alerts)   │  │    (APM)      │              │
│  └───────────────┘  └───────────────┘  └───────────────┘              │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Store Filtering & Comparison Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Multi-Store Filtering & Comparison System                              │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  FRONTEND LAYER - Filter UI Component                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │  Filter Panel                                                   │   │
│  │                                                                  │   │
│  │  🏢 Store Selection:                                            │   │
│  │  ┌───────────────────────────────────────────────────────┐    │   │
│  │  │  [Search stores...]                          🔍        │    │   │
│  │  │                                                         │    │   │
│  │  │  ☑ All Stores (123)                                    │    │   │
│  │  │  ▼ Region: North America (45)                          │    │   │
│  │  │    ▼ District: Northeast (15)                          │    │   │
│  │  │      ☑ Store 001 - New York Flagship                   │    │   │
│  │  │      ☑ Store 002 - Boston Downtown                     │    │   │
│  │  │      ☐ Store 003 - Philadelphia Center                 │    │   │
│  │  │    ▼ District: Midwest (12)                            │    │   │
│  │  │      ☐ Store 020 - Chicago Loop                        │    │   │
│  │  │  ▼ Region: Europe (38)                                 │    │   │
│  │  │    ...                                                  │    │   │
│  │  └───────────────────────────────────────────────────────┘    │   │
│  │                                                                  │   │
│  │  📁 Quick Filters:                                              │   │
│  │  [Top 10 Performers] [My Stores] [New Stores] [Flagged]       │   │
│  │                                                                  │   │
│  │  🏷️ Tags:                                                       │   │
│  │  [Urban ×] [Large Format ×] [High Traffic ×]                  │   │
│  │                                                                  │   │
│  │  📊 Comparison Mode:                                            │   │
│  │  ◉ Side-by-side   ○ Overlay   ○ Benchmark vs Average         │   │
│  │                                                                  │   │
│  │  Selected: 2 stores         [Apply] [Reset] [Save Preset]     │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
                                   │
                                   │ Filter changes trigger:
                                   │ - State update (Zustand)
                                   │ - URL param update (shareable)
                                   │ - API re-fetch (TanStack Query)
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STATE MANAGEMENT - Filter State                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  filterStore (Zustand):                                                 │
│  {                                                                       │
│    selectedStores: [1, 2],                                              │
│    hierarchyExpanded: {                                                 │
│      'region-north-america': true,                                      │
│      'district-northeast': true                                         │
│    },                                                                    │
│    tags: ['urban', 'large-format', 'high-traffic'],                     │
│    comparisonMode: 'side-by-side',                                      │
│    dateRange: { start: '2025-01-01', end: '2025-10-08' },              │
│    granularity: 'weekly'                                                │
│  }                                                                       │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────┐        │
│  │  Computed Queries (React Query)                             │        │
│  │                                                              │        │
│  │  useStoreData(filterStore.selectedStores) {                │        │
│  │    queryKey: ['stores', [1, 2], '2025-01-01', ...]         │        │
│  │    queryFn: () => fetchStoreData(...)                      │        │
│  │    staleTime: 5 minutes                                     │        │
│  │    cacheTime: 1 hour                                        │        │
│  │  }                                                           │        │
│  └────────────────────────────────────────────────────────────┘        │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
                                   │
                                   │ Query execution
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  BACKEND LAYER - Query Processing                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  StoreFilterService:                                                    │
│                                                                          │
│  processStoreFilter(selectedStores, userPermissions) {                 │
│    // Step 1: Authorization check                                      │
│    allowedStores = intersect(selectedStores, userPermissions.stores)   │
│                                                                          │
│    // Step 2: Hierarchy resolution                                     │
│    if (selectedStores.includes('region-*')) {                          │
│      expandedStores = getStoresInRegion(regionId)                     │
│    }                                                                     │
│                                                                          │
│    // Step 3: Tag filtering                                            │
│    if (tags.length > 0) {                                              │
│      filteredStores = filterByTags(allowedStores, tags)               │
│    }                                                                     │
│                                                                          │
│    // Step 4: Build SQL filter clause                                  │
│    return {                                                             │
│      storeIds: [1, 2],                                                 │
│      sqlClause: "store_id IN (1, 2)"                                   │
│    }                                                                     │
│  }                                                                       │
│                                                                          │
│  ComparisonService:                                                     │
│                                                                          │
│  buildComparisonQuery(stores, mode) {                                  │
│    switch(mode) {                                                       │
│      case 'side-by-side':                                              │
│        // Fetch separate data for each store                           │
│        return stores.map(id => fetchStoreData(id))                    │
│                                                                          │
│      case 'overlay':                                                   │
│        // Fetch combined data, group by store                          │
│        SELECT store_id, date_key, SUM(revenue) as revenue              │
│        FROM sales                                                       │
│        WHERE store_id IN (1, 2)                                        │
│        GROUP BY store_id, date_key                                     │
│                                                                          │
│      case 'benchmark':                                                 │
│        // Fetch selected stores + peer group average                   │
│        WITH peer_avg AS (                                              │
│          SELECT date_key, AVG(revenue) as avg_revenue                 │
│          FROM sales                                                     │
│          WHERE store_id IN (SELECT peer_stores(1))                    │
│          GROUP BY date_key                                             │
│        )                                                                │
│        SELECT s.store_id, s.date_key, s.revenue,                      │
│               p.avg_revenue as benchmark                               │
│        FROM sales s                                                     │
│        JOIN peer_avg p ON s.date_key = p.date_key                     │
│        WHERE s.store_id IN (1, 2)                                      │
│    }                                                                     │
│  }                                                                       │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
                                   │
                                   │ Query results
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  DATA LAYER - Store Hierarchy Schema                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  stores table:                                                          │
│  ┌────┬───────────┬───────────┬────────────┬───────────┬──────────┐  │
│  │ id │   name    │ region_id │ district_id│  format   │ size_tier│  │
│  ├────┼───────────┼───────────┼────────────┼───────────┼──────────┤  │
│  │ 1  │ NY Flag   │    10     │    101     │ Flagship  │  Large   │  │
│  │ 2  │ Boston DT │    10     │    101     │ Urban     │  Medium  │  │
│  │ 3  │ Philly Ctr│    10     │    101     │ Urban     │  Medium  │  │
│  └────┴───────────┴───────────┴────────────┴───────────┴──────────┘  │
│                                                                          │
│  store_hierarchies table (materialized path):                          │
│  ┌─────────┬───────────┬─────────┬────────────────────────────┐      │
│  │ store_id│ parent_id │  level  │     path                    │      │
│  ├─────────┼───────────┼─────────┼────────────────────────────┤      │
│  │    1    │    101    │    3    │ /org/region-10/dist-101/1  │      │
│  │   101   │     10    │    2    │ /org/region-10/dist-101    │      │
│  │    10   │    org    │    1    │ /org/region-10             │      │
│  └─────────┴───────────┴─────────┴────────────────────────────┘      │
│                                                                          │
│  store_tags table (many-to-many):                                      │
│  ┌─────────┬──────────────┐                                            │
│  │ store_id│   tag_name   │                                            │
│  ├─────────┼──────────────┤                                            │
│  │    1    │    urban     │                                            │
│  │    1    │ high-traffic │                                            │
│  │    1    │ large-format │                                            │
│  └─────────┴──────────────┘                                            │
│                                                                          │
│  Indexes for performance:                                               │
│  - CREATE INDEX idx_stores_region ON stores(region_id)                 │
│  - CREATE INDEX idx_stores_district ON stores(district_id)             │
│  - CREATE INDEX idx_hierarchy_path ON store_hierarchies                │
│      USING GIN (path gin_trgm_ops)  -- for path queries                │
│  - CREATE INDEX idx_tags_store ON store_tags(store_id)                 │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
                                   │
                                   │ Filtered data
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  VISUALIZATION LAYER - Comparison Display                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Side-by-Side Mode:                                                     │
│  ┌──────────────────────────┬──────────────────────────┐              │
│  │  Store 001 - NY Flagship │  Store 002 - Boston DT   │              │
│  ├──────────────────────────┼──────────────────────────┤              │
│  │  Revenue: $1.2M          │  Revenue: $850K          │              │
│  │  ▲ 15% YoY               │  ▲ 8% YoY                │              │
│  │                           │                          │              │
│  │  📈 [Line chart]         │  📈 [Line chart]         │              │
│  │                           │                          │              │
│  │  Top Products:            │  Top Products:           │              │
│  │  1. Product A ($120K)     │  1. Product B ($95K)     │              │
│  │  2. Product B ($110K)     │  2. Product A ($88K)     │              │
│  └──────────────────────────┴──────────────────────────┘              │
│                                                                          │
│  Overlay Mode:                                                          │
│  ┌──────────────────────────────────────────────────────┐              │
│  │  Comparative Revenue Trends                          │              │
│  │  ┌────────────────────────────────────────────────┐ │              │
│  │  │  Revenue ($)                                    │ │              │
│  │  │    ▲                                            │ │              │
│  │  │    │                                            │ │              │
│  │  │    │      ━━━━  Store 001 (NY)                 │ │              │
│  │  │    │     /                                      │ │              │
│  │  │    │    /  ┄┄┄┄  Store 002 (Boston)            │ │              │
│  │  │    │   /  /                                     │ │              │
│  │  │    │  /  /                                      │ │              │
│  │  │    │ /  /                                       │ │              │
│  │  │    └──────────────────────────────► Time       │ │              │
│  │  │     Jan   Mar   May   Jul   Sep                │ │              │
│  │  └────────────────────────────────────────────────┘ │              │
│  │                                                       │              │
│  │  Gap Analysis:                                       │              │
│  │  • Store 001 outperforms by avg 32%                 │              │
│  │  • Largest gap in July (seasonal peak)              │              │
│  └──────────────────────────────────────────────────────┘              │
│                                                                          │
│  Benchmark Mode:                                                        │
│  ┌──────────────────────────────────────────────────────┐              │
│  │  Performance vs Peer Group Average                   │              │
│  │  ┌────────────────────────────────────────────────┐ │              │
│  │  │        Your Stores    Peer Average             │ │              │
│  │  │  Revenue    $1.0M        $920K      ▲ 8.7%    │ │              │
│  │  │  Traffic      52K         48K       ▲ 8.3%    │ │              │
│  │  │  Conversion  15.2%       14.8%      ▲ 2.7%    │ │              │
│  │  │  Basket $    $82         $78        ▲ 5.1%    │ │              │
│  │  └────────────────────────────────────────────────┘ │              │
│  │                                                       │              │
│  │  Percentile Ranking: 73rd (Top quartile)            │              │
│  │  Areas of Excellence: Traffic, Basket Size          │              │
│  │  Improvement Opportunities: Inventory Turnover      │              │
│  └──────────────────────────────────────────────────────┘              │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

**File:** /mnt/d/github/pycaret/docs/dashboard_architecture_diagrams.md
**Purpose:** Visual architecture diagrams for store manager dashboard
**Coverage:** Component interactions, data flows, deployment architecture, store filtering
