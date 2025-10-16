# Alert Engine Installation Guide

## Prerequisites

- Python 3.8+
- pip package manager

## Dependencies

```bash
pip install pandas>=1.3.0 numpy>=1.21.0 scikit-learn>=1.0.0
```

## Quick Installation

```bash
# Navigate to project directory
cd /mnt/d/github/pycaret

# Install dependencies
pip install -r requirements.txt  # If available
# OR
pip install pandas numpy scikit-learn

# Verify installation
python3 -c "from src.engine.alert_engine import AlertEngine; print('✅ Alert Engine ready')"
```

## Environment Setup

```bash
# Optional: Set custom database path
export ALERT_DB_PATH=".swarm/alerts.db"

# Optional: Set ML parameters
export ML_CONTAMINATION=0.1
export ML_MIN_SAMPLES=10
```

## First Run

```python
from src.engine.alert_engine import AlertEngine
import pandas as pd

# Initialize engine
engine = AlertEngine()

# Test with sample data
inventory = pd.DataFrame([
    {
        "product_id": "TEST001",
        "stock_level": 15,
        "safety_stock": 50,
        "lead_time_days": 3
    }
])

# Run check
results = engine.run_full_check(inventory_data=inventory, sales_data=pd.DataFrame())

print(f"✅ Generated {results['total_alerts']} alerts")
```

## Troubleshooting

### Import Error
```bash
# Ensure src is in Python path
export PYTHONPATH="${PYTHONPATH}:/mnt/d/github/pycaret"
```

### Database Permission Error
```bash
# Ensure .swarm directory exists
mkdir -p .swarm
chmod 755 .swarm
```

### Scikit-learn Import Error
```bash
# Reinstall scikit-learn
pip uninstall scikit-learn -y
pip install scikit-learn
```

## Next Steps

1. Read documentation: `docs/alert_engine_guide.md`
2. Run examples: `python3 examples/alert_engine_examples.py`
3. Integrate with your application
