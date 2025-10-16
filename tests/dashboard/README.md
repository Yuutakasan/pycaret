# Dashboard Test Suite

Comprehensive test suite for the retail analytics dashboard with 90%+ code coverage.

## 📁 Test Structure

```
tests/dashboard/
├── conftest.py                      # Shared fixtures and configuration
├── pytest.ini                       # Pytest configuration
├── requirements-test.txt            # Test dependencies
├── unit/                           # Unit tests (fast, isolated)
│   ├── test_cache_manager.py       # Cache operations
│   ├── test_data_pipeline.py       # Data loading and preprocessing
│   └── test_orchestrator.py        # Orchestrator logic
├── integration/                    # Integration tests (component workflows)
│   └── test_workflow.py            # End-to-end workflows
├── performance/                    # Performance benchmarks
│   └── test_benchmarks.py          # Speed and resource tests
├── visualization/                  # Visualization tests
│   └── test_rendering.py           # Chart rendering and data prep
├── alerts/                         # Alert engine tests
│   └── test_alert_engine.py        # Alert logic and notifications
├── comparison/                     # Store comparison tests
│   └── test_store_accuracy.py      # Ranking and benchmarking
└── forecast/                       # Forecast evaluation tests
    └── test_model_evaluation.py    # Model accuracy and validation
```

## 🚀 Quick Start

### Install Dependencies

```bash
cd tests/dashboard
pip install -r requirements-test.txt
```

### Run All Tests

```bash
# Run all tests with coverage
pytest

# Run with parallel execution
pytest -n auto

# Run specific test category
pytest -m unit
pytest -m integration
pytest -m performance
```

## 📊 Test Categories

### Unit Tests (Fast)
```bash
pytest -m unit
```
- Cache manager operations
- Data loading and preprocessing
- Pipeline filtering
- Orchestrator logic

### Integration Tests
```bash
pytest -m integration
```
- End-to-end workflows
- Component interaction
- Data integrity
- Error handling

### Performance Tests
```bash
pytest -m performance
```
- Load time benchmarks
- Cache performance
- Memory usage
- Concurrent operations

### Visualization Tests
```bash
pytest -m visualization
```
- Chart data preparation
- Configuration validation
- Export formats
- Dashboard layouts

### Alert Tests
```bash
pytest -m alerts
```
- Threshold monitoring
- Alert triggers
- Notification generation
- Anomaly detection

### Comparison Tests
```bash
pytest -m comparison
```
- Store ranking algorithms
- Benchmarking accuracy
- Statistical significance
- Multi-dimensional analysis

### Forecast Tests
```bash
pytest -m forecast
```
- Accuracy metrics (MAE, RMSE, MAPE)
- Confidence intervals
- Residual analysis
- Cross-validation

## 📈 Coverage Reports

### Generate HTML Coverage Report
```bash
pytest --cov-report=html
# Open htmlcov/index.html in browser
```

### Terminal Coverage Summary
```bash
pytest --cov-report=term-missing
```

### Coverage Requirements
- Statements: >80%
- Branches: >75%
- Functions: >80%
- Lines: >80%
- **Overall Target: 90%+**

## 🎯 Test Execution Options

### Run Specific Test File
```bash
pytest tests/dashboard/unit/test_cache_manager.py
```

### Run Specific Test Class
```bash
pytest tests/dashboard/unit/test_cache_manager.py::TestCacheManager
```

### Run Specific Test Method
```bash
pytest tests/dashboard/unit/test_cache_manager.py::TestCacheManager::test_set_and_get_memory_cache
```

### Run Tests Matching Pattern
```bash
pytest -k "cache"
pytest -k "test_load"
```

### Skip Slow Tests
```bash
pytest -m "not slow"
```

### Verbose Output
```bash
pytest -v
pytest -vv  # Extra verbose
```

### Stop on First Failure
```bash
pytest -x
```

### Show Local Variables on Failure
```bash
pytest -l
```

## 🛠️ Writing New Tests

### Test Naming Conventions
- Files: `test_*.py`
- Classes: `Test*`
- Functions: `test_*`

### Using Fixtures
```python
def test_my_feature(sample_data, cache_manager, performance_timer):
    """Test with fixtures from conftest.py"""
    with performance_timer() as timer:
        result = cache_manager.set("key", sample_data)

    assert timer.elapsed_ms < 100
```

### Marking Tests
```python
import pytest

@pytest.mark.unit
def test_unit_feature():
    pass

@pytest.mark.slow
@pytest.mark.performance
def test_performance_feature():
    pass
```

### Parametrized Tests
```python
@pytest.mark.parametrize("value,expected", [
    (1, 2),
    (2, 4),
    (3, 6)
])
def test_doubling(value, expected):
    assert value * 2 == expected
```

## 🔍 Debugging Tests

### Print Debugging
```bash
pytest -s  # Show print statements
```

### PDB Debugging
```bash
pytest --pdb  # Drop into debugger on failure
```

### Show Captured Output
```bash
pytest --capture=no
```

## ⚡ Performance Tips

1. **Use Parallel Execution**
   ```bash
   pytest -n auto
   ```

2. **Run Fast Tests First**
   ```bash
   pytest -m unit && pytest -m integration
   ```

3. **Cache Test Results**
   ```bash
   pytest --lf  # Run last failed
   pytest --ff  # Failed first, then others
   ```

4. **Use Appropriate Fixtures**
   - Use `session` scope for expensive setup
   - Use `function` scope for test isolation

## 📝 Test Documentation

Each test should include:
- Clear docstring explaining what is tested
- Expected behavior
- Edge cases covered
- Any special setup required

Example:
```python
def test_cache_expiration(self, cache_manager):
    """
    Test that cache entries expire after TTL.

    Creates entry with 1-second TTL, verifies it's available
    immediately, then confirms it expires after timeout.
    """
    cache_manager.set("test_key", "test_value", ttl=1)
    assert cache_manager.get("test_key") == "test_value"

    time.sleep(1.1)
    assert cache_manager.get("test_key") is None
```

## 🐛 Common Issues

### Import Errors
- Ensure `src/` is in PYTHONPATH
- Install package in development mode: `pip install -e .`

### Fixture Not Found
- Check `conftest.py` is in correct location
- Verify fixture scope and name

### Tests Pass Locally but Fail in CI
- Check for hardcoded paths
- Ensure timezone consistency
- Verify random seed usage

## 📊 Continuous Integration

### Pre-commit Hooks
```bash
# Run tests before commit
pytest -m unit --maxfail=1
```

### CI Pipeline
```yaml
# .github/workflows/test.yml
- name: Run Tests
  run: |
    pytest -n auto --cov --cov-report=xml
- name: Upload Coverage
  uses: codecov/codecov-action@v3
```

## 🎓 Best Practices

1. **Test Independence**: Each test should run independently
2. **Clear Assertions**: Use descriptive assertion messages
3. **Test One Thing**: Each test should verify one behavior
4. **Fast Tests**: Keep unit tests under 100ms
5. **Meaningful Names**: Test names should explain what they test
6. **Arrange-Act-Assert**: Structure tests clearly
7. **Mock External Dependencies**: Keep tests isolated
8. **Use Fixtures**: Reuse test data and setup

## 📚 Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Coverage.py Guide](https://coverage.readthedocs.io/)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)
