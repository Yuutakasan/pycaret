# Production Validation Test Suite

Comprehensive production readiness testing for the Dashboard System.

## Overview

This test suite validates the dashboard system against production requirements across 7 critical dimensions:

1. **Performance** - Response times, throughput, resource utilization
2. **Load** - Concurrent users, sustained load, stress testing
3. **Security** - Access control, input validation, data integrity
4. **Usability** - User workflows, error handling, data freshness
5. **Accessibility** - WCAG 2.1 AA compliance (backend API)
6. **Mobile Responsiveness** - Bandwidth optimization, mobile performance
7. **Disaster Recovery** - Backup/restore, failure recovery, resilience

## Test Suite Structure

```
production_validation/
├── __init__.py                      # Test suite metadata
├── test_performance.py              # 12 performance tests
├── test_load.py                     # 8 load/concurrency tests
├── test_security.py                 # 15 security audit tests
├── test_usability.py                # 11 usability workflow tests
├── test_accessibility.py            # 10 WCAG compliance tests
├── test_mobile_responsiveness.py    # 13 mobile optimization tests
└── test_disaster_recovery.py        # 12 disaster recovery tests
```

**Total:** 81 comprehensive validation tests

## Running the Tests

### Run All Tests

```bash
# Run complete test suite
pytest tests/dashboard/production_validation/ -v

# Run with coverage
pytest tests/dashboard/production_validation/ --cov=src/dashboard --cov-report=html
```

### Run by Category

```bash
# Performance tests
pytest tests/dashboard/production_validation/test_performance.py -v

# Load tests
pytest tests/dashboard/production_validation/test_load.py -v

# Security tests
pytest tests/dashboard/production_validation/test_security.py -v

# Usability tests
pytest tests/dashboard/production_validation/test_usability.py -v

# Accessibility tests
pytest tests/dashboard/production_validation/test_accessibility.py -v

# Mobile tests
pytest tests/dashboard/production_validation/test_mobile_responsiveness.py -v

# Disaster recovery tests
pytest tests/dashboard/production_validation/test_disaster_recovery.py -v
```

### Run Specific Test Classes

```bash
# Performance benchmarks only
pytest tests/dashboard/production_validation/test_performance.py::TestProductionPerformance -v

# Concurrent user tests only
pytest tests/dashboard/production_validation/test_load.py::TestConcurrentUserLoad -v

# Security audit only
pytest tests/dashboard/production_validation/test_security.py::TestDataAccessControl -v
```

### Run with Markers

```bash
# Run only slow tests
pytest tests/dashboard/production_validation/ -v -m slow

# Skip slow tests
pytest tests/dashboard/production_validation/ -v -m "not slow"
```

## Test Requirements

### Performance Requirements

- Initial data load: < 10 seconds (146K records)
- Data preprocessing: < 15 seconds
- Query response: < 2 seconds
- Cached query: < 100ms
- Memory usage: < 500MB for production dataset

### Load Requirements

- 5 concurrent users: < 15 seconds total
- 10 concurrent users: < 30 seconds total
- 20 concurrent users: 95%+ success rate
- Sustained load: < 5% error rate

### Security Requirements

- Store-level data isolation enforced
- SQL injection attempts blocked
- Path traversal attacks prevented
- Cache keys sanitized (MD5 hashing)
- Input validation on all parameters

### Usability Requirements

- Daily dashboard check: < 5 seconds
- Weekly report generation: < 10 seconds
- Multi-store comparison: < 15 seconds
- Clear error messages
- Graceful degradation on errors

### Accessibility Requirements (WCAG 2.1 Level AA)

- Structured, semantic data (SC 1.3.1)
- Operations complete < 20s (SC 2.2.1)
- Consistent API structure (SC 3.2.4)
- Clear status messages (SC 4.1.3)
- Programmatic access (keyboard equivalent)

### Mobile Requirements

- Payload size: < 100KB for simple queries
- 3G network performance: < 3 seconds
- Cached mobile queries: < 200ms
- Offline capability via cache
- Compact JSON format

### Disaster Recovery Requirements

- Cache persistence through restart
- Recovery from corrupted cache
- Graceful handling of missing data
- Failure isolation (partial failures don't crash system)
- System uptime: > 99%

## Test Data

Tests use realistic production data:

- **Small Dataset:** 3,650 records (10 stores × 365 days)
- **Medium Dataset:** 36,500 records (100 stores × 365 days)
- **Large Dataset:** 146,000 records (100 stores × 4 years)
- **Extra Large:** 365,000 records (200 stores × 5 years)

All test data is generated programmatically using numpy/pandas with controlled random seeds for reproducibility.

## Expected Results

### Passing Criteria

All tests should pass with these characteristics:

- **Performance:** All response times within limits
- **Load:** 95%+ success rate under concurrent load
- **Security:** No data leakage, all attacks blocked
- **Usability:** All workflows complete within target times
- **Accessibility:** WCAG 2.1 AA compliance
- **Mobile:** Bandwidth-optimized, offline-capable
- **Disaster Recovery:** >99% uptime, graceful degradation

### Sample Output

```
tests/dashboard/production_validation/test_performance.py::TestProductionPerformance::test_initial_load_performance PASSED
tests/dashboard/production_validation/test_performance.py::TestProductionPerformance::test_preprocessing_performance PASSED
tests/dashboard/production_validation/test_performance.py::TestProductionPerformance::test_query_response_time PASSED
...

====== 81 passed in 120.45s ======
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Production Validation

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Run production validation tests
        run: |
          pytest tests/dashboard/production_validation/ \
            --cov=src/dashboard \
            --cov-report=xml \
            --junit-xml=test-results.xml
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

## Troubleshooting

### Common Issues

**Issue:** Tests fail with "ModuleNotFoundError"
```bash
# Solution: Ensure src directory is in PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/pycaret"
```

**Issue:** Tests timeout
```bash
# Solution: Increase timeout for slow tests
pytest tests/dashboard/production_validation/ --timeout=300
```

**Issue:** Memory errors on large datasets
```bash
# Solution: Run tests individually or increase system resources
pytest tests/dashboard/production_validation/test_performance.py::TestProductionPerformance -v
```

## Dependencies

Required Python packages:

```
pytest>=7.0.0
pandas>=1.5.0
numpy>=1.24.0
psutil>=5.9.0
```

For coverage:
```
pytest-cov>=4.0.0
```

## Continuous Monitoring

These tests should be run:

- **Pre-commit:** Critical tests (fast subset)
- **Pre-merge:** Full test suite
- **Nightly:** Full test suite with large datasets
- **Pre-release:** Full test suite + manual validation

## Related Documentation

- [Production Readiness Report](../../../docs/production_readiness_report.md)
- [System Architecture](../../../docs/architecture.md)
- [API Documentation](../../../docs/api.md)
- [Security Guidelines](../../../docs/security.md)

## Contributing

When adding new tests:

1. Follow existing test structure and naming conventions
2. Use pytest fixtures for test data
3. Include clear docstrings explaining what is tested
4. Update this README with new test categories
5. Ensure tests are independent and can run in parallel

## License

Same as parent project.

---

**Last Updated:** 2025-10-08
**Test Suite Version:** 1.0.0
**Maintained by:** Production Validation Team
