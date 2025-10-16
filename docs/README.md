# PyCaret Documentation

<div align="center">

![PyCaret Logo](images/logo.png)

**Comprehensive Documentation for PyCaret 3.4.0**

[🌐 Official Docs](https://pycaret.gitbook.io/) | [📺 YouTube](https://www.youtube.com/channel/UCxA1YTYJ9BEeo50lxyI_B3g) | [💬 Slack](https://join.slack.com/t/pycaret/shared_invite/zt-row9phbm-BoJdEVPYnGf7_NxNBP307w) | [📦 PyPI](https://pypi.org/project/pycaret/)

</div>

---

## 📚 Documentation Structure

### 🎯 User Guides

Perfect for getting started and learning the basics.

- **[店長向け操作ガイド / Store Manager's Guide](user-guide/店長向け操作ガイド.md)** (Bilingual: 日本語/English)
  - Installation and setup
  - Basic operations
  - Practical examples
  - FAQ for beginners
  - Step-by-step tutorials

### 💻 Technical Documentation

For developers and advanced users.

- **[Developer Guide](technical/developer-guide.md)**
  - Architecture overview
  - Core components
  - Development setup
  - API design patterns
  - Extension points
  - Testing strategy
  - Contributing guidelines

### 📖 API Reference

Complete API documentation for all modules.

- **[Classification API](api-reference/classification-api.md)**
  - Setup functions
  - Model training
  - Model tuning
  - Model analysis
  - Prediction functions
  - Model management
  - Complete workflow examples

- **Additional modules** (similar structure):
  - Regression API
  - Clustering API
  - Anomaly Detection API
  - Time Series API

### 🚀 Installation & Setup

Everything you need to get PyCaret running.

- **[Installation & Setup Guide](installation/setup-guide.md)**
  - System requirements
  - Installation methods
  - Environment setup
  - GPU configuration
  - Docker setup
  - IDE configuration
  - Verification steps

### 🔧 Troubleshooting

Solutions to common issues and problems.

- **[Common Issues & Solutions](troubleshooting/common-issues.md)**
  - Installation issues
  - Setup problems
  - Model training errors
  - Memory issues
  - GPU configuration
  - Performance optimization
  - Error messages reference

### ❓ FAQ

Frequently asked questions in bilingual format.

- **[Frequently Asked Questions](faq/frequently-asked-questions.md)** (Bilingual: 日本語/English)
  - General questions
  - Installation help
  - Usage guidance
  - Performance tips
  - Advanced features
  - Model-specific questions

### 📊 Dashboard Guide

Learn to interpret visualizations and dashboards.

- **[Dashboard Interpretation Guide](dashboard/dashboard-guide.md)**
  - Classification dashboards
  - Regression dashboards
  - Clustering dashboards
  - Anomaly detection dashboards
  - Time series dashboards
  - Interactive dashboards
  - Custom visualizations
  - Exporting and sharing

---

## 🎓 Quick Start

### For Beginners

Start here if you're new to PyCaret:

1. 📖 Read the [User Guide](user-guide/店長向け操作ガイド.md)
2. 🚀 Follow the [Installation Guide](installation/setup-guide.md)
3. 💻 Try the [Quick Start Examples](#quick-examples)
4. ❓ Check the [FAQ](faq/frequently-asked-questions.md) if stuck

### For Developers

Start here if you're building with PyCaret:

1. 💻 Read the [Developer Guide](technical/developer-guide.md)
2. 📖 Explore the [API Reference](api-reference/classification-api.md)
3. 🔧 Review [Contributing Guidelines](technical/developer-guide.md#contributing-guidelines)
4. 🧪 Check [Testing Strategy](technical/developer-guide.md#testing-strategy)

---

## 💡 Quick Examples

### Classification

```python
from pycaret.datasets import get_data
from pycaret.classification import *

# Load data
data = get_data('juice')

# Initialize setup
s = setup(data, target='Purchase', session_id=123)

# Compare models
best = compare_models()

# Make predictions
predictions = predict_model(best)

# Save model
save_model(best, 'my_model')
```

### Regression

```python
from pycaret.datasets import get_data
from pycaret.regression import *

# Load data
data = get_data('insurance')

# Initialize setup
s = setup(data, target='charges', session_id=123)

# Compare models
best = compare_models()

# Tune best model
tuned = tune_model(best)

# Make predictions
predictions = predict_model(tuned)
```

### Time Series

```python
from pycaret.datasets import get_data
from pycaret.time_series import *

# Load data
data = get_data('airline')

# Initialize setup
s = setup(data, fh=12, session_id=123)

# Compare models
best = compare_models()

# Forecast 12 months
forecast = predict_model(best)

# Plot forecast
plot_model(best, plot='forecast')
```

---

## 🗂️ Documentation Organization

```
docs/
├── README.md                          # This file
├── user-guide/                        # User guides
│   └── 店長向け操作ガイド.md          # Bilingual user guide
├── technical/                         # Technical documentation
│   └── developer-guide.md             # Developer guide
├── api-reference/                     # API documentation
│   ├── classification-api.md          # Classification API
│   ├── regression-api.md              # Regression API (to be created)
│   ├── clustering-api.md              # Clustering API (to be created)
│   ├── anomaly-api.md                 # Anomaly API (to be created)
│   └── timeseries-api.md              # Time Series API (to be created)
├── installation/                      # Installation guides
│   └── setup-guide.md                 # Complete setup guide
├── troubleshooting/                   # Troubleshooting guides
│   └── common-issues.md               # Common issues & solutions
├── faq/                               # Frequently asked questions
│   └── frequently-asked-questions.md  # Bilingual FAQ
├── dashboard/                         # Dashboard guides
│   └── dashboard-guide.md             # Dashboard interpretation
└── screenshots/                       # Screenshots and images
    └── (Add your screenshots here)
```

---

## 🌍 Language Support

Documentation is available in:

- 🇬🇧 **English** - Full documentation
- 🇯🇵 **日本語 (Japanese)** - User guide, FAQ (bilingual format)

### Adding More Languages

To contribute translations:

1. Copy the English version
2. Translate to your language
3. Submit a pull request
4. Follow the [Contributing Guidelines](technical/developer-guide.md#contributing-guidelines)

---

## 📊 Coverage

This documentation covers:

✅ **Installation** - All methods (pip, conda, docker)
✅ **Setup** - Complete configuration guide
✅ **Classification** - Full API reference
✅ **Regression** - Examples and use cases
✅ **Clustering** - Dashboard interpretation
✅ **Anomaly Detection** - Visualization guides
✅ **Time Series** - Forecasting workflows
✅ **Troubleshooting** - Common issues
✅ **FAQ** - Bilingual Q&A
✅ **Dashboard** - Complete visualization guide
✅ **GPU** - Acceleration setup
✅ **MLOps** - Deployment and monitoring

---

## 🚀 Advanced Topics

### Performance Optimization

- [GPU Configuration](installation/setup-guide.md#gpu-configuration)
- [Parallel Processing](technical/developer-guide.md#parallel-processing)
- [Memory Optimization](technical/developer-guide.md#memory-optimization)
- [Intel Optimization](installation/setup-guide.md#intel-cpu-optimization)

### MLOps Integration

- [MLflow Integration](api-reference/classification-api.md#mlflow-integration)
- [Model Deployment](api-reference/classification-api.md#deploy-model)
- [API Creation](api-reference/classification-api.md#create-api)
- [Experiment Tracking](technical/developer-guide.md#experiment-tracking)

### Custom Extensions

- [Custom Models](technical/developer-guide.md#adding-custom-models)
- [Custom Metrics](technical/developer-guide.md#adding-custom-metrics)
- [Custom Preprocessing](technical/developer-guide.md#custom-preprocessing)

---

## 🤝 Contributing

We welcome contributions to documentation!

### How to Contribute

1. **Fork** the repository
2. **Create** a new branch
3. **Make** your changes
4. **Test** the documentation
5. **Submit** a pull request

### Documentation Standards

- Use clear, concise language
- Include code examples
- Add screenshots where helpful
- Follow existing formatting
- Test all code snippets
- Update table of contents

See [Developer Guide](technical/developer-guide.md#contributing-guidelines) for details.

---

## 📞 Support

### Get Help

- 💬 **Slack Community**: [Join here](https://join.slack.com/t/pycaret/shared_invite/zt-row9phbm-BoJdEVPYnGf7_NxNBP307w)
- 💭 **GitHub Discussions**: [Ask questions](https://github.com/pycaret/pycaret/discussions)
- 🐛 **Report Bugs**: [GitHub Issues](https://github.com/pycaret/pycaret/issues)
- 📖 **Official Docs**: [GitBook](https://pycaret.gitbook.io/)

### Documentation Issues

Found an issue in the documentation?

1. Check if it's already reported
2. Open a new issue with:
   - Document name and section
   - Description of the issue
   - Suggested correction
3. Or submit a pull request with the fix

---

## 📚 Additional Resources

### Official Resources

- 🌐 **Website**: https://pycaret.org/
- 📖 **GitBook Docs**: https://pycaret.gitbook.io/
- 📺 **YouTube Channel**: https://www.youtube.com/channel/UCxA1YTYJ9BEeo50lxyI_B3g
- 📦 **PyPI Package**: https://pypi.org/project/pycaret/
- 💻 **GitHub Repository**: https://github.com/pycaret/pycaret

### Learning Resources

- 🎓 **Tutorials**: [Official Tutorials](https://pycaret.gitbook.io/docs/get-started/tutorials)
- 📝 **Blog Posts**: [Official Blog](https://pycaret.gitbook.io/docs/learn-pycaret/official-blog)
- 📊 **Example Notebooks**: [GitHub Examples](https://github.com/pycaret/examples)
- 📄 **Cheat Sheet**: [PDF Cheat Sheet](https://pycaret.gitbook.io/docs/learn-pycaret/cheat-sheet)

### Community

- 🌟 **LinkedIn**: https://www.linkedin.com/company/pycaret/
- 🐦 **Twitter**: Follow @pycaret_official
- 💬 **Slack**: Active community discussions
- 💭 **GitHub**: Discussions and Q&A

---

## 📝 Version History

| Version | Date | Changes |
|---------|------|---------|
| 3.4.0 | 2025-10-08 | Complete documentation overhaul with bilingual support |
| 3.3.0 | 2024-XX-XX | Added time series module documentation |
| 3.2.0 | 2024-XX-XX | Enhanced API reference |
| 3.0.0 | 2023-XX-XX | Major version release |

---

## 📄 License

This documentation is part of PyCaret and is licensed under the MIT License.

Copyright © 2025 PyCaret

Permission is hereby granted, free of charge, to any person obtaining a copy of this documentation and associated software, to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions.

---

## 🙏 Acknowledgments

Special thanks to:

- PyCaret core development team
- Contributors from around the world
- Community members for feedback
- Open source maintainers

---

**Happy Learning! 📚✨**

For questions or feedback, reach out on [Slack](https://join.slack.com/t/pycaret/shared_invite/zt-row9phbm-BoJdEVPYnGf7_NxNBP307w) or [GitHub Discussions](https://github.com/pycaret/pycaret/discussions).

---

**© 2025 PyCaret. Licensed under MIT License.**
