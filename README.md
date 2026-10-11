# al1-surrogate

A surrogate model for the AL1 potential based on [Meng, Wang, and Oka (2024)](https://doi.org/10.48550/arXiv.2404.01238), providing fast and accurate approximations for calculating mass of mesons.

## Overview

This repository contains a surrogate model implementation for the AL1 potential. Surrogate models are machine learning-based approximations that can significantly speed up calculations compared to full quantum mechanical or classical potential evaluations.

## Features

- Fast approximation of AL1 potential calculations
- Machine learning-based surrogate model
- Optimized for computational efficiency
- Accurate predictions for materials simulations

## Installation

### Prerequisites

- Python 3.7 or higher
- pip or conda

### Setup

Clone the repository:

```bash
git clone https://github.com/o0muay72670o/al1-surrogate.git
cd al1-surrogate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

```
al1-surrogate/
├── README.md              # This file
├── .gitignore             # Git ignore file
├── requirements.txt       # Python dependencies
└── src/                   # Source code directory
```

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests to help improve this project.

## Citation

If you use this dataset in your research, please cite it using the metadata provided in [`CITATION.cff`](./CITATION.cff).
## Contact

For questions or inquiries, please open an issue in the repository.

## References

AL1 potential: [https://arxiv.org/abs/2404.01238](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.111.014018)

## Acknowledgments

This work was conducted as part of the [R-IH Summer Program Internship 2026 (Wako Campus)](https://www.riken.jp/en/news_pubs/events/lectures/20260723_2/index.html), hosted by the [Fundamental Physics Data Sharing Unit at the RIKEN Information R&D and Strategy Headquarters (R-IH)](https://r-ih.riken.jp/en/index-2/irdd/physics/).

