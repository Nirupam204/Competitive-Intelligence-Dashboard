# Moonshot AI Agent Internship Assignment

Interactive competitive intelligence dashboard for luggage brands on Amazon India. The project scrapes product listings and visible review excerpts, synthesizes sentiment and themes, and serves a decision-ready dashboard for premium-vs-value comparisons.

## What the project covers

- 4 luggage brands tracked on Amazon India
- 10 products collected per brand
- Visible customer review excerpts scraped from each product page
- Brand and product comparison on price, discount, rating, sentiment, and review themes
- Agent insights that surface non-obvious competitive conclusions
- Aspect-level tracking for wheels, handles, durability, zippers, material, lock, space, design, and value

## Stack

- Python
- Playwright for scraping
- Pandas + scikit-learn + VADER for analytics
- Streamlit + Plotly for the dashboard

## Project structure

```text
.
├── app.py
├── data
│   ├── raw
│   └── processed
├── requirements.txt
└── src
    └── moonshot_dashboard
        ├── analyze.py
        ├── config.py
        └── scraper.py
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m playwright install chromium
```

## Run the full pipeline

1. Scrape Amazon India products and visible product-page reviews:

```bash
$env:PYTHONPATH="src"
python -m moonshot_dashboard.scraper
```

2. Build processed dashboard datasets:

```bash
$env:PYTHONPATH="src"
python -m moonshot_dashboard.analyze
```

3. Launch the dashboard:

```bash
streamlit run app.py
```

## Dashboard views

- Overview KPIs for brands, products, reviews, sentiment, and discounting
- Brand comparison view with price vs sentiment positioning and discount benchmarks
- Aspect sentiment heatmap to compare strengths and weaknesses
- Product drilldown with review synthesis, praise themes, complaint themes, and review excerpts
- Agent insights layer with premium positioning, value-for-money, discount dependency, and anomaly flags

## Sentiment methodology

- Each review uses a hybrid score:
  - VADER compound sentiment from the review title + body
  - A rating adjustment term derived from the star score
- Final review sentiment is clipped to `[-1, 1]`
- Brand and product sentiment are aggregated as the mean of review-level scores
- Themes are extracted from positive and negative review subsets using frequency-ranked uni/bi-grams
- Aspect-level signals are keyword-tagged across core luggage concerns like wheels, handle, durability, material, zipper, lock, and space

## Data notes and limitations

- Amazon India blocks simple HTTP scraping, so this project uses Playwright with a browser session.
- The environment used here exposed product listings and top review excerpts on product pages, but not the full paginated review pages without additional gating.
- Because of that, review coverage is based on visible excerpts from each product page rather than the complete review universe.
- Product availability, pricing, and discounts can change quickly on Amazon; rerun the scraper for the latest snapshot.

## Submission checklist

- Working dashboard
- Source code
- README
- Cleaned dataset in `data/raw` and `data/processed`

## Recommended next steps

- Record a short Loom walkthrough of the scraper, analytics flow, and dashboard interactions
- Add screenshots to the README after launching the app locally
- If needed, extend the scraper with authenticated review pagination to increase review depth
