from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from moonshot_dashboard.config import PROCESSED_DIR


st.set_page_config(
    page_title="Luggage Brand Intelligence",
    page_icon="🧳",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_data() -> dict[str, pd.DataFrame]:
    files = {
        "overview": PROCESSED_DIR / "overview.csv",
        "brand_summary": PROCESSED_DIR / "brand_summary.csv",
        "product_summary": PROCESSED_DIR / "product_summary.csv",
        "reviews_enriched": PROCESSED_DIR / "reviews_enriched.csv",
        "aspect_summary": PROCESSED_DIR / "aspect_summary.csv",
        "agent_insights": PROCESSED_DIR / "agent_insights.csv",
    }
    missing = [name for name, path in files.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing processed data files: "
            + ", ".join(missing)
            + ". Run the scraper and analysis pipeline first."
        )
    return {name: pd.read_csv(path) for name, path in files.items()}


def sentiment_label(score: float) -> str:
    if score >= 0.25:
        return "Strong positive"
    if score >= 0.05:
        return "Leaning positive"
    if score <= -0.2:
        return "Negative"
    return "Mixed"


data = load_data()
overview = data["overview"].iloc[0]
brand_summary = data["brand_summary"]
product_summary = data["product_summary"]
reviews_enriched = data["reviews_enriched"]
aspect_summary = data["aspect_summary"]
agent_insights = data["agent_insights"]

brand_options = sorted(product_summary["brand"].dropna().unique().tolist())
price_min = int(product_summary["current_price"].fillna(0).min())
price_max = int(product_summary["current_price"].fillna(0).max())
rating_min = float(product_summary["rating"].fillna(0).min())

with st.sidebar:
    st.markdown("## Filters")
    selected_brands = st.multiselect("Brand", brand_options, default=brand_options)
    selected_price = st.slider("Price range (Rs)", min_value=price_min, max_value=price_max, value=(price_min, price_max))
    min_rating = st.slider("Minimum rating", min_value=0.0, max_value=5.0, value=max(0.0, round(rating_min, 1)), step=0.1)
    size_options = sorted(product_summary["size_segment"].dropna().unique().tolist())
    selected_sizes = st.multiselect("Size segment", size_options, default=size_options)
    type_options = sorted(product_summary["luggage_type"].dropna().unique().tolist())
    selected_types = st.multiselect("Luggage type", type_options, default=type_options)
    selected_sentiments = st.multiselect("Review sentiment", ["Positive", "Neutral", "Negative"], default=["Positive", "Neutral", "Negative"])

filtered_products = product_summary[
    product_summary["brand"].isin(selected_brands)
    & product_summary["current_price"].fillna(0).between(*selected_price)
    & (product_summary["rating"].fillna(0) >= min_rating)
]
if selected_sizes:
    filtered_products = filtered_products[filtered_products["size_segment"].isin(selected_sizes)]
if selected_types:
    filtered_products = filtered_products[filtered_products["luggage_type"].isin(selected_types)]

filtered_reviews = reviews_enriched[
    reviews_enriched["brand"].isin(selected_brands)
    & reviews_enriched["asin"].isin(filtered_products["asin"])
    & reviews_enriched["sentiment_bucket"].isin(selected_sentiments)
]

brand_view = brand_summary[brand_summary["brand"].isin(selected_brands)].copy()
brand_view = brand_view.merge(
    filtered_products.groupby("brand", as_index=False)
    .agg(filtered_products=("asin", "count"), filtered_avg_price=("current_price", "mean")),
    on="brand",
    how="left",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(255, 232, 194, 0.9), transparent 35%),
            radial-gradient(circle at top right, rgba(189, 224, 254, 0.85), transparent 32%),
            linear-gradient(180deg, #f7f3ea 0%, #fffdf9 48%, #eef3f8 100%);
    }
    .hero {
        padding: 1.4rem 1.6rem;
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(18, 52, 86, 0.95), rgba(37, 99, 126, 0.90));
        color: #fff8ef;
        box-shadow: 0 20px 40px rgba(18, 52, 86, 0.18);
        margin-bottom: 1rem;
    }
    .hero h1 { font-size: 2.1rem; margin: 0 0 0.4rem 0; }
    .hero p { margin: 0; font-size: 1rem; color: rgba(255, 248, 239, 0.86); }
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid rgba(18, 52, 86, 0.08);
        border-radius: 20px;
        padding: 0.8rem;
        box-shadow: 0 10px 20px rgba(15, 23, 42, 0.06);
    }
    .insight-card {
        background: rgba(255, 255, 255, 0.88);
        padding: 1rem;
        border-radius: 18px;
        border: 1px solid rgba(18, 52, 86, 0.08);
        min-height: 160px;
    }
    .insight-card h4 { margin: 0 0 0.35rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="hero">
        <h1>Amazon India Luggage Intelligence</h1>
        <p>
            Competitive dashboard for premium-vs-value positioning, discount dependence,
            review sentiment, and recurring product strengths or weaknesses across luggage brands.
            Current filtered view covers <strong>{filtered_products['brand'].nunique()}</strong> brands,
            <strong>{len(filtered_products)}</strong> products, and <strong>{len(filtered_reviews)}</strong> review excerpts.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_cols = st.columns(5)
metric_cols[0].metric("Brands tracked", f"{int(overview['brands_tracked'])}")
metric_cols[1].metric("Products tracked", f"{len(filtered_products)}")
metric_cols[2].metric("Review excerpts", f"{len(filtered_reviews)}")
metric_cols[3].metric("Avg sentiment", f"{filtered_reviews['sentiment_score'].mean():.2f}")
metric_cols[4].metric("Avg discount", f"{filtered_products['discount_pct'].mean():.1f}%")

left, right = st.columns((1.25, 1))
with left:
    fig = px.scatter(
        brand_view,
        x="avg_price",
        y="avg_sentiment_score",
        size="total_review_count",
        color="brand",
        hover_data={"avg_discount_pct": ":.1f", "avg_rating": ":.2f"},
        labels={"avg_price": "Average selling price (Rs)", "avg_sentiment_score": "Average sentiment"},
        title="Brand positioning: price vs sentiment",
    )
    fig.update_layout(height=430, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.55)")
    st.plotly_chart(fig, use_container_width=True)

with right:
    bar = px.bar(
        brand_view.sort_values("avg_discount_pct", ascending=False),
        x="brand",
        y="avg_discount_pct",
        color="avg_discount_pct",
        color_continuous_scale="Sunsetdark",
        title="Average discount by brand",
        labels={"avg_discount_pct": "Discount %", "brand": ""},
    )
    bar.update_layout(height=430, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.55)", coloraxis_showscale=False)
    st.plotly_chart(bar, use_container_width=True)

st.markdown("## Agent Insights")
insight_cols = st.columns(min(3, max(1, len(agent_insights))))
for idx, (_, row) in enumerate(agent_insights.head(6).iterrows()):
    with insight_cols[idx % len(insight_cols)]:
        st.markdown(
            f"""
            <div class="insight-card">
                <h4>{row['insight_type']}: {row['brand']}</h4>
                <p>{row['insight']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("## Brand Comparison")
comparison_cols = st.columns((1.1, 0.9))
with comparison_cols[0]:
    table = brand_view[
        [
            "brand",
            "products_tracked",
            "avg_price",
            "avg_discount_pct",
            "avg_rating",
            "avg_sentiment_score",
            "value_for_money_score",
            "top_praise_themes",
            "top_complaint_themes",
        ]
    ].sort_values(["avg_sentiment_score", "value_for_money_score"], ascending=False)
    st.dataframe(
        table.rename(
            columns={
                "brand": "Brand",
                "products_tracked": "Products",
                "avg_price": "Avg price (Rs)",
                "avg_discount_pct": "Avg discount %",
                "avg_rating": "Avg rating",
                "avg_sentiment_score": "Sentiment",
                "value_for_money_score": "Value score",
                "top_praise_themes": "Top praise",
                "top_complaint_themes": "Top complaints",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
with comparison_cols[1]:
    heatmap_source = aspect_summary[aspect_summary["brand"].isin(selected_brands)].copy()
    if not heatmap_source.empty:
        heat = heatmap_source.pivot(index="aspect", columns="brand", values="avg_sentiment_score").fillna(0)
        heatmap = go.Figure(
            data=go.Heatmap(
                z=heat.values,
                x=heat.columns,
                y=heat.index,
                colorscale="RdYlGn",
                zmid=0,
                text=heat.round(2).values,
                texttemplate="%{text}",
            )
        )
        heatmap.update_layout(
            title="Aspect sentiment heatmap",
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.55)",
        )
        st.plotly_chart(heatmap, use_container_width=True)
    else:
        st.info("Aspect-level mentions were not detected for the active filters.")

st.markdown("## Product Drilldown")
asin_lookup = dict(zip(filtered_products["title"], filtered_products["asin"]))
selected_title = st.selectbox("Select a product", filtered_products["title"].sort_values().tolist())
selected_asin = asin_lookup[selected_title]
product_row = filtered_products[filtered_products["asin"] == selected_asin].iloc[0]
product_reviews = filtered_reviews[filtered_reviews["asin"] == selected_asin]

drill_cols = st.columns((1.15, 0.85))
with drill_cols[0]:
    st.subheader(product_row["title"])
    st.write(product_row["review_summary"])
    mini_metrics = st.columns(4)
    mini_metrics[0].metric("Price", f"Rs {product_row['current_price']:,.0f}")
    mini_metrics[1].metric("Discount", f"{product_row['discount_pct']:.1f}%")
    mini_metrics[2].metric("Rating", f"{product_row['rating']:.1f}")
    mini_metrics[3].metric("Sentiment", sentiment_label(float(product_row["avg_sentiment_score"])))
    st.markdown(f"**Top praise themes:** {product_row['top_praise_themes']}")
    st.markdown(f"**Top complaint themes:** {product_row['top_complaint_themes']}")

with drill_cols[1]:
    review_mix = product_reviews["sentiment_bucket"].value_counts().reindex(["Positive", "Neutral", "Negative"], fill_value=0).reset_index()
    review_mix.columns = ["sentiment_bucket", "count"]
    donut = px.pie(
        review_mix,
        names="sentiment_bucket",
        values="count",
        hole=0.58,
        color="sentiment_bucket",
        color_discrete_map={"Positive": "#2a9d8f", "Neutral": "#e9c46a", "Negative": "#e76f51"},
        title="Review sentiment mix",
    )
    donut.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(donut, use_container_width=True)

st.markdown("### Review excerpts")
review_preview = product_reviews[
    ["rating", "sentiment_bucket", "title", "body", "review_date", "verified_purchase"]
].rename(
    columns={
        "rating": "Stars",
        "sentiment_bucket": "Sentiment",
        "title": "Title",
        "body": "Review",
        "review_date": "Date",
        "verified_purchase": "Verified",
    }
)
st.dataframe(review_preview, use_container_width=True, hide_index=True)
