from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from moonshot_dashboard.config import ASPECT_KEYWORDS, PROCESSED_DIR, RAW_DIR


THEME_STOPWORDS = ENGLISH_STOP_WORDS.union(
    {
        "bag",
        "bags",
        "luggage",
        "trolley",
        "travel",
        "product",
        "amazon",
        "good",
        "nice",
        "great",
        "quality",
        "use",
        "using",
    }
)


def normalize_text(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", str(text).lower())
    return re.sub(r"\s+", " ", text).strip()


def sentiment_bucket(score: float) -> str:
    if score >= 0.25:
        return "Positive"
    if score <= -0.2:
        return "Negative"
    return "Neutral"


def detect_aspects(text: str) -> list[str]:
    detected: list[str] = []
    text_lower = text.lower()
    for aspect, keywords in ASPECT_KEYWORDS.items():
        if any(keyword in text_lower for keyword in keywords):
            detected.append(aspect)
    return detected


def extract_top_phrases(texts: list[str], top_n: int = 5) -> list[str]:
    cleaned = [normalize_text(text) for text in texts if text]
    cleaned = [text for text in cleaned if len(text.split()) >= 2]
    if not cleaned:
        return []
    vectorizer = CountVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        stop_words=list(THEME_STOPWORDS),
        max_features=300,
    )
    matrix = vectorizer.fit_transform(cleaned)
    frequencies = np.asarray(matrix.sum(axis=0)).ravel()
    phrases = vectorizer.get_feature_names_out()
    ranked = sorted(zip(phrases, frequencies), key=lambda item: item[1], reverse=True)
    results: list[str] = []
    for phrase, _ in ranked:
        if phrase.isdigit():
            continue
        if any(phrase in existing or existing in phrase for existing in results):
            continue
        results.append(phrase)
        if len(results) >= top_n:
            break
    return results


def build_agent_insights(brand_df: pd.DataFrame, reviews_df: pd.DataFrame) -> pd.DataFrame:
    insights: list[dict] = []
    if brand_df.empty:
        return pd.DataFrame(columns=["brand", "insight_type", "insight"])

    premium_brand = brand_df.sort_values("avg_price", ascending=False).iloc[0]
    best_sentiment = brand_df.sort_values("avg_sentiment_score", ascending=False).iloc[0]
    value_leader = brand_df.sort_values("value_for_money_score", ascending=False).iloc[0]
    discount_heavy = brand_df.sort_values("avg_discount_pct", ascending=False).iloc[0]
    complaint_risk = brand_df.sort_values("durability_negative_share", ascending=False).iloc[0]

    insights.extend(
        [
            {
                "brand": premium_brand["brand"],
                "insight_type": "Positioning",
                "insight": (
                    f"{premium_brand['brand']} sits at the highest average selling price "
                    f"(Rs {premium_brand['avg_price']:,.0f}), making it the clearest premium-positioned brand."
                ),
            },
            {
                "brand": best_sentiment["brand"],
                "insight_type": "Sentiment Lead",
                "insight": (
                    f"{best_sentiment['brand']} leads on sentiment while maintaining a solid {best_sentiment['avg_rating']:.2f} star average, "
                    "suggesting customers broadly feel the product promise matches reality."
                ),
            },
            {
                "brand": value_leader["brand"],
                "insight_type": "Value",
                "insight": (
                    f"{value_leader['brand']} delivers the strongest value-for-money score, balancing sentiment, star rating, and price better than peers."
                ),
            },
            {
                "brand": discount_heavy["brand"],
                "insight_type": "Promotion Dependence",
                "insight": (
                    f"{discount_heavy['brand']} relies most on discounting with an average markdown of {discount_heavy['avg_discount_pct']:.1f}%, "
                    "which may be boosting appeal but can signal weaker full-price pricing power."
                ),
            },
            {
                "brand": complaint_risk["brand"],
                "insight_type": "Risk",
                "insight": (
                    f"{complaint_risk['brand']} shows the highest share of durability-related negative reviews, making after-sales trust a likely pressure point."
                ),
            },
        ]
    )

    high_rating = reviews_df[(reviews_df["rating"].fillna(0) >= 4.0) & (reviews_df["sentiment_bucket"] == "Negative")]
    if not high_rating.empty:
        anomaly_brand = (
            high_rating.groupby("brand")
            .size()
            .sort_values(ascending=False)
            .reset_index(name="anomaly_count")
            .iloc[0]
        )
        insights.append(
            {
                "brand": anomaly_brand["brand"],
                "insight_type": "Anomaly",
                "insight": (
                    f"{anomaly_brand['brand']} has the most high-star but text-negative reviews, which often indicates customers like the look or price but still mention operational issues."
                ),
            }
        )

    return pd.DataFrame(insights)


def compute_metrics(products_df: pd.DataFrame, reviews_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    analyzer = SentimentIntensityAnalyzer()

    reviews_df = reviews_df.copy()
    reviews_df["review_text"] = (
        reviews_df["title"].fillna("").astype(str).str.strip()
        + ". "
        + reviews_df["body"].fillna("").astype(str).str.strip()
    ).str.strip(". ")
    reviews_df["text_sentiment"] = reviews_df["review_text"].apply(lambda text: analyzer.polarity_scores(text)["compound"])
    reviews_df["rating_adjustment"] = reviews_df["rating"].fillna(0).apply(lambda rating: ((rating - 3) / 2) * 0.35)
    reviews_df["sentiment_score"] = (reviews_df["text_sentiment"] + reviews_df["rating_adjustment"]).clip(-1, 1)
    reviews_df["sentiment_bucket"] = reviews_df["sentiment_score"].apply(sentiment_bucket)
    reviews_df["aspects"] = reviews_df["review_text"].apply(detect_aspects)
    reviews_df["review_length"] = reviews_df["body"].fillna("").astype(str).str.split().str.len()

    aspect_rows: list[dict] = []
    for _, row in reviews_df.iterrows():
        for aspect in row["aspects"]:
            aspect_rows.append(
                {
                    "brand": row["brand"],
                    "asin": row["asin"],
                    "aspect": aspect,
                    "sentiment_score": row["sentiment_score"],
                    "sentiment_bucket": row["sentiment_bucket"],
                }
            )
    aspect_df = pd.DataFrame(aspect_rows)

    products_df = products_df.copy()
    calculated_discount = pd.Series(
        np.where(
            (products_df["current_price"].notna())
            & (products_df["list_price"].notna())
            & (products_df["list_price"] > 0),
            ((products_df["list_price"] - products_df["current_price"]) / products_df["list_price"]) * 100,
            np.nan,
        ),
        index=products_df.index,
    )
    products_df["discount_pct"] = products_df["discount_pct"].fillna(calculated_discount)

    product_sentiment = (
        reviews_df.groupby("asin", as_index=False)
        .agg(
            avg_sentiment_score=("sentiment_score", "mean"),
            positive_share=("sentiment_bucket", lambda values: (values == "Positive").mean()),
            negative_share=("sentiment_bucket", lambda values: (values == "Negative").mean()),
            visible_review_samples=("review_id", "count"),
        )
    )

    products_enriched = products_df.merge(product_sentiment, on="asin", how="left")
    products_enriched["price_band"] = pd.qcut(
        products_enriched["current_price"].rank(method="first"),
        q=4,
        labels=["Value", "Mid", "Premium", "Luxury"],
    )
    products_enriched["value_for_money_score"] = (
        products_enriched["avg_sentiment_score"].fillna(0) * 50
        + products_enriched["rating"].fillna(0) * 10
        + products_enriched["discount_pct"].fillna(0) * 0.2
        - products_enriched["current_price"].fillna(products_enriched["current_price"].median()) / 400
    )

    product_themes: list[dict] = []
    for asin, group in reviews_df.groupby("asin"):
        positive_texts = group.loc[group["sentiment_bucket"] == "Positive", "review_text"].tolist()
        negative_texts = group.loc[group["sentiment_bucket"] == "Negative", "review_text"].tolist()
        praise = extract_top_phrases(positive_texts, top_n=4)
        complaints = extract_top_phrases(negative_texts, top_n=4)
        product_themes.append(
            {
                "asin": asin,
                "top_praise_themes": " | ".join(praise),
                "top_complaint_themes": " | ".join(complaints),
                "review_summary": (
                    f"Customers most often praise {', '.join(praise) or 'overall usability'}; "
                    f"complaints center on {', '.join(complaints) or 'isolated issues'}."
                ),
            }
        )

    products_enriched = products_enriched.merge(pd.DataFrame(product_themes), on="asin", how="left")

    brand_summary = (
        products_enriched.groupby("brand", as_index=False)
        .agg(
            products_tracked=("asin", "count"),
            avg_price=("current_price", "mean"),
            avg_list_price=("list_price", "mean"),
            avg_discount_pct=("discount_pct", "mean"),
            avg_rating=("rating", "mean"),
            total_review_count=("review_count", "sum"),
            avg_sentiment_score=("avg_sentiment_score", "mean"),
            avg_value_score=("value_for_money_score", "mean"),
        )
        .rename(columns={"avg_value_score": "value_for_money_score"})
    )

    theme_summary: list[dict] = []
    for brand, group in reviews_df.groupby("brand"):
        positives = group.loc[group["sentiment_bucket"] == "Positive", "review_text"].tolist()
        negatives = group.loc[group["sentiment_bucket"] == "Negative", "review_text"].tolist()
        theme_summary.append(
            {
                "brand": brand,
                "top_praise_themes": " | ".join(extract_top_phrases(positives)),
                "top_complaint_themes": " | ".join(extract_top_phrases(negatives)),
                "positive_review_share": (group["sentiment_bucket"] == "Positive").mean(),
                "negative_review_share": (group["sentiment_bucket"] == "Negative").mean(),
            }
        )
    brand_summary = brand_summary.merge(pd.DataFrame(theme_summary), on="brand", how="left")

    if not aspect_df.empty:
        durability_negative = (
            aspect_df.assign(is_durability_negative=lambda df: (df["aspect"] == "durability") & (df["sentiment_bucket"] == "Negative"))
            .groupby("brand", as_index=False)["is_durability_negative"]
            .mean()
            .rename(columns={"is_durability_negative": "durability_negative_share"})
        )
    else:
        durability_negative = pd.DataFrame({"brand": brand_summary["brand"], "durability_negative_share": 0.0})
    brand_summary = brand_summary.merge(durability_negative, on="brand", how="left").fillna({"durability_negative_share": 0.0})

    aspect_summary = (
        aspect_df.groupby(["brand", "aspect"], as_index=False)
        .agg(
            mentions=("aspect", "count"),
            avg_sentiment_score=("sentiment_score", "mean"),
            negative_share=("sentiment_bucket", lambda values: (values == "Negative").mean()),
        )
        if not aspect_df.empty
        else pd.DataFrame(columns=["brand", "aspect", "mentions", "avg_sentiment_score", "negative_share"])
    )

    overview = pd.DataFrame(
        [
            {
                "brands_tracked": products_enriched["brand"].nunique(),
                "products_tracked": len(products_enriched),
                "visible_reviews_analyzed": len(reviews_df),
                "avg_sentiment_score": reviews_df["sentiment_score"].mean(),
                "avg_price": products_enriched["current_price"].mean(),
                "avg_discount_pct": products_enriched["discount_pct"].mean(),
            }
        ]
    )

    agent_insights = build_agent_insights(brand_summary, reviews_df)

    return {
        "overview": overview,
        "brand_summary": brand_summary,
        "product_summary": products_enriched,
        "reviews_enriched": reviews_df,
        "aspect_summary": aspect_summary,
        "agent_insights": agent_insights,
    }


def save_outputs(outputs: dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, df in outputs.items():
        df.to_csv(output_dir / f"{name}.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze scraped luggage data and build dashboard-ready datasets.")
    parser.add_argument("--products-path", type=Path, default=RAW_DIR / "products.csv")
    parser.add_argument("--reviews-path", type=Path, default=RAW_DIR / "reviews.csv")
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    products_df = pd.read_csv(args.products_path)
    reviews_df = pd.read_csv(args.reviews_path)
    outputs = compute_metrics(products_df, reviews_df)
    save_outputs(outputs, args.output_dir)
    print(f"Saved processed outputs to {args.output_dir}.")


if __name__ == "__main__":
    main()
