from __future__ import annotations

import argparse
import asyncio
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Iterable
from urllib.parse import quote_plus, urljoin

import pandas as pd
from bs4 import BeautifulSoup
from playwright.async_api import Browser, Error as PlaywrightError, Page, async_playwright

from moonshot_dashboard.config import DEFAULT_BRANDS, MAX_REVIEWS_PER_PRODUCT, PRODUCTS_PER_BRAND, RAW_DIR


BASE_URL = "https://www.amazon.in"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


@dataclass
class ProductRecord:
    asin: str
    brand: str
    title: str
    product_url: str
    search_query: str
    current_price: float | None
    list_price: float | None
    discount_pct: float | None
    rating: float | None
    review_count: int | None
    badges: str
    bought_recently: str | None
    luggage_type: str | None
    size_segment: str | None
    material: str | None
    scraped_at_utc: str


@dataclass
class ReviewRecord:
    asin: str
    brand: str
    product_title: str
    review_id: str
    author: str | None
    rating: float | None
    title: str | None
    body: str
    review_date: str | None
    verified_purchase: bool
    scraped_at_utc: str


def parse_money(value: str | None) -> float | None:
    if not value:
        return None
    numbers = re.sub(r"[^\d.]", "", value)
    return float(numbers) if numbers else None


def parse_int(value: str | None) -> int | None:
    if not value:
        return None
    cleaned = value.replace(",", "").strip().lower()
    match = re.search(r"(\d+(?:\.\d+)?)", cleaned)
    if not match:
        return None
    number = float(match.group(1))
    if "k" in cleaned:
        number *= 1000
    return int(number)


def parse_rating(value: str | None) -> float | None:
    if not value:
        return None
    match = re.search(r"(\d+(?:\.\d+)?)", value)
    return float(match.group(1)) if match else None


def clean_text(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = re.sub(r"\s+", " ", value).strip()
    return cleaned or None


def infer_luggage_type(title: str) -> str | None:
    title_lower = title.lower()
    if "duffel" in title_lower:
        return "Duffel"
    if "backpack" in title_lower:
        return "Backpack"
    if "set of 3" in title_lower or "set of 2" in title_lower or "set" in title_lower:
        return "Luggage Set"
    if "hard case" in title_lower or "hard side" in title_lower:
        return "Hard Case"
    if "soft" in title_lower:
        return "Soft Case"
    if "trolley" in title_lower or "suitcase" in title_lower:
        return "Suitcase"
    return None


def infer_size_segment(title: str) -> str | None:
    title_lower = title.lower()
    if "cabin" in title_lower or "carry-on" in title_lower or "carry on" in title_lower:
        return "Cabin"
    if "medium" in title_lower:
        return "Medium"
    if "large" in title_lower:
        return "Large"
    if "set" in title_lower:
        return "Multi-size Set"
    return None


def infer_material(title: str) -> str | None:
    title_lower = title.lower()
    for candidate in ["polycarbonate", "polypropylene", "abs", "fabric", "nylon", "aluminium"]:
        if candidate in title_lower:
            return candidate.title()
    return None


async def build_page(browser: Browser) -> Page:
    context = await browser.new_context(
        locale="en-IN",
        user_agent=USER_AGENT,
        viewport={"width": 1440, "height": 2200},
    )
    return await context.new_page()


async def fetch_search_results(page: Page, brand: str, target_products: int) -> list[dict]:
    url = f"{BASE_URL}/s?k={quote_plus(f'{brand} luggage trolley')}"
    await page.goto(url, wait_until="domcontentloaded", timeout=120000)
    await page.wait_for_timeout(3000)
    await page.evaluate("window.scrollTo(0, document.body.scrollHeight * 0.6)")
    await page.wait_for_timeout(1500)
    soup = BeautifulSoup(await page.content(), "lxml")

    products: list[dict] = []
    seen_asins: set[str] = set()

    for item in soup.select('div[data-component-type="s-search-result"]'):
        asin = item.get("data-asin", "").strip()
        if not asin or asin in seen_asins:
            continue
        if "AdHolder" in item.get("class", []):
            continue

        title_node = item.select_one("h2 a span")
        link_node = item.select_one("h2 a")
        image_node = item.select_one("img.s-image")
        title = clean_text(title_node.get_text(" ", strip=True) if title_node else None)
        if not title and image_node:
            title = clean_text(image_node.get("alt"))

        href = link_node.get("href") if link_node else None
        if not href:
            for anchor in item.select("a[href]"):
                candidate = anchor.get("href", "")
                if "/dp/" in candidate:
                    href = candidate
                    break

        if not title or not href or brand.lower() not in title.lower():
            continue

        seen_asins.add(asin)
        badges = [clean_text(node.get_text(" ", strip=True)) for node in item.select(".a-badge-label-inner")]
        bought_node = item.select("span.a-size-base.a-color-secondary")
        bought_recently = None
        for node in bought_node:
            text = clean_text(node.get_text(" ", strip=True))
            if text and "bought" in text.lower():
                bought_recently = text
                break

        review_count_text = None
        review_node = item.select_one("span.a-size-base.s-underline-text")
        if review_node:
            review_count_text = review_node.get_text(" ", strip=True)
        else:
            text_blob = item.get_text(" ", strip=True)
            review_match = re.search(r"\(([\d,\.]+[Kk]?)\)", text_blob)
            if review_match:
                review_count_text = review_match.group(1)

        products.append(
            {
                "asin": asin,
                "brand": brand,
                "title": title,
                "product_url": urljoin(BASE_URL, href.split("?")[0]),
                "search_query": f"{brand} luggage trolley",
                "current_price": parse_money(
                    item.select_one("span.a-price span.a-offscreen").get_text(strip=True)
                    if item.select_one("span.a-price span.a-offscreen")
                    else None
                ),
                "list_price": parse_money(
                    item.select_one("span.a-text-price span.a-offscreen").get_text(strip=True)
                    if item.select_one("span.a-text-price span.a-offscreen")
                    else None
                ),
                "rating": parse_rating(
                    item.select_one("span.a-icon-alt").get_text(" ", strip=True)
                    if item.select_one("span.a-icon-alt")
                    else None
                ),
                "review_count": parse_int(review_count_text),
                "badges": " | ".join([badge for badge in badges if badge]),
                "bought_recently": bought_recently,
            }
        )
        if len(products) >= target_products:
            break

    return products


async def enrich_product(page: Page, product: dict, max_reviews: int) -> tuple[ProductRecord, list[ReviewRecord]]:
    await page.goto(product["product_url"], wait_until="domcontentloaded", timeout=120000)
    await page.wait_for_timeout(2500)
    await page.evaluate("window.scrollTo(0, document.body.scrollHeight * 0.45)")
    await page.wait_for_timeout(1500)
    soup = BeautifulSoup(await page.content(), "lxml")

    title = clean_text(
        soup.select_one("#productTitle").get_text(" ", strip=True)
        if soup.select_one("#productTitle")
        else product["title"]
    ) or product["title"]
    current_price = parse_money(
        soup.select_one("span.a-price span.a-offscreen").get_text(strip=True)
        if soup.select_one("span.a-price span.a-offscreen")
        else None
    ) or product["current_price"]
    list_price = parse_money(
        soup.select_one("span.a-price.a-text-price span.a-offscreen").get_text(strip=True)
        if soup.select_one("span.a-price.a-text-price span.a-offscreen")
        else None
    ) or product["list_price"]
    discount_pct = None
    if current_price and list_price and list_price > current_price:
        discount_pct = round(((list_price - current_price) / list_price) * 100, 2)
    else:
        savings = clean_text(
            soup.select_one("span.savingsPercentage").get_text(" ", strip=True)
            if soup.select_one("span.savingsPercentage")
            else None
        )
        if savings:
            discount_pct = parse_rating(savings)

    rating = parse_rating(
        soup.select_one("#acrPopover .a-icon-alt").get_text(" ", strip=True)
        if soup.select_one("#acrPopover .a-icon-alt")
        else None
    ) or product["rating"]
    review_count = parse_int(
        soup.select_one("#acrCustomerReviewText").get_text(" ", strip=True)
        if soup.select_one("#acrCustomerReviewText")
        else None
    ) or product["review_count"]

    scraped_at = datetime.now(timezone.utc).isoformat()
    product_record = ProductRecord(
        asin=product["asin"],
        brand=product["brand"],
        title=title,
        product_url=product["product_url"],
        search_query=product["search_query"],
        current_price=current_price,
        list_price=list_price,
        discount_pct=discount_pct,
        rating=rating,
        review_count=review_count,
        badges=product["badges"],
        bought_recently=product["bought_recently"],
        luggage_type=infer_luggage_type(title),
        size_segment=infer_size_segment(title),
        material=infer_material(title),
        scraped_at_utc=scraped_at,
    )

    reviews: list[ReviewRecord] = []
    for review in soup.select("li.review[data-hook='review']")[:max_reviews]:
        review_id = review.get("id") or ""
        body = clean_text(
            review.select_one("[data-hook='review-body']").get_text(" ", strip=True)
            if review.select_one("[data-hook='review-body']")
            else None
        )
        if not review_id or not body:
            continue
        reviews.append(
            ReviewRecord(
                asin=product["asin"],
                brand=product["brand"],
                product_title=title,
                review_id=review_id,
                author=clean_text(
                    review.select_one(".a-profile-name").get_text(" ", strip=True)
                    if review.select_one(".a-profile-name")
                    else None
                ),
                rating=parse_rating(
                    review.select_one("[data-hook='review-star-rating'] .a-icon-alt").get_text(" ", strip=True)
                    if review.select_one("[data-hook='review-star-rating'] .a-icon-alt")
                    else None
                ),
                title=clean_text(
                    review.select_one("[data-hook='review-title']").get_text(" ", strip=True)
                    if review.select_one("[data-hook='review-title']")
                    else None
                ),
                body=body,
                review_date=clean_text(
                    review.select_one("[data-hook='review-date']").get_text(" ", strip=True)
                    if review.select_one("[data-hook='review-date']")
                    else None
                ),
                verified_purchase=review.select_one("[data-hook='avp-badge']") is not None,
                scraped_at_utc=scraped_at,
            )
        )

    return product_record, reviews


async def scrape(brands: Iterable[str], products_per_brand: int, max_reviews: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    products_out: list[dict] = []
    reviews_out: list[dict] = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        search_page = await build_page(browser)
        detail_page = await build_page(browser)
        try:
            for brand in brands:
                search_results = await fetch_search_results(search_page, brand, products_per_brand)
                for product in search_results:
                    try:
                        enriched_product, product_reviews = await enrich_product(detail_page, product, max_reviews)
                    except PlaywrightError:
                        await detail_page.context.close()
                        detail_page = await build_page(browser)
                        try:
                            enriched_product, product_reviews = await enrich_product(detail_page, product, max_reviews)
                        except PlaywrightError:
                            continue
                    products_out.append(asdict(enriched_product))
                    reviews_out.extend(asdict(review) for review in product_reviews)
        finally:
            await search_page.context.close()
            await detail_page.context.close()
            await browser.close()

    products_df = pd.DataFrame(products_out).drop_duplicates(subset=["asin"])
    reviews_df = pd.DataFrame(reviews_out).drop_duplicates(subset=["asin", "review_id"])
    return products_df, reviews_df


def save_outputs(products_df: pd.DataFrame, reviews_df: pd.DataFrame) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    products_df.to_csv(RAW_DIR / "products.csv", index=False)
    reviews_df.to_csv(RAW_DIR / "reviews.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape Amazon India luggage listings and top reviews.")
    parser.add_argument("--brands", nargs="+", default=DEFAULT_BRANDS)
    parser.add_argument("--products-per-brand", type=int, default=PRODUCTS_PER_BRAND)
    parser.add_argument("--max-reviews", type=int, default=MAX_REVIEWS_PER_PRODUCT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    products_df, reviews_df = asyncio.run(scrape(args.brands, args.products_per_brand, args.max_reviews))
    save_outputs(products_df, reviews_df)
    print(f"Saved {len(products_df)} products and {len(reviews_df)} reviews to {RAW_DIR}.")


if __name__ == "__main__":
    main()
