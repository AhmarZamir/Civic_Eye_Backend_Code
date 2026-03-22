from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
from supabase import create_client, Client
import os
from dotenv import load_dotenv
import time
import re
from transformers import pipeline



summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# ------------------- Hugging Face API Config -------------------
API_URL_SUMMARY = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
API_URL_SENTIMENT = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment-latest"

# Get API key with proper validation
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
if not HUGGINGFACE_API_KEY or HUGGINGFACE_API_KEY == 'YOUR_HF_API_KEY':
    print(" WARNING: HuggingFace API key not set! Using fallback methods.")
    HUGGINGFACE_API_KEY = None

headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"} if HUGGINGFACE_API_KEY else {}

# ------------------- Configuration -------------------
MAX_ARTICLES = 20

# Initialize Supabase
supabase: Client = create_client(
    os.getenv('SUPABASE_URL'),
    os.getenv('SUPABASE_KEY')
)

def query_huggingface(payload, api_url, retries=5):
    """Improved HuggingFace API call with better error handling"""
    for i in range(retries):
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=60)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 503:
                # Model is loading - wait longer
                error_msg = response.json().get('error', '')
                if 'loading' in error_msg.lower():
                    wait_time = 10 * (i + 1)
                    print(f" Model loading... waiting {wait_time} seconds ({i+1}/{retries})")
                    time.sleep(wait_time)
                else:
                    print(f" API error 503: {error_msg}")
                    break
            elif response.status_code == 422:
                print(f" Validation error: {response.text}")
                break
            else:
                print(f" API error {response.status_code}: {response.text}")
                break
                
        except requests.exceptions.Timeout:
            print(f"⏰ Request timeout ({i+1}/{retries})")
            time.sleep(5)
        except Exception as e:
            print(f" API call failed ({i+1}/{retries}): {e}")
            time.sleep(3)
    
    print(f" Failed after {retries} retries")
    return None
def bart_chunked_summary(text, min_length=60, max_length=150):
    """
    Summarize long text by chunking it into 1024-token pieces,
    using BART CNN. Works with HF API or local pipeline.
    """

    # Clean text
    cleaned = re.sub(r'\s+', ' ', text).strip()

    # If short, summarize directly
    if len(cleaned) < 800:
        try:
            out = summarizer(cleaned, min_length=min_length, max_length=max_length, do_sample=False)
            return out[0]["summary_text"]
        except:
            return cleaned[:300] + "..."

    # ---- TOKEN-SAFE CHUNKING ----
    # BART uses byte-level BPE, roughly 3–6 chars/token → safe chunk = 3500 chars
    CHUNK_SIZE = 3500

    chunks = []
    for i in range(0, len(cleaned), CHUNK_SIZE):
        chunks.append(cleaned[i:i + CHUNK_SIZE])

    print(f" Splitting into {len(chunks)} chunks for BART summarization")

    partial_summaries = []

    # ---- First-pass summarization ----
    for idx, chunk in enumerate(chunks, start=1):
        print(f"📄 Summarizing chunk {idx}/{len(chunks)}")

        if HUGGINGFACE_API_KEY:
            payload = {
                "inputs": chunk,
                "parameters": {
                    "min_length": 40,
                    "max_length": 120,
                    "do_sample": False,
                    "early_stopping": True
                },
                "options": {"wait_for_model": True}
            }
            resp = query_huggingface(payload, API_URL_SUMMARY)
            if resp and isinstance(resp, list) and "summary_text" in resp[0]:
                partial_summaries.append(resp[0]["summary_text"])
            else:
                partial_summaries.append(chunk[:300])
        else:
            # Local pipeline
            try:
                out = summarizer(chunk, min_length=40, max_length=120, do_sample=False)
                partial_summaries.append(out[0]["summary_text"])
            except:
                partial_summaries.append(chunk[:300])

    # ---- Combine partial summaries ----
    merged_text = " ".join(partial_summaries)

    print("🔁 Running second-pass summary (merge summary)")

    # ---- Final summarization ----
    if HUGGINGFACE_API_KEY:
        final_payload = {
            "inputs": merged_text[:3500],
            "parameters": {
                "min_length": min_length,
                "max_length": max_length,
                "do_sample": False,
                "early_stopping": True
            },
            "options": {"wait_for_model": True}
        }
        resp = query_huggingface(final_payload, API_URL_SUMMARY)
        if resp and isinstance(resp, list) and "summary_text" in resp[0]:
            return resp[0]["summary_text"]

    # Local final summary fallback
    try:
        out = summarizer(merged_text, min_length=min_length, max_length=max_length, do_sample=False)
        return out[0]["summary_text"]
    except:
        return merged_text[:300] + "..."



def generate_summary(text):
    if not text:
        return ""

    try:
        return bart_chunked_summary(text)
    except Exception as e:
        print(f"⚠️ Summary error: {e}")
        return text[:200] + "..."



def analyze_sentiment(text):
    """Analyze sentiment using HuggingFace API (Twitter-RoBERTa version)"""
    if HUGGINGFACE_API_KEY:
        sentiment_resp = query_huggingface({"inputs": text[:512]}, API_URL_SENTIMENT)
        
        if sentiment_resp and isinstance(sentiment_resp, list):
            # CardiffNLP model returns nested list
            predictions = sentiment_resp[0]
            if isinstance(predictions, list) and len(predictions) > 0:
                top = max(predictions, key=lambda x: x.get("score", 0))
                label = top.get("label", "neutral").upper()
                score = round(top.get("score", 0.5), 2)

                if "POS" in label:
                    return "POSITIVE", score
                elif "NEG" in label:
                    return "NEGATIVE", score
                else:
                    return "NEUTRAL", score

    # 🔸 Fallback if HuggingFace call fails
    text_lower = text.lower()
    positive_words = ['good', 'great', 'excellent', 'positive', 'win', 'success', 'happy', 'achievement', 'breakthrough', 'celebrate']
    negative_words = ['bad', 'poor', 'negative', 'attack', 'death', 'protest', 'crisis', 'violence', 'disaster', 'accident']

    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)

    if positive_count > negative_count:
        return "POSITIVE", 0.7
    elif negative_count > positive_count:
        return "NEGATIVE", 0.7
    else:
        return "NEUTRAL", 0.5

def scrape_geo_news():
    """Scrape article URLs from Geo.tv"""
    try:
        data = requests.get("https://www.geo.tv/latest-news", timeout=15)
        soup = BeautifulSoup(data.content, "html.parser")
        
        article_urls = []
        
        # Method 1: Find lazy loaded images with data-src
        lazy_images = soup.find_all('img', {'data-src': True})
        print(f" Found {len(lazy_images)} lazy loaded images")
        
        for img in lazy_images:
            parent_link = img.find_parent('a')
            if parent_link and parent_link.get('href'):
                href = parent_link['href']
                # Filter out video content and ensure it's a news article
                if '/shows/' not in href and '/video/' not in href:
                    if href.startswith('/'):
                        href = 'https://www.geo.tv' + href
                    if href not in article_urls and ('/latest/' in href or '/news/' in href):
                        article_urls.append(href)
        
        # Remove duplicates and limit
        article_urls = list(set(article_urls))[:MAX_ARTICLES]
        print(f" Total unique articles: {len(article_urls)}")
        
        return article_urls
        
    except Exception as e:
        print(f" Error scraping URLs: {e}")
        return []

def extract_article_content(soup, url):
    """Extract content from article page with multiple fallback methods"""
    # Extract heading
    heading = "No Title"
    heading_selectors = ['h1', '.story-title', '.entry-title', 'title', 'h2', '.heading']
    for selector in heading_selectors:
        element = soup.select_one(selector) if '.' in selector else soup.find(selector)
        if element and element.get_text(strip=True):
            heading = element.get_text(strip=True)
            break
    
    # Extract image - FIRST find the article container to avoid global images
    image_url = ""
    
    # Find the main article content area first
    article_container = None
    article_selectors = ['.story-detail', '.entry-content', '.story-content', 'article', '.detail-content', '.news-content', '.post-content']
    
    for selector in article_selectors:
        container = soup.select_one(selector)
        if container:
            article_container = container
            break
    
    # If we found an article container, search for image within it
    if article_container:
        image_selectors = [
            'img[data-src]:not([class*="logo"]):not([class*="icon"])',
            'img[src]:not([class*="logo"]):not([class*="icon"])',
            'img[class*="featured"]',
            'img[class*="thumbnail"]',
            'img'
        ]
        
        for selector in image_selectors:
            img = article_container.select_one(selector)
            if img:
                image_url = img.get('data-src') or img.get('src') or ""
                if image_url and len(image_url) > 5:  # Ensure it's a real URL
                    if image_url.startswith('//'):
                        image_url = 'https:' + image_url
                    elif image_url.startswith('/'):
                        image_url = 'https://www.geo.tv' + image_url
                    # Verify it's not a tracking pixel or banner
                    if not any(x in image_url.lower() for x in ['pixel', '1x1', 'banner', 'ad', 'logo']):
                        break
    
    # Fallback: if no image in article container, try page-level selectors (but more specific)
    if not image_url:
        image_selectors = [
            'meta[property="og:image"]',
            'img[data-src]',
            '.story-image img',
            'img[src*="upload"]',
            '.wp-post-image'
        ]
        
        for selector in image_selectors:
            if selector.startswith('meta'):
                meta = soup.select_one(selector)
                if meta and meta.get('content'):
                    image_url = meta.get('content')
            else:
                img = soup.select_one(selector)
                if img:
                    image_url = img.get('data-src') or img.get('src') or ""
            
            if image_url:
                if image_url.startswith('//'):
                    image_url = 'https:' + image_url
                elif image_url.startswith('/'):
                    image_url = 'https://www.geo.tv' + image_url
                break
    
    # Extract content - try multiple content areas
    full_content = ""
    content_selectors = [
        '.story-detail',
        '.entry-content',
        '.story-content',
        'article',
        '[class*="content"]',
        '.detail-content',
        '.news-content',
        '.post-content'
    ]
    
    for selector in content_selectors:
        content_div = soup.select_one(selector)
        if content_div:
            # Remove script and style elements
            for element in content_div(['script', 'style', 'nav', 'header', 'footer', 'aside', '.social-share', '.comments']):
                element.decompose()
            
            # Get all text content
            content_text = content_div.get_text(separator=' ', strip=True)
            # Clean up extra whitespace
            content_text = re.sub(r'\s+', ' ', content_text)
            
            if len(content_text) > 150:  # Only use if we have substantial content
                full_content = content_text
                print(f"Found content with selector: {selector} - {len(content_text)} chars")
                break
    
    # If no substantial content found, try meta description
    if not full_content or len(full_content) < 100:
        meta_desc = soup.find('meta', {'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            full_content = meta_desc.get('content')
            print(" Using meta description as content")
        else:
            # Try Open Graph description
            meta_og = soup.find('meta', {'property': 'og:description'})
            if meta_og and meta_og.get('content'):
                full_content = meta_og.get('content')
                print(" Using Open Graph description as content")
            else:
                # Last resort: use heading
                full_content = heading
                print(" Using heading as fallback content")
    
    return heading, image_url, full_content

def process_article(url):
    """Extract data from a single article with robust error handling"""
    try:
        print(f" Processing: {url}")
        
        # Get the article page with custom headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code != 200:
            print(f" Failed to fetch URL: {response.status_code}")
            return None
            
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Extract content using our robust function
        heading, image_url, full_content = extract_article_content(soup, url)
        
        print(f" Heading: {heading}")
        print(f" Image: {image_url[:50]}..." if image_url else "🖼️ No image")
        print(f" Original content: {len(full_content)} chars")

        # AI PROCESSING
        print(f" Starting AI processing...")
        
        # Generate summary
        summary = generate_summary(full_content)
        print(f" Final Summary ({len(summary)} chars): {summary}")
        
        # Analyze sentiment
        sentiment, confidence = analyze_sentiment(summary if summary else full_content)
        print(f" Sentiment: {sentiment} (confidence: {confidence})")

        return {
            "url": url,
            "picture": image_url,
            "heading": heading,
            "blog": summary,
            "sentiment": sentiment,
            "confidence": confidence
        }
        
    except Exception as e:
        print(f" Error processing {url}: {str(e)[:100]}...")
        import traceback
        traceback.print_exc()
        return None

def is_article_exists(url):
    """Check if article already exists in database"""
    try:
        response = supabase.table("headlinesdata").select("url").eq("url", url).execute()
        return len(response.data) > 0
    except Exception as e:
        print(f" Error checking existing article: {e}")
        return False

def save_to_supabase(article_data):
    """Save article data to Supabase with duplicate check"""
    try:
        # Check if article already exists
        if is_article_exists(article_data["url"]):
            print(f"Skipping duplicate: {article_data['heading'][:50]}...")
            return False
        
        # Insert new article
        response = supabase.table("headlinesdata").insert({
            "url": article_data["url"],
            "picture": article_data["picture"],
            "heading": article_data["heading"],
            "blog": article_data["blog"],
            "sentiment": article_data["sentiment"],
            "confidence": article_data["confidence"]
        }).execute()
        
        print(f" Saved NEW article: {article_data['heading'][:50]}...")
        return True
        
    except Exception as e:
        print(f" Error saving to Supabase: {e}")
        return False

@app.route('/scrape', methods=['GET'])
def scrape_and_save():
    """Main endpoint that scrapes and saves to Supabase"""
    try:
        print(" Starting scraping process...")
        print(f" HuggingFace API: {' Enabled' if HUGGINGFACE_API_KEY else '❌ Disabled (using fallback)'}")
        
        # Get article URLs
        urls = scrape_geo_news()
        
        if not urls:
            return jsonify({"error": "No articles found"}), 400
        
        # Check existing articles
        print("Checking for existing articles...")
        existing_urls = []
        for url in urls:
            if is_article_exists(url):
                existing_urls.append(url)
        
        new_urls = [url for url in urls if url not in existing_urls]
        print(f" Found {len(existing_urls)} existing articles, {len(new_urls)} new articles")
        
        # Process only NEW articles
        saved_count = 0
        failed_count = 0
        
        for i, url in enumerate(new_urls):
            print(f"Processing {i+1}/{len(new_urls)}: {url}")
            
            # Add small delay to avoid overwhelming
            if i > 0:
                time.sleep(2)
            
            article_data = process_article(url)
            
            if article_data:
                if save_to_supabase(article_data):
                    saved_count += 1
                else:
                    failed_count += 1
                    print(f" Failed to save: {url}")
            else:
                failed_count += 1
                print(f"Failed to process: {url}")
        
        return jsonify({
            "status": "success",
            "message": f"Found {len(urls)} articles, {len(existing_urls)} already existed, saved {saved_count} new articles",
            "total_found": len(urls),
            "already_existed": len(existing_urls),
            "newly_saved": saved_count,
            "failed_to_process": failed_count,
            "ai_processing": True,
            "huggingface_enabled": bool(HUGGINGFACE_API_KEY)
        })
        
    except Exception as e:
        print(f"Scraping error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/news', methods=['GET'])
def get_news():
    """Endpoint to get all news from database"""
    try:
        limit = request.args.get('limit', default=20, type=int)
        sentiment_filter = request.args.get('sentiment', default=None, type=str)
        
        query = supabase.table("headlinesdata").select("*").order("timestamp", desc=True).limit(limit)
        
        if sentiment_filter:
            query = query.eq("sentiment", sentiment_filter.upper())
        
        response = query.execute()
        
        return jsonify({
            "status": "success",
            "count": len(response.data),
            "data": response.data
        })
        
    except Exception as e:
        print(f"Error fetching news: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return jsonify({
        "message": "AI News Scraper API is running",
        "endpoints": {
            "/scrape": "Scrape and save news with AI analysis",
            "/news": "Get all news (use ?limit=10&sentiment=POSITIVE)"
        }
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
    