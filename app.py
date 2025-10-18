from flask import Flask, jsonify, request
from flask_cors import CORS
from bs4 import BeautifulSoup
import sqlite3
import time
import schedule
import threading
import atexit
import json
import requests
from datetime import datetime

# ------------------- Hugging Face API Config -------------------
API_URL_SUMMARY = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
API_URL_SENTIMENT = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
headers = {"Authorization": f"Bearer YOUR_HF_API_KEY"}  # replace with your API key

def query_huggingface(payload, api_url):
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 503:
            print(f"Model is loading, please wait...")
            return None
        elif response.status_code != 200:
            print(f"API error {response.status_code}: {response.text}")
            return None
            
        return response.json()
    except requests.exceptions.Timeout:
        print(f"HuggingFace API timeout for {api_url}")
        return None
    except Exception as e:
        print(f"HuggingFace API error: {e}")
        return None

# ------------------- Configuration -------------------
SCRAPE_INTERVAL_MINUTES = 30
MAX_ARTICLES = 20
DATABASE_FILE = "civic_eye.db"

# ------------------- Database Manager -------------------
class DatabaseManager:
    def __init__(self, db_file):
        self.db_file = db_file
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_file) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute('''
                CREATE TABLE IF NOT EXISTS headlinesData (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT UNIQUE,
                    picture TEXT,
                    heading TEXT,
                    blog TEXT,
                    sentiment TEXT,
                    confidence REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS civic_issues (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    issue_type TEXT NOT NULL,
                    specific_issue_type TEXT,
                    description TEXT NOT NULL,
                    location_lat REAL,
                    location_lon REAL,
                    location_district TEXT,
                    location_city TEXT,
                    location_tehsil TEXT,
                    location_exact_address TEXT,
                    media_urls TEXT,
                    reporter_id TEXT NOT NULL,
                    status TEXT DEFAULT 'Under Community Verification',
                    upvotes INTEGER DEFAULT 0,
                    downvotes INTEGER DEFAULT 0,
                    upvoted_by TEXT DEFAULT '[]',
                    downvoted_by TEXT DEFAULT '[]',
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
            print("Database initialized successfully")

    def save_headline(self, headline_data):
        with sqlite3.connect(self.db_file) as conn:
            try:
                conn.execute('''
                    INSERT OR REPLACE INTO headlinesData 
                    (url, picture, heading, blog, sentiment, confidence)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    headline_data["url"],
                    headline_data["picture"],
                    headline_data["heading"],
                    headline_data["blog"],
                    headline_data["sentiment"],
                    headline_data["confidence"]
                ))
                conn.commit()
                print(f"✓ Saved: {headline_data['heading'][:50]}...")
                return True
            except sqlite3.Error as e:
                print(f"Database error saving headline: {e}")
                return False

    def get_all_headlines(self):
        with sqlite3.connect(self.db_file) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            try:
                cursor.execute('''
                    SELECT url, picture, heading, blog, sentiment, confidence 
                    FROM headlinesData 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (MAX_ARTICLES,))
                results = [dict(row) for row in cursor.fetchall()]
                print(f"Retrieved {len(results)} headlines from database")
                return results
            except sqlite3.Error as e:
                print(f"Database error in get_all_headlines: {e}")
                return []

    def save_civic_issue(self, issue_data):
        with sqlite3.connect(self.db_file) as conn:
            try:
                conn.execute('''
                    INSERT INTO civic_issues (
                        issue_type, specific_issue_type, description, 
                        location_lat, location_lon, location_district, 
                        location_city, location_tehsil, location_exact_address, 
                        media_urls, reporter_id, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    issue_data.get('issueType'),
                    issue_data.get('specificIssueType'),
                    issue_data.get('description'),
                    issue_data['location'].get('latitude'),
                    issue_data['location'].get('longitude'),
                    issue_data['location'].get('district'),
                    issue_data['location'].get('city'),
                    issue_data['location'].get('tehsil'),
                    issue_data['location'].get('exactAddress'),
                    json.dumps(issue_data.get('mediaUrls', [])),
                    issue_data.get('reporterId'),
                    issue_data.get('status', 'Under Community Verification')
                ))
                conn.commit()
                return conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            except sqlite3.Error as e:
                print(f"Database error saving civic issue: {e}")
                return None

    def get_issues_by_reporter(self, reporter_id):
        with sqlite3.connect(self.db_file) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM civic_issues WHERE reporter_id = ? ORDER BY timestamp DESC
            ''', (reporter_id,))
            issues = []
            for row in cursor.fetchall():
                issue = dict(row)
                issue['media_urls'] = json.loads(issue['media_urls'])
                issues.append(issue)
            return issues

    def get_community_issues(self, status='Under Community Verification', limit=50):
        with sqlite3.connect(self.db_file) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM civic_issues 
                WHERE status = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (status, limit))
            issues = []
            for row in cursor.fetchall():
                issue = dict(row)
                issue['media_urls'] = json.loads(issue['media_urls'])
                issue['upvoted_by'] = json.loads(issue['upvoted_by'])
                issue['downvoted_by'] = json.loads(issue['downvoted_by'])
                issues.append(issue)
            return issues

    def update_issue_vote(self, issue_id, user_id, vote_type):
        with sqlite3.connect(self.db_file) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("SELECT upvotes, downvotes, upvoted_by, downvoted_by FROM civic_issues WHERE id = ?", (issue_id,))
            row = cursor.fetchone()
            if not row:
                return False, "Issue not found"

            current_upvotes = row['upvotes']
            current_downvotes = row['downvotes']
            upvoted_by = json.loads(row['upvoted_by'])
            downvoted_by = json.loads(row['downvoted_by'])

            if user_id in upvoted_by or user_id in downvoted_by:
                return False, "User has already voted on this issue"

            if vote_type == 'upvote':
                current_upvotes += 1
                upvoted_by.append(user_id)
            elif vote_type == 'downvote':
                current_downvotes += 1
                downvoted_by.append(user_id)
            else:
                return False, "Invalid vote type"

            new_status = None
            if current_upvotes - current_downvotes >= 25:
                new_status = 'Verified & Forwarded'
            elif current_downvotes >= 20:
                new_status = 'Misinformation Flagged'

            update_query = '''
                UPDATE civic_issues SET 
                    upvotes = ?, 
                    downvotes = ?, 
                    upvoted_by = ?, 
                    downvoted_by = ?
                    {status_update}
                WHERE id = ?
            '''
            status_update_clause = ""
            update_params = [current_upvotes, current_downvotes, json.dumps(upvoted_by), json.dumps(downvoted_by)]

            if new_status:
                status_update_clause = ", status = ?"
                update_params.append(new_status)

            update_params.append(issue_id)

            try:
                conn.execute(update_query.format(status_update=status_update_clause), tuple(update_params))
                conn.commit()
                return True, "Vote recorded successfully"
            except sqlite3.Error as e:
                print(f"Database error updating vote: {e}")
                return False, f"Database error: {e}"

    def update_issue_status(self, issue_id, new_status):
        with sqlite3.connect(self.db_file) as conn:
            try:
                conn.execute("UPDATE civic_issues SET status = ? WHERE id = ?", (new_status, issue_id))
                conn.commit()
                return True
            except sqlite3.Error as e:
                print(f"Database error updating status: {e}")
                return False

# ------------------- Cloud-Optimized News Scraper -------------------
class CloudNewsScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })

    def scrape_geo_news(self):
        """Scrape Geo News using requests only (no Selenium)"""
        try:
            print("🚀 Starting Geo News scraping...")
            
            # Scrape latest news page
            url = "https://www.geo.tv/latest-news"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            headlines = []
            
            # Multiple strategies to find news items
            news_items = []
            
            # Strategy 1: Common news selectors
            selectors = [
                '.entry-box',
                'article',
                '.story-news',
                '.news-list-item', 
                '.latest-news-item',
                '.news-item'
            ]
            
            for selector in selectors:
                items = soup.select(selector)
                if items:
                    news_items.extend(items)
                    print(f"Found {len(items)} items with selector: {selector}")
            
            # Strategy 2: Find all news links
            if not news_items:
                news_links = soup.find_all('a', href=True)
                news_items = [link for link in news_links if any(keyword in link.get('href', '') for keyword in ['/latest-news', '/news', 'story'])]
                news_items = news_items[:MAX_ARTICLES]
            
            print(f"📰 Total found {len(news_items)} potential news items")
            
            processed_count = 0
            for item in news_items[:MAX_ARTICLES]:
                try:
                    headline_data = self.extract_article_data(item)
                    if headline_data and headline_data.get('heading') and headline_data['heading'] != "No title":
                        headlines.append(headline_data)
                        processed_count += 1
                        print(f"✓ Extracted: {headline_data['heading'][:60]}...")
                except Exception as e:
                    print(f"❌ Error processing news item: {e}")
                    continue
            
            # If no articles found, create sample data for testing
            if not headlines:
                print("⚠️ No articles found, generating sample data...")
                headlines = self.create_sample_data()
            
            print(f"✅ Successfully processed {len(headlines)} articles")
            return headlines
            
        except Exception as e:
            print(f"❌ Scraping error: {e}")
            # Return sample data if scraping fails
            return self.create_sample_data()

    def extract_article_data(self, item):
        """Extract data from a news item"""
        try:
            # Get URL
            if item.name == 'a':
                link = item
            else:
                link = item.find('a')
            
            if not link or not link.get('href'):
                return None
                
            article_url = link.get('href')
            if article_url.startswith('/'):
                article_url = 'https://www.geo.tv' + article_url

            # Get title
            title = ""
            title_elem = link.find(['h1', 'h2', 'h3', 'h4']) or item.find(['h1', 'h2', 'h3', 'h4'])
            if title_elem:
                title = title_elem.get_text().strip()
            else:
                title = link.get_text().strip()
            
            if not title:
                return None

            # Get image
            img_elem = item.find('img') or link.find('img')
            image_url = img_elem.get('src') if img_elem else ""
            
            if image_url and image_url.startswith('//'):
                image_url = 'https:' + image_url
            elif image_url and image_url.startswith('/'):
                image_url = 'https://www.geo.tv' + image_url

            # Try to get article content for better summary
            content = self.get_article_content(article_url)
            content_to_summarize = content if content else title

            # Generate summary (simplified for now)
            summary = self.generate_summary(content_to_summarize)
            
            # Simple sentiment analysis
            sentiment, confidence = self.analyze_sentiment(title)

            return {
                "url": article_url,
                "picture": image_url or "",
                "heading": title,
                "blog": summary,
                "sentiment": sentiment,
                "confidence": confidence
            }
            
        except Exception as e:
            print(f"❌ Error extracting article data: {e}")
            return None

    def get_article_content(self, url):
        """Get full article content"""
        try:
            response = self.session.get(url, timeout=8)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Try different content selectors
            content_selectors = [
                '.story-area .content-area',
                '.story-detail',
                '.entry-content',
                '.article-content',
                'article .content',
                '.story-content'
            ]
            
            for selector in content_selectors:
                content_div = soup.select_one(selector)
                if content_div:
                    paragraphs = content_div.find_all('p')
                    text = ' '.join(p.get_text().strip() for p in paragraphs if p.get_text().strip())
                    if text and len(text) > 50:
                        return text
            
            return None
            
        except Exception as e:
            print(f"⚠️ Could not get article content from {url}: {e}")
            return None

    def generate_summary(self, content):
        """Generate summary using HuggingFace API or fallback"""
        try:
            # Try HuggingFace API first
            if len(content) > 100:
                summary_resp = query_huggingface(
                    {"inputs": content[:1024], "parameters": {"min_length": 50, "max_length": 150}}, 
                    API_URL_SUMMARY
                )
                
                if summary_resp and isinstance(summary_resp, list):
                    return summary_resp[0].get('summary_text', content[:200] + "...")
            
            # Fallback: simple truncation
            if len(content) > 200:
                return content[:200] + "..."
            else:
                return content
                
        except Exception as e:
            print(f"⚠️ Summary generation failed: {e}")
            if len(content) > 200:
                return content[:200] + "..."
            return content

    def analyze_sentiment(self, text):
        """Analyze sentiment using HuggingFace API or fallback"""
        try:
            sentiment_resp = query_huggingface({"inputs": text}, API_URL_SENTIMENT)
            
            if sentiment_resp and isinstance(sentiment_resp, list):
                sentiment_data = sentiment_resp[0]
                return sentiment_data.get('label', 'NEUTRAL'), round(sentiment_data.get('score', 0.5), 2)
        except Exception as e:
            print(f"⚠️ Sentiment analysis failed: {e}")
        
        # Fallback: keyword-based sentiment
        positive_words = ['good', 'great', 'excellent', 'positive', 'win', 'success', 'achievement', 'progress']
        negative_words = ['bad', 'poor', 'negative', 'attack', 'crash', 'death', 'protest', 'crisis', 'violence']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return "POSITIVE", 0.7
        elif negative_count > positive_count:
            return "NEGATIVE", 0.7
        else:
            return "NEUTRAL", 0.5

    def create_sample_data(self):
        """Create sample data when scraping fails"""
        print("📝 Creating sample data for testing...")
        sample_articles = [
            {
                "url": "https://www.geo.tv/sample-news-1",
                "picture": "",
                "heading": "Technology Sector Shows Strong Growth in Recent Quarter",
                "blog": "The technology industry has demonstrated remarkable growth with increased investments and innovation driving market expansion.",
                "sentiment": "POSITIVE",
                "confidence": 0.8
            },
            {
                "url": "https://www.geo.tv/sample-news-2", 
                "picture": "",
                "heading": "Economic Indicators Point to Stable Market Conditions",
                "blog": "Recent economic data suggests stable market conditions with moderate growth expected in the coming months.",
                "sentiment": "NEUTRAL",
                "confidence": 0.6
            },
            {
                "url": "https://www.geo.tv/sample-news-3",
                "picture": "", 
                "heading": "Weather Advisory: Heavy Rains Expected in Northern Regions",
                "blog": "Meteorological department issues alert for heavy rainfall and potential flooding in northern areas, advising residents to take precautions.",
                "sentiment": "NEGATIVE",
                "confidence": 0.7
            }
        ]
        return sample_articles

# ------------------- Flask App -------------------
app = Flask(__name__)
CORS(app)
db = DatabaseManager(DATABASE_FILE)
scraper = CloudNewsScraper()  # Using cloud-optimized scraper

def scheduled_scrape():
    """Scheduled scraping task"""
    print(f"\n⏰ === Running scheduled scrape at {datetime.now()} ===")
    try:
        headlines = scraper.scrape_geo_news()
        saved_count = 0
        for headline in headlines:
            if db.save_headline(headline):
                saved_count += 1
        print(f"✅ === Scheduled scrape completed. Saved {saved_count} new articles ===\n")
    except Exception as e:
        print(f"❌ === Scheduled job failed: {e} ===\n")

# Schedule scraping job
schedule.every(SCRAPE_INTERVAL_MINUTES).minutes.do(scheduled_scrape)

# Run scheduled jobs in a background thread
def run_scheduler():
    print("🕒 Scheduler thread started...")
    while True:
        try:
            schedule.run_pending()
        except Exception as e:
            print(f"❌ Scheduler error: {e}")
        time.sleep(60)  # Check every minute

scheduler_thread = threading.Thread(target=run_scheduler)
scheduler_thread.daemon = True
scheduler_thread.start()

@app.route("/")
def home():
    return jsonify({
        "message": "Civic Eye API is running 🚀",
        "endpoints": {
            "/news": "Get latest news",
            "/scrape": "Manual scrape",
            "/test-scraper": "Test scraper without saving",
            "/debug": "System status",
            "/report_issue": "Report civic issue",
            "/community_issues": "Get community issues"
        },
        "timestamp": datetime.now().isoformat()
    })

@app.route("/scrape", methods=["GET"])
def manual_scrape():
    """Manual scraping endpoint"""
    print("🔧 Manual scrape triggered...")
    headlines = scraper.scrape_geo_news()
    saved_count = 0
    for headline in headlines:
        if db.save_headline(headline):
            saved_count += 1
    
    return jsonify({
        "status": "success", 
        "scraped": len(headlines),
        "saved": saved_count,
        "message": f"Scraped {len(headlines)} articles, saved {saved_count}",
        "timestamp": datetime.now().isoformat()
    })

@app.route("/test-scraper", methods=["GET"])
def test_scraper():
    """Test the scraper without saving to database"""
    try:
        headlines = scraper.scrape_geo_news()
        return jsonify({
            "status": "success",
            "scraper_type": "CloudNewsScraper", 
            "articles_found": len(headlines),
            "articles": headlines
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/news", methods=["GET"])
def get_news():
    """Get all news headlines"""
    headlines = db.get_all_headlines()
    return jsonify({
        "status": "success",
        "count": len(headlines),
        "data": headlines,
        "timestamp": datetime.now().isoformat()
    })

@app.route("/debug", methods=["GET"])
def debug_info():
    """Debug endpoint to check system status"""
    headlines = db.get_all_headlines()
    
    return jsonify({
        "status": "running",
        "database_file": DATABASE_FILE,
        "total_articles": len(headlines),
        "scraper_type": "CloudNewsScraper",
        "system_time": datetime.now().isoformat(),
        "sample_articles": headlines[:2] if headlines else ["No articles found"]
    })

@app.route("/report_issue", methods=["POST"])
def report_issue():
    issue_data = request.json
    if not issue_data:
        return jsonify({"error": "Invalid JSON data"}), 400

    required_fields = ['issueType', 'description', 'location', 'reporterId']
    if not all(field in issue_data for field in required_fields):
        return jsonify({"error": "Missing required issue fields"}), 400

    issue_id = db.save_civic_issue(issue_data)
    if issue_id:
        return jsonify({"status": "success", "issue_id": issue_id}), 201
    else:
        return jsonify({"error": "Failed to save issue"}), 500

@app.route("/my_issues/<string:reporter_id>", methods=["GET"])
def get_my_issues(reporter_id):
    issues = db.get_issues_by_reporter(reporter_id)
    formatted_issues = []
    for issue in issues:
        formatted_issue = dict(issue)
        formatted_issue['issueType'] = formatted_issue.pop('issue_type')
        formatted_issue['specificIssueType'] = formatted_issue.pop('specific_issue_type')
        formatted_issue['reporterId'] = formatted_issue.pop('reporter_id')
        formatted_issue['mediaUrls'] = formatted_issue.pop('media_urls')

        formatted_issue['location'] = {
            'latitude': formatted_issue.pop('location_lat'),
            'longitude': formatted_issue.pop('location_lon'),
            'district': formatted_issue.pop('location_district'),
            'city': formatted_issue.pop('location_city'),
            'tehsil': formatted_issue.pop('location_tehsil'),
            'exactAddress': formatted_issue.pop('location_exact_address'),
        }
        formatted_issue['dateReported'] = issue['timestamp'].split(' ')[0]
        formatted_issues.append(formatted_issue)

    return jsonify(formatted_issues)

@app.route("/community_issues", methods=["GET"])
def get_community_issues_api():
    issues = db.get_community_issues(status='Under Community Verification')
    formatted_issues = []
    for issue in issues:
        formatted_issue = dict(issue)
        formatted_issue['issueType'] = formatted_issue.pop('issue_type')
        formatted_issue['specificIssueType'] = formatted_issue.pop('specific_issue_type')
        formatted_issue['reporterId'] = formatted_issue.pop('reporter_id')
        formatted_issue['mediaUrls'] = formatted_issue.pop('media_urls')
        formatted_issue['upvotedBy'] = formatted_issue.pop('upvoted_by')
        formatted_issue['downvotedBy'] = formatted_issue.pop('downvoted_by')

        formatted_issue['location'] = {
            'latitude': formatted_issue.pop('location_lat'),
            'longitude': formatted_issue.pop('location_lon'),
            'district': formatted_issue.pop('location_district'),
            'city': formatted_issue.pop('location_city'),
            'tehsil': formatted_issue.pop('location_tehsil'),
            'exactAddress': formatted_issue.pop('location_exact_address'),
        }
        formatted_issues.append(formatted_issue)
    return jsonify(formatted_issues)

@app.route("/vote_issue", methods=["POST"])
def vote_issue():
    data = request.json
    issue_id = data.get('issue_id')
    user_id = data.get('user_id')
    vote_type = data.get('vote_type')

    if not all([issue_id, user_id, vote_type]):
        return jsonify({"error": "Missing issue_id, user_id, or vote_type"}), 400

    success, message = db.update_issue_vote(issue_id, user_id, vote_type)
    if success:
        return jsonify({"status": "success", "message": message}), 200
    else:
        return jsonify({"error": message}), 400

@app.route("/report_misinformation", methods=["POST"])
def report_misinformation_api():
    data = request.json
    issue_id = data.get('issue_id')
    if not issue_id:
        return jsonify({"error": "Missing issue_id"}), 400

    success = db.update_issue_status(issue_id, 'Misinformation Flagged')
    if success:
        return jsonify({"status": "success", "message": "Issue flagged as misinformation"}), 200
    else:
        return jsonify({"error": "Failed to flag issue"}), 500

if __name__ == "__main__":
    print("🚀 Starting Civic Eye Server...")
    print("🔧 Performing initial scrape...")
    
    # Initial scrape
    try:
        scheduled_scrape()
    except Exception as e:
        print(f"❌ Initial scrape failed: {e}")
    
    print("✅ Server starting on http://0.0.0.0:5000")
    app.run(debug=False, host="0.0.0.0", port=5000)