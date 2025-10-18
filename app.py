from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
from flask import Flask, jsonify, request
from flask_cors import CORS
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import sqlite3
import time
import schedule
import threading
import atexit
import json
import requests

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
SCRAPE_INTERVAL_MINUTES = 30  # 3 hours
MAX_ARTICLES = 50
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
            except sqlite3.Error as e:
                print(f"Database error saving headline: {e}")

    def get_all_headlines(self):
        with sqlite3.connect(self.db_file) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT url, picture, heading, blog, sentiment, confidence t, confidence 
                FROM headlinesData 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (MAX_ARTICLES,))
            return [dict(row) for row in cursor.fetchall()]
        

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


# ------------------- News Scraper -------------------
class NewsScraper:
    def __init__(self):
        self.driver = None

    def initialize_driver(self):
     try:
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from webdriver_manager.chrome import ChromeDriverManager
        
        options = Options()
        options.add_argument('--headless=new')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option('excludeSwitches', ['enable-automation'])
        options.add_experimental_option('useAutomationExtension', False)
        
        # Set user agent to avoid detection
        options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        self.driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=options
        )
        
        # Additional anti-detection measures
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
     except Exception as e:
         print(f"Chrome driver initialization failed: {e}")
        # Fallback: you might want to use requests instead
         self.driver = None

    
    
    
    def scrape_geo_news(self):
     if not self.driver:
        self.initialize_driver()
    
     if not self.driver:
        print("Selenium not available, falling back to requests")
        return self.scrape_geo_news_requests()  # You'll need to implement this fallback

     try:
        self.driver.get("https://www.geo.tv")
        time.sleep(6)

        image_elements = self.driver.find_elements(By.CSS_SELECTOR, "img.lazyloaded")
        hrefs = list(set([
            img.find_element(By.XPATH, "./ancestor::a[1]").get_attribute("href")
            for img in image_elements
            if img.find_elements(By.XPATH, "./ancestor::a[1]")
        ]))

        print(f"Found {len(hrefs)} unique news links")
        headlines = []

        for href in hrefs[:MAX_ARTICLES]:
            try:
                headline = self.process_article(href)
                if headline:
                    headlines.append(headline)
            except Exception as e:
                print(f"Error processing {href}: {e}")

        return headlines
     except Exception as e:
        print(f"Scraping error: {e}")
        return []
     finally:
        if self.driver:
            self.driver.quit()
            self.driver = None
    
    
    def scrape_geo_news_requests(self):
    # """Fallback scraper using requests and BeautifulSoup"""
     try:
        import requests
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        response = requests.get('https://www.geo.tv/latest-news', headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        headlines = []
        
        # Find news articles - you'll need to adjust selectors based on Geo.tv's structure
        articles = soup.find_all('article') or soup.find_all('div', class_='news-item')
        
        for article in articles[:MAX_ARTICLES]:
            try:
                link = article.find('a')
                if not link:
                    continue
                    
                url = link.get('href')
                if url and not url.startswith('http'):
                    url = 'https://www.geo.tv' + url
                
                title_elem = article.find(['h1', 'h2', 'h3'])
                title = title_elem.get_text().strip() if title_elem else "No title"
                
                img_elem = article.find('img')
                image = img_elem.get('src') if img_elem else None
                
                if url and title != "No title":
                    # For fallback, use title as summary
                    headline_data = {
                        "url": url,
                        "picture": image or "",
                        "heading": title,
                        "blog": title,  # Using title as placeholder
                        "sentiment": "NEUTRAL",
                        "confidence": 0.5
                    }
                    headlines.append(headline_data)
                    
            except Exception as e:
                print(f"Error processing article: {e}")
                continue
        
        print(f"Requests fallback found {len(headlines)} articles")
        return headlines
        
     except Exception as e:
        print(f"Requests fallback scraping error: {e}")
        return []
    
    

    
    
    

    
    
    

   

    
    
    
    

    
    

    
    
    
    
    
    
    

    
    
    
    
    
    
    
    

    def process_article(self, url):
        self.driver.get(url)
        time.sleep(2)
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')

        heading = soup.find("div", class_="heading_H").find("h1").text.strip()
        image = soup.find("div", class_="content-area").find('img').get("src")
        paragraphs = soup.find("div", class_="content-area").find_all('p')
        text = " ".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))

        if not text:
            return None

        # Summarization via Hugging Face API
        summary_resp = query_huggingface({"inputs": text, "parameters": {"min_length": 100, "max_length": 300}}, API_URL_SUMMARY)
        summary = summary_resp[0]['summary_text'] if summary_resp and isinstance(summary_resp, list) else text[:500]

        # Sentiment via Hugging Face API
        sentiment_resp = query_huggingface({"inputs": summary}, API_URL_SENTIMENT)
        if sentiment_resp and isinstance(sentiment_resp, list):
            sentiment = sentiment_resp[0]
        else:
            sentiment = {"label": "NEUTRAL", "score": 0.0}

        return {
            "url": url,
            "picture": image,
            "heading": heading,
            "blog": summary,
            "sentiment": sentiment["label"],
            "confidence": round(sentiment["score"], 2)
        }


# ------------------- Flask App -------------------
app = Flask(__name__)
CORS(app)
db = DatabaseManager(DATABASE_FILE)
scraper = NewsScraper()


def scheduled_scrape():
    print("Running scheduled news scrape...")
    try:
        headlines = scraper.scrape_geo_news()
        for headline in headlines:
            db.save_headline(headline)
        print(f"Added {len(headlines)} new articles")
    except Exception as e:
        print(f"Scheduled job failed: {e}")


# Schedule scraping job
schedule.every(SCRAPE_INTERVAL_MINUTES).minutes.do(scheduled_scrape)


# Run scheduled jobs in a background thread
def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(1)


scheduler_thread = threading.Thread(target=run_scheduler)
scheduler_thread.daemon = True
scheduler_thread.start()

# Ensure cleanup on exit
atexit.register(lambda: scraper.driver.quit() if scraper.driver else None)


@app.route("/scrape", methods=["GET"])
def manual_scrape():
    headlines = scraper.scrape_geo_news()
    for headline in headlines:
        db.save_headline(headline)
    return jsonify({"status": "success", "count": len(headlines)})


@app.route("/news", methods=["GET"])
def get_news():
    return jsonify(db.get_all_headlines())


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
    scheduled_scrape()
    app.run(debug=True, host="0.0.0.0")
