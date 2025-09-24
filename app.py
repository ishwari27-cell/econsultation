from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_from_directory
from collections import defaultdict, Counter
from datetime import datetime
from wordcloud import WordCloud
from textblob import TextBlob
from langdetect import detect
from transformers import MarianMTModel, MarianTokenizer
import os, uuid, re
import unicodedata

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # replace with secure secret in production

# Admin demo credentials
ADMIN_CREDENTIALS = {'username': 'admin', 'password': 'securepass'}

# Translation model (Hindi/Marathi -> English)
MODEL_NAME = "Helsinki-NLP/opus-mt-hi-en"
tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model = MarianMTModel.from_pretrained(MODEL_NAME)

# Directories for uploaded PDFs and generated wordclouds
PROPOSAL_DIR = os.path.join("static", "proposals")
WORDCLOUD_DIR = os.path.join("static", "wordclouds")
os.makedirs(PROPOSAL_DIR, exist_ok=True)
os.makedirs(WORDCLOUD_DIR, exist_ok=True)

# --- Stopwords and improved tokenization for top-words and wordcloud ---
EN_STOPWORDS = {
    "the","and","for","that","this","with","are","was","were","is","it","its","of","to","a","an",
    "in","on","be","by","as","at","from","or","we","you","they","their","our","i","me","my","has",
    "have","but","not","so","if","would","can","could","should","will","may","about","which","what",
    "when","where","who","whom","how","these","those","there","then","than","also","per","each","any",
    "such","through","during","between","into","over","under","above","below","because","while","both",
    "more","most","some","no","only","own","same","other","others","much","many","here","why","shall"
}

HINDI_MARATHI_STOPWORDS = {
    "hai","hain","hai.","hai,","hai?","tha","thi","the","kaa","ka","ke","ki","ki.","ko","mein","me","mein.",
    "mein,","se","se.","se,","par","lekin","aur","ya","yae","ye","yeh","woh","woh.","woh,","unka","unka.","unka,",
    "unka","unka","ke","kya","kuch","kuchh","bahut","bahut.","bahut,","aisa","aise","unka","us","uska","uske",
    "aahe","ahe","ahet","ahe.","ahe,","ahe?","mi","mazi","majhe","tu","tujha","tya","tyachi","tya.","tya,",
    "aat","aat.","aat,","kaahe","karan","mhanun","mhanje","mhanje.","mhanje,","kay","kon","kahi","kahi.",
    "also","please","see","note","kindly","respectively"
}

STOPWORDS = EN_STOPWORDS.union(HINDI_MARATHI_STOPWORDS)
_word_token_re = re.compile(r"[a-zA-Z]{3,}")  # tokens with at least 3 letters

def tokenize_filtered(text):
    if not text:
        return []
    text = text.lower()
    cleaned = re.sub(r'[^a-z\s]', ' ', text)
    tokens = _word_token_re.findall(cleaned)
    return [t for t in tokens if t not in STOPWORDS]

def top_n_words_filtered(text, n=5):
    tokens = tokenize_filtered(text)
    return Counter(tokens).most_common(n)

# --- Helpers ---
def safe_filename(s: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', s)

def translate_to_english(text, lang_code):
    if lang_code in ['hi', 'mr']:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        translated = model.generate(**inputs)
        return tokenizer.decode(translated[0], skip_special_tokens=True)
    return text

def analyze_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return 'positive'
    if polarity < -0.1:
        return 'negative'
    return 'neutral'

# Heuristic summarizer focused on suggestions/changes
SUGGESTION_KEYWORDS = {
    "suggest","suggests","suggested","recommend","recommendation","recommendations",
    "should","need","needs","require","requires","propose","proposes","proposed",
    "change","changes","improve","improves","improved","add","include","consider",
    "remove","replace","clarify","clarification","strengthen","support","example","examples"
}

def extract_suggestion_summary(original_text, translated_text):
    source = original_text.strip() or translated_text.strip() or ""
    if not source:
        return ""
    sentences = re.split(r'(?<=[.!?])\s+', source)
    suggestions = [s.strip() for s in sentences if any(kw in s.lower() for kw in SUGGESTION_KEYWORDS)]
    if suggestions:
        return ' '.join(suggestions[:2])
    if sentences:
        longest = max(sentences, key=lambda s: len(s))
        return (longest.strip()[:200] + ('...' if len(longest.strip()) > 200 else '')).strip()
    return source[:200].strip()

# Phrase extraction utilities (cleaner n-grams, filter noisy fragments)
def _normalize_text_for_ngrams(text):
    # prefer ascii-translated text; normalize unicode and remove diacritics
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(ch for ch in text if not unicodedata.combining(ch))
    return text

def valid_token_for_ngrams(tok):
    tok = tok.strip()
    if len(tok) < 3:
        return False
    if tok in STOPWORDS:
        return False
    return tok.isalpha()

def extract_ngrams_from_text(text, min_n=2, max_n=5):
    t = _normalize_text_for_ngrams(text).lower()
    t_clean = re.sub(r'[^a-z\s]', ' ', t)
    toks = [tok for tok in t_clean.split() if valid_token_for_ngrams(tok)]
    ngrams = []
    l = len(toks)
    for n in range(min_n, min(max_n, l) + 1):
        for i in range(0, l- n + 1):
            ng = ' '.join(toks[i:i + n])
            ngrams.append(ng)
    return ngrams

def is_reasonable_quote(q):
    if not q or len(q) < 6:
        return False
    toks = q.split()
    if len(toks) < 2:
        return False
    return any(len(t) >= 4 for t in toks)

# --- In-memory storage ---
proposals_db = {}
comments_db = defaultdict(list)

# Seed initial sample proposal and comments
sample_id = "draft_policy_2025"
proposals_db[sample_id] = {
    "id": sample_id,
    "title": "draft_policy_2025.pdf",
    "filename": None,
    "uploaded_at": "2025-07-01"
}

comments_db[sample_id].extend([
    {"id": "c201", "name": "Aarav Mehta", "profession": "Lawyer", "text": "यह संशोधन नागरिक अधिकारों को मजबूत करता है और एक स्वागत योग्य बदलाव है।", "date": "2025-07-10", "original": "यह संशोधन नागरिक अधिकारों को मजबूत करता है और एक स्वागत योग्य बदलाव है।"},
    {"id": "c202", "name": "Neha Sharma", "profession": "Teacher", "text": "शिक्षकों के लिए यह प्रस्ताव अस्पष्ट आहे.", "date": "2025-07-18", "original": "शिक्षकों के लिए यह प्रस्ताव अस्पष्ट आहे."},
    {"id": "c203", "name": "Ravi Deshmukh", "profession": "Engineer", "text": "ही योजना डिजिटल पारदर्शकतेसाठी उत्कृष्ट आहे.", "date": "2025-08-05", "original": "ही योजना डिजिटल पारदर्शकतेसाठी उत्कृष्ट आहे."},
    {"id": "c204", "name": "Fatima Khan", "profession": "Social Worker", "text": "ग्रामीण भागातील अडचणींकडे दुर्लक्ष केले गेले आहे.", "date": "2025-08-15", "original": "ग्रामीण भागातील अडचणींकडे दुर्लक्ष केले गेले आहे."},
    {"id": "c205", "name": "Kiran Joshi", "profession": "Student", "text": "This draft is progressive and easy to understand.", "date": "2025-09-01", "original": "This draft is progressive and easy to understand."},
    {"id": "c206", "name": "Anil Patil", "profession": "Retired Officer", "text": "The language is bureaucratic and lacks clarity. I suggest simplifying section 3 and adding examples.", "date": "2025-09-10", "original": "The language is bureaucratic and lacks clarity. I suggest simplifying section 3 and adding examples."},
    {"id": "c207", "name": "Meena Rao", "profession": "Doctor", "text": "स्वास्थ्य संबंधित तरतुदी समावेशक आहेत.", "date": "2025-09-20", "original": "स्वास्थ्य संबंधित तरतुदी समावेशक आहेत."},
    {"id": "c208", "name": "Suresh Kulkarni", "profession": "Farmer", "text": "कृषी सुधारणा या मसुद्यात नाहीत.", "date": "2025-10-02", "original": "कृषी सुधारणा या मसुद्यात नाहीत."},
    {"id": "c209", "name": "Priya Nair", "profession": "Entrepreneur", "text": "Encouraging innovation and public-private collaboration.", "date": "2025-10-15", "original": "Encouraging innovation and public-private collaboration."},
    {"id": "c210", "name": "Mohammed Abbas", "profession": "Journalist", "text": "The technical jargon makes it inaccessible to the public.", "date": "2025-10-25", "original": "The technical jargon makes it inaccessible to the public."},
    {"id":"s301","name":"Asha Patel","profession":"Teacher","original":"इस मसुद्याचा उद्देश चांगला आहे परंतु अंमलबजावणीसाठी स्पष्ट मार्गदर्शन आणि बजेट वाटप आवश्यक आहे. शाळांमध्ये डिजिटल साधनांची उपलब्धता सुनिश्चित करण्यासाठी प्रयोगात्मक पायलट योजनांची नोंद घ्या; तसेच शिक्षक प्रशिक्षणासाठी तीन महिन्याचे विशेष प्रशिक्षण सत्र असावे.","text":"इस मसुद्याचा उद्देश चांगला आहे परंतु अंमलबजावणीसाठी स्पष्ट मार्गदर्शन आणि बजेट वाटप आवश्यक आहे. शाळांमध्ये डिजिटल साधनांची उपलब्धता सुनिश्चित करण्यासाठी प्रयोगात्मक पायलट योजनांची नोंद घ्या; तसेच शिक्षक प्रशिक्षणासाठी तीन महिन्याचे विशेष प्रशिक्षण सत्र असावे.","date":"2025-07-12"},
    {"id":"s302","name":"Vikram Singh","profession":"Small Business Owner","original":"The draft sounds promising for large investors, but it overlooks compliance costs for small vendors. I recommend adding a phased compliance window, tax relief for the first two years, and a simpler reporting template for micro-enterprises.","text":"The draft sounds promising for large investors, but it overlooks compliance costs for small vendors. I recommend adding a phased compliance window, tax relief for the first two years, and a simpler reporting template for micro-enterprises.","date":"2025-07-25"},
    {"id":"s303","name":"Sunita Rao","profession":"Social Worker","original":"ग्रामीण भागातील सेवा पोहोचवण्यासाठी स्थानिक भाषा आणि स्थानिक स्वयंसेवकांचा समावेश करणे गरजेचे आहे. कृपया स्थानिक समुदायांमधून प्रतिनिधी नियुक्त करून त्यांच्या अडचणीचे नकाशीकरण करा; प्रस्तावित संवाद केंद्रे आठवड्यातून किमान दोन वेळा स्थानिक नागरिकांसाठी खुले असावीत.","text":"ग्रामीण भागातील सेवा पोहोचवण्यासाठी स्थानिक भाषा आणि स्थानिक स्वयंसेवकांचा समावेश करणे गरजेचे आहे. कृपया स्थानिक समुदायांमधून प्रतिनिधी नियुक्त करून त्यांच्या अडचणीचे नकाशीकरण करा; प्रस्तावित संवाद केंद्रे आठवड्यातून किमान दोन वेळा स्थानिक नागरिकांसाठी खुले असावीत.","date":"2025-08-03"},
    {"id":"s304","name":"Rahul Mehra","profession":"Engineer","original":"Technically this is a good start, but the data privacy clauses are insufficient. Mandatory biometric data storage without strict retention and access controls is risky. I propose: (1) limit retention to 90 days by default, (2) log all access with audit trails, (3) require explicit consent for secondary use.","text":"Technically this is a good start, but the data privacy clauses are insufficient. Mandatory biometric data storage without strict retention and access controls is risky. I propose: (1) limit retention to 90 days by default, (2) log all access with audit trails, (3) require explicit consent for secondary use.","date":"2025-08-18"},
    {"id":"s305","name":"Meera Kulkarni","profession":"Doctor","original":"स्वास्थ्य धोरणांसाठी आरोग्यसेवांमध्ये समानता आवश्यक आहे. प्राथमिक स्वास्थ्य केंद्रांसाठी पुरवठा आणि औषधे नियमित करावी; तसेच ग्रामीण आरोग्य कर्मचारी यांना ठराविक प्रशिक्षण आणि मोबाईल रुग्णनोंदणीची सुविधा द्यावी.","text":"स्वास्थ्य धोरणांसाठी आरोग्यसेवांमध्ये समानता आवश्यक आहे. प्राथमिक स्वास्थ्य केंद्रांसाठी पुरवठा आणि औषधे नियमित करावी; तसेच ग्रामीण आरोग्य कर्मचारी यांना ठराविक प्रशिक्षण आणि मोबाईल रुग्णनोंदणीची सुविधा द्यावी.","date":"2025-09-02"},
    {"id":"s306","name":"Anil Kapoor","profession":"Journalist","original":"The consultation process appears to be tokenistic unless public submissions are acknowledged with responses. My suggestion: publish a short 'what we changed' report after consultation closes and summarize major issues and the government's response. Transparency will increase trust.","text":"The consultation process appears to be tokenistic unless public submissions are acknowledged with responses. My suggestion: publish a short 'what we changed' report after consultation closes and summarize major issues and the government's response. Transparency will increase trust.","date":"2025-09-12"},
    {"id":"s307","name":"Lakshmi Iyer","profession":"Teacher","original":"Some sections use high-level jargon that ordinary parents won't understand. Add a plain-language summary at the start of each chapter and include FAQ sections for common stakeholder groups like parents, teachers, and small business owners.","text":"Some sections use high-level jargon that ordinary parents won't understand. Add a plain-language summary at the start of each chapter and include FAQ sections for common stakeholder groups like parents, teachers, and small business owners.","date":"2025-09-20"},
    {"id":"s308","name":"Ramesh Patil","profession":"Farmer","original":"कृषी सुधारणा भागात प्रत्यक्ष मदत आणि आजारवाढीसाठी उपाय हवा. सेंद्रिय शेतीला प्रोत्साहनासाठी अनुदान आणि गावथर प्रशिक्षण केंद्र आवश्यक आहेत. मी सुचवतो की शेतकऱ्यांसाठी प्रत्यक्ष अनुदानाची तरतूद आणि पिकविम्यासाठी सुलभता द्यावी.","text":"कृषी सुधारणा भागात प्रत्यक्ष मदत आणि आजारवाढीसाठी उपाय हवा. सेंद्रिय शेतीला प्रोत्साहनासाठी अनुदान आणि गावथर प्रशिक्षण केंद्र आवश्यक आहेत. मी सुचवतो की शेतकऱ्यांसाठी प्रत्यक्ष अनुदानाची तरतूद आणि पिकविम्यासाठी सुलभता द्यावी.","date":"2025-10-05"},
    {"id":"s309","name":"Priya Nair","profession":"Entrepreneur","original":"Overall positive — the policy encourages innovation, but the procurement clauses could lock out startups. Recommend a separate micro-procurement lane with simplified tendering and a cap on pre-qualification requirements for firms under a certain turnover.","text":"Overall positive — the policy encourages innovation, but the procurement clauses could lock out startups. Recommend a separate micro-procurement lane with simplified tendering and a cap on pre-qualification requirements for firms under a certain turnover.","date":"2025-10-18"},
    {"id":"s310","name":"Suresh Kulkarni","profession":"Farmer","original":"मसुद्यातील आर्थिक मदत योजनाही असामान्य आणि अपर्याप्त वाटतात. जर प्रत्यक्ष जोड दिला गेला नाही तर या सुधारणा शेतकरी वर्गाला लाभ नव्हेत. प्रस्तावित अनुदानाचे तपशील व अटी स्पष्ट कराव्यात.","text":"मसुद्यातील आर्थिक मदत योजनाही असामान्य आणि अपर्याप्त वाटतात. जर प्रत्यक्ष जोड दिला गेला नाही तर या सुधारणा शेतकरी वर्गाला लाभ नव्हेत. प्रस्तावित अनुदानाचे तपशील व अटी स्पष्ट कराव्यात.","date":"2025-11-02"},
    {"id":"s311","name":"Anita Desai","profession":"Retired Officer","original":"The draft recycles old frameworks and lacks measurable targets. I recommend adding SMART targets with timelines and assigning departmental accountability points. Without measurable KPIs, monitoring will remain ineffective.","text":"The draft recycles old frameworks and lacks measurable targets. I recommend adding SMART targets with timelines and assigning departmental accountability points. Without measurable KPIs, monitoring will remain ineffective.","date":"2025-11-10"},
    {"id":"s312","name":"Imran Sheikh","profession":"Journalist","original":"I find several ambiguities in the grievance redressal section. Citizens need a simple escalation matrix and maximum resolution timelines. Please include a one-page flowchart and contact points for each escalation level.","text":"I find several ambiguities in the grievance redressal section. Citizens need a simple escalation matrix and maximum resolution timelines. Please include a one-page flowchart and contact points for each escalation level.","date":"2025-11-20"}

])

# --- Routes ---
@app.route('/')
def home():
    proposals = list(proposals_db.values())
    return render_template('home.html', proposals=proposals, is_admin=session.get('admin', False))

@app.route('/proposal/<proposal_id>', methods=['GET'])
def proposal_page(proposal_id):
    prop = proposals_db.get(proposal_id)
    if not prop:
        return "Proposal not found", 404
    comments = comments_db.get(proposal_id, [])
    for c in comments:
        if 'original' not in c:
            try:
                detected = detect(c.get('text', ''))
            except Exception:
                detected = 'en'
            c['original'] = c.get('text', '')
            c['text'] = translate_to_english(c['original'], detected)
    return render_template('proposal.html', proposal=prop, comments=comments, is_admin=session.get('admin', False))

@app.route('/submit_comment', methods=['POST'])
def submit_comment():
    data = request.form
    proposal_id = data.get('proposal')
    if not proposal_id or proposal_id not in proposals_db:
        return redirect(url_for('home'))
    raw_text = data.get('comment', '').strip()
    name = data.get('name', '').strip() or 'Anonymous'
    profession = data.get('profession', '').strip() or 'Citizen'
    if not raw_text:
        return redirect(url_for('proposal_page', proposal_id=proposal_id))
    try:
        lang_code = detect(raw_text)
    except Exception:
        lang_code = 'en'
    translated = translate_to_english(raw_text, lang_code)
    comment = {
        'id': str(uuid.uuid4()),
        'name': name,
        'profession': profession,
        'text': translated,
        'original': raw_text,
        'date': datetime.today().strftime("%Y-%m-%d")
    }
    comments_db.setdefault(proposal_id, []).append(comment)
    return redirect(url_for('proposal_page', proposal_id=proposal_id))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = request.form.get('username','')
        pwd = request.form.get('password','')
        if user == ADMIN_CREDENTIALS['username'] and pwd == ADMIN_CREDENTIALS['password']:
            session['admin'] = True
            return redirect(url_for('admin_dashboard'))
        return render_template('login.html', error="Invalid credentials")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('admin', None)
    return redirect(url_for('home'))

@app.route('/admin/upload', methods=['GET', 'POST'])
def upload_proposal():
    if not session.get('admin'):
        return redirect(url_for('login'))
    if request.method == 'POST':
        title = request.form.get('title', '').strip()
        file = request.files.get('file')
        if not title or not file:
            return render_template('upload_proposal.html', error="Title and PDF required")
        if not file.filename.lower().endswith('.pdf'):
            return render_template('upload_proposal.html', error="Only PDF files are allowed")
        pid = safe_filename(title)
        if pid in proposals_db:
            pid = f"{pid}_{str(uuid.uuid4())[:8]}"
        fname = f"{pid}_{int(datetime.now().timestamp())}.pdf"
        safe_fname = safe_filename(fname)
        save_path = os.path.join(PROPOSAL_DIR, safe_fname)
        file.save(save_path)
        proposals_db[pid] = {
            "id": pid,
            "title": title,
            "filename": safe_fname,
            "uploaded_at": datetime.today().strftime("%Y-%m-%d")
        }
        comments_db.setdefault(pid, [])
        return redirect(url_for('admin_dashboard'))
    return render_template('upload_proposal.html')

@app.route('/admin')
def admin_dashboard():
    if not session.get('admin'):
        return redirect(url_for('login'))
    proposals = list(proposals_db.values())
    return render_template('admin_dashboard.html', proposals=proposals)

@app.route('/static/proposals/<path:filename>')
def serve_proposal_file(filename):
    return send_from_directory(PROPOSAL_DIR, filename)

@app.route('/analysis/<proposal_id>')
def analysis(proposal_id):
    if not session.get('admin'):
        return redirect(url_for('login'))
    prop = proposals_db.get(proposal_id)
    if not prop:
        return "Proposal not found", 404
    comments = comments_db.get(proposal_id, [])
    for c in comments:
        if 'original' not in c:
            try:
                detected = detect(c.get('text', ''))
            except Exception:
                detected = 'en'
            c['original'] = c.get('text', '')
            c['text'] = translate_to_english(c['original'], detected)

    # --- TOP WORDS & WORDCLOUD: only from sentiment-bearing (positive or negative) comments and filtered tokens
    sentiment_tokens = []
    for c in comments:
        s = analyze_sentiment(c['text'])
        if s in ('positive', 'negative'):
            sentiment_tokens.extend(tokenize_filtered(c['text']))
    top_words = Counter(sentiment_tokens).most_common(5)
    summary = ' '.join(sentiment_tokens[:50]) if sentiment_tokens else ''

    # --- SENTIMENT TIMELINE, PROFESSION SENTIMENT, QUOTES, CONTROVERSIAL: use FULL comments (no extra filtering)
    timeline_sentiments = defaultdict(lambda: {'positive':0,'neutral':0,'negative':0})
    profession_sentiments = defaultdict(lambda: {'positive':0,'neutral':0,'negative':0})
    phrase_counter = Counter()
    pos = neg = neu = 0

    # Use translated English text for phrase extraction where possible (fall back to original)
    english_texts = [c['text'] for c in comments if isinstance(c.get('text'), str) and c['text'].strip()]

    for c in comments:
        s = analyze_sentiment(c['text'])
        profession_sentiments[c['profession']][s] += 1
        if s == 'positive':
            pos += 1
        elif s == 'negative':
            neg += 1
        else:
            neu += 1
        # ensure month key is consistently formatted as YYYY-MM for correct chronological sorting
        try:
            month_key = datetime.strptime(c['date'], "%Y-%m-%d").strftime("%Y-%m")
        except Exception:
            # fallback: if date is partial or malformed, try to parse year-month if present
            try:
                parsed = datetime.strptime(c['date'], "%Y-%m")
                month_key = parsed.strftime("%Y-%m")
            except Exception:
                month_key = c['date']
        timeline_sentiments[month_key][s] += 1

        # Use translated English text for phrase extraction if available, else original
        text_for_phrases = c.get('text') or c.get('original') or ''
        phrase_ngrams = extract_ngrams_from_text(text_for_phrases, min_n=2, max_n=4)
        phrase_counter.update(phrase_ngrams)

    # sort months chronologically by parsing YYYY-MM keys, then format labels
    def _parse_month_key(k):
        try:
            return datetime.strptime(k, "%Y-%m")
        except Exception:
            # try common alternatives, fallback to epoch ordering
            try:
                return datetime.strptime(k, "%Y-%m-%d")
            except Exception:
                return None

    month_items = []
    for k, v in timeline_sentiments.items():
        dt = _parse_month_key(k)
        month_items.append((k, v, dt))
    # sort by datetime when available, else by original key
    month_items.sort(key=lambda x: (x[2] is None, x[2] or x[0]))
    timeline_ordered = {}
    for k, v, dt in month_items:
        if dt:
            label = dt.strftime("%b %Y")
        else:
            label = k
        timeline_ordered[label] = v

    # Wordcloud generated only from sentiment_tokens (filtered)
    wordcloud_filename = None
    if sentiment_tokens:
        try:
            wc_text = ' '.join(sentiment_tokens)
            wc = WordCloud(width=1000, height=500, background_color='white', collocations=False).generate(wc_text)
            safe_pid = safe_filename(proposal_id)
            fname = f"{safe_pid}_{int(datetime.now().timestamp())}.png"
            full_path = os.path.join(WORDCLOUD_DIR, fname)
            wc.to_file(full_path)
            wordcloud_filename = f"wordclouds/{fname}"
        except Exception as e:
            print("WordCloud error:", e)
            wordcloud_filename = None

    # Quote selection: pick top ngram if it is reasonable (avoid short/noisy fragments)
    quote = None
    if phrase_counter:
        cand = phrase_counter.most_common(10)
        for ph, cnt in cand:
            if is_reasonable_quote(ph):
                quote = ph
                break

    # Controversial phrases: require minimum frequency before cross-sentiment check
    controversial_phrases = []
    MIN_PHRASE_FREQ = 2
    for p, count in phrase_counter.items():
        if count < MIN_PHRASE_FREQ:
            continue
        if any(analyze_sentiment(t) == 'positive' for t in english_texts if p in t) and \
           any(analyze_sentiment(t) == 'negative' for t in english_texts if p in t):
            controversial_phrases.append(p)

    profession_sentiments = {p: dict(v) for p, v in profession_sentiments.items()}

    return render_template('analysis.html',
                           proposal=prop,
                           summary=summary,
                           top_words=top_words,
                           pos=pos, neg=neg, neu=neu,
                           timeline_sentiments=timeline_ordered,
                           wordcloud_filename=wordcloud_filename,
                           quote=quote,
                           controversial_phrases=controversial_phrases,
                           profession_sentiments=profession_sentiments,
                           comments=comments)

@app.route('/comment_analysis/<proposal_id>/<comment_id>')
def comment_analysis(proposal_id, comment_id):
    if not session.get('admin'):
        return jsonify({'error':'unauthorized'}), 401
    comments = comments_db.get(proposal_id, [])
    comment = next((c for c in comments if c['id'] == comment_id), None)
    if not comment:
        return jsonify({'error':'not found'}), 404
    text = comment.get('text', '') or ''
    sentiment = analyze_sentiment(text)
    top5 = top_n_words_filtered(text, 5)
    summary = extract_suggestion_summary(comment.get('original',''), text)
    return jsonify({
        'id': comment_id,
        'name': comment.get('name'),
        'profession': comment.get('profession'),
        'date': comment.get('date'),
        'sentiment': sentiment,
        'top_words': top5,
        'summary': summary,
        'original': comment.get('original',''),
        'translated': text
    })

@app.route('/delete_comment/<proposal_id>/<comment_id>')
def delete_comment(proposal_id, comment_id):
    if not session.get('admin'):
        return redirect(url_for('login'))
    comments_db[proposal_id] = [c for c in comments_db.get(proposal_id, []) if c['id'] != comment_id]
    return redirect(url_for('analysis', proposal_id=proposal_id))

if __name__ == '__main__':
    app.run(debug=True)
