from flask import Flask, render_template, request, jsonify
import requests
from bs4 import BeautifulSoup
from groq import Groq

app = Flask(__name__)
app.secret_key = 'api'

# Initialize Groq client
client = Groq(api_key='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')

def scrape_website(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)

        return text
    except Exception as e:
        return str(e)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    try:
        data = request.json
        url = data.get('url')
        question = data.get('question')

        website_content = scrape_website(url)

        if isinstance(website_content, str) and website_content.startswith('HTTP'):
            return jsonify({'error': 'Failed to access website'})

        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant analyzing website content."},
                {"role": "user", "content": f"Based on this website content: {website_content}\n\nQuestion: {question}"}
            ],
            model="mixtral-8x7b-32768",
            temperature=0.7
        )

        answer = response.choices[0].message.content
        return jsonify({'response': answer})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)