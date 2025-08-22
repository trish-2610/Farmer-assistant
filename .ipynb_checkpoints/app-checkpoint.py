from flask import Flask, request, jsonify, render_template
from model import final_model

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        query = data.get("query", "").strip()
        if not query:
            return jsonify({"error": "❌ Query is missing"}), 400

        result = final_model(query)
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("✅ Flask app starting...")
    app.run(debug=True, use_reloader=False)