import logging
from flask import Flask, request, jsonify
from src.Agentic_RAG_Evaluation import ask
from flask_cors import CORS

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
CORS(app)

@app.route("/test", methods=['GET'])
def testing():  # plain‑text response is fine for probes
    return "ok", 200


@app.route("/chat", methods=['POST'])
def chat_view():                     # **make sure the function is renamed**
    data = request.get_json(silent=True) or {}
    if data is None:
            return jsonify({
                "error": "Invalid JSON payload. Ensure Content-Type is application/json and that your JSON uses double quotes."
            }), 400

    question = data.get("question")

    if not question:
        return jsonify(error="JSON body must contain 'question'"), 400

    try:
        answer = ask(question)       # ← call the helper, NOT chat()
    except Exception as exc:
        return jsonify(error="internal error", detail=str(exc)), 500

    return jsonify(answer=answer)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
