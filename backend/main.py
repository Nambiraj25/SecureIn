from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
import random as r


os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.api_key='AIzaSyAwYmVsNNPSQOfsUJXvwb3uNmaCRJLgiLk'
    
@app.route("/predict", methods=['GET','POST'])
@cross_origin()
def trainRoute():
    data = request.json
    from datetime import datetime

    # Get current timestamp and format it
    formatted_timestamp = datetime.now().strftime("%Y-%m-%d%H-%M-%S")
    file_name = formatted_timestamp
    print(file_name)
    os.system("python app.py "+data['endpoint']+" "+data['api_key']+" "+file_name)
    csv_file = "../frontend/public/output/"+file_name+"/llm_vulnerability_report.csv"
    pdf_file = "../frontend/public/output/"+file_name+"/llm_penetration_report.pdf"
    if os.path.exists(csv_file) and os.path.exists(pdf_file):
        print("Files exist")
        return jsonify({"folder_name": file_name, "status": "Success"})
    return

if __name__ == "__main__":
    clApp = ClientApp()
    #app.run(port=8080,debug=True)
    app.run(host='0.0.0.0', port=8080)