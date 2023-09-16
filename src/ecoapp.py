from flask import Flask, render_template
import analyze_image

app = Flask(__name__)

@app.route('/')
def home():
    print("HIIIII")
    data = {'img_path': 'last_img.png', 'result': analyze_image.image_analyze()}
    return render_template('index.html', data=data)

@app.route('/call_python_function')
def call_python_function():
    result = analyze_image.image_analyze()  # Update the image
    print("Called call_python_function")
    return "Image updated"

if __name__ == '__main__':
    app.run(debug=True)
