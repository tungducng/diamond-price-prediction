from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
#install sklearn

app = Flask(__name__)

# Đường dẫn tới mô hình và pipeline đã lưu
model_paths = {
    "Linear Regression": './model/lin_reg.pkl',
    "Decision Tree": './model/tree_reg.pkl',
    "Random Forest": './model/forest_reg.pkl'
}
pipeline_path = './model/pipeline.pkl'

# Đọc dữ liệu diamonds
df = pd.read_csv('diamonds_test_set_with_index.csv')  # Thay đổi đường dẫn phù hợp với file của bạn

# Tải pipeline đã lưu
with open(pipeline_path, 'rb') as f:
    pipeline = pickle.load(f)

# Tải tất cả mô hình đã lưu
models = {}
for model_name, model_path in model_paths.items():
    with open(model_path, 'rb') as f:
        models[model_name] = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

# Thêm đoạn code này để debug
@app.route('/get_data')
def get_data():
    try:
        start = int(request.args.get('start', 0))
        length = int(request.args.get('length', 10))
        
        print("DataFrame shape:", df.shape)  # Kiểm tra kích thước DataFrame
        print("Start:", start)
        print("Length:", length)
        
        data_slice = df.iloc[start:start+length]
        data = data_slice.to_dict('records')
        
        print("Data slice:", data)  # Kiểm tra dữ liệu trước khi gửi
        
        return jsonify({
            'data': data,
            'total': len(df)
        })
    except Exception as e:
        print("Error:", str(e))
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    try:
        # Chuyển đổi dữ liệu đầu vào thành DataFrame
        new_data = pd.DataFrame([{
            'carat': float(data['carat']),
            'cut': data['cut'],
            'color': data['color'],
            'clarity': data['clarity'],
            'depth': float(data['depth']),
            'table': float(data['table']),
            'x': float(data['x']),
            'y': float(data['y']),
            'z': float(data['z'])
        }])
        
        # Xử lý dữ liệu bằng pipeline
        new_data_prepared = pipeline.transform(new_data)
        
        # Lấy danh sách mô hình được chọn
        selected_models = request.form.getlist('models')
        
        # Dự đoán giá trị
        predictions = {}
        for model_key in selected_models:
            model_name = next(name for name, key in model_paths.items() if key.endswith(f"{model_key}.pkl"))
            predictions[model_name] = models[model_name].predict(new_data_prepared)[0]
            
        # Trả về kết quả dưới dạng bảng
        return render_template('index.html', predictions=predictions)
    except Exception as e:
        return render_template('index.html', predictions={}, error=str(e))

if __name__ == "__main__":
    app.run(debug=True, port=5005)
