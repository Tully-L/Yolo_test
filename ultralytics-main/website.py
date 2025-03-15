# website.py 2025/3/14
from flask import Flask, render_template_string, request, send_file
import io
import cv2
from ultralytics import YOLO
from PyQt5.QtWidgets import QApplication
import sys
import os


app = Flask(__name__)
app_qt = QApplication(sys.argv)


# 模拟 Worker 类和 MainWindow 类的功能
class Worker:
    def __init__(self):
        self.model = None
        self.current_annotated_image = None
        self.detection_type = None


    def load_model(self, model_path):
        if model_path:
            self.model = YOLO(model_path)
            self.model.to('cpu')
            if self.model:
                return True
        return False


    def detect_objects(self, results):
        det_info = []
        for frame in results:
            class_names_dict = frame.names
            if hasattr(frame, 'obb') and frame.obb is not None:
                class_ids = frame.obb.cls
                for class_id in class_ids:
                    class_name = class_names_dict[int(class_id)]
                    det_info.append(class_name)
            else:
                print("No OBB data found in the frame.")
        return det_info


    def detect_image(self, image_path):
        if image_path:
            image = cv2.imread(image_path)
            if image is not None:
                results = self.model.predict(image)
                self.detection_type = "image"
                if results:
                    self.current_annotated_image = results[0].plot()
                    return results
        return None


    def get_annotated_image_bytes(self):
        if self.current_annotated_image is not None:
            _, buffer = cv2.imencode('.jpg', self.current_annotated_image)
            return io.BytesIO(buffer.tobytes())
        return None




@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 获取用户上传的模型文件和图片文件
        model_file = request.files['model_file']
        image_file = request.files['image_file']


        if model_file and image_file:
            # 保存上传的文件到临时目录
            model_path = 'temp_model.pt'
            image_path = 'temp_image.png'
            model_file.save(model_path)
            image_file.save(image_path)


            worker = Worker()
            if worker.load_model(model_path):
                results = worker.detect_image(image_path)
                if results:
                    det_info = worker.detect_objects(results)
                    object_count = len(det_info)
                    object_info = f"识别到的物体总个数：{object_count}\n"
                    object_dict = {}
                    for obj in det_info:
                        if obj in object_dict:
                            object_dict[obj] += 1
                        else:
                            object_dict[obj] = 1
                    sorted_objects = sorted(object_dict.items(), key=lambda x: x[1], reverse=True)
                    for obj_name, obj_count in sorted_objects:
                        object_info += f"{obj_name}: {obj_count}\n"


                    # 获取带标注的图片字节流
                    annotated_image_bytes = worker.get_annotated_image_bytes()
                    if annotated_image_bytes:
                        import base64
                        annotated_image_base64 = base64.b64encode(annotated_image_bytes.getvalue()).decode('utf-8')
                    else:
                        annotated_image_base64 = ""


                    # 读取原始图片并转换为 Base64 字符串
                    with open(image_path, 'rb') as f:
                        original_image_bytes = f.read()
                    original_image_base64 = base64.b64encode(original_image_bytes).decode('utf-8')


                    html = f"""
                    <!DOCTYPE html>
                    <html lang="en">
                    <head>
                        <meta charset="UTF-8">
                        <title>目标检测结果</title>
                    </head>
                    <body>
                        <h1>目标检测结果</h1>
                        <h2>检测信息</h2>
                        <pre>{object_info}</pre>
                        <h2>原始图片</h2>
                        <img src="data:image/png;base64,{original_image_base64}" alt="Original Image">
                        <h2>检测结果图片</h2>
                        {f'<img src="data:image/jpeg;base64,{annotated_image_base64}" alt="Annotated Image">' if annotated_image_base64 else '未找到检测图片'}
                    </body>
                    </html>
                    """
                    # 删除临时文件
                    os.remove(model_path)
                    os.remove(image_path)
                    return render_template_string(html)
            # 删除临时文件
            os.remove(model_path)
            os.remove(image_path)
            return "模型加载或图片检测失败"


    # 显示文件上传表单
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>目标检测</title>
    </head>
    <body>
        <h1>目标检测</h1>
        <form method="post" enctype="multipart/form-data">
            <label for="model_file">选择模型文件:</label>
            <input type="file" id="model_file" name="model_file" accept=".pt"><br><br>
            <label for="image_file">选择图片文件:</label>
            <input type="file" id="image_file" name="image_file" accept=".jpg,.jpeg,.png"><br><br>
            <input type="submit" value="显示结果">
        </form>
    </body>
    </html>
    """
    return render_template_string(html)




if __name__ == '__main__':
    app.run(debug=True)