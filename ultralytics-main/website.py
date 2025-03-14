from flask import Flask, render_template_string, send_file
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


@app.route('/')
def index():
    # 简单模拟加载模型和检测图片的过程
    worker = Worker()
    model_path = r'E:\Yolo\ultralytics-main\ultralytics-main\runs\train\exp33\weights\best.pt'  # 替换为实际的模型路径
    if worker.load_model(model_path):
        image_path = r'E:\Yolo\ultralytics-main\ultralytics-main\ultralytics\images\xray_hard02907.png'  # 替换为实际的图片路径
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
            image_bytes = worker.get_annotated_image_bytes()
            if image_bytes:
                import base64
                image_base64 = base64.b64encode(image_bytes.getvalue()).decode('utf-8')
            else:
                image_base64 = ""

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
                <h2>检测图片</h2>
                {f'<img src="data:image/jpeg;base64,{image_base64}" alt="Annotated Image">' if image_base64 else '未找到检测图片'}
            </body>
            </html>
            """
            return render_template_string(html)
    return "模型加载或图片检测失败"


if __name__ == '__main__':
    print("#############################################################################")
    print("将UI.py的核心功能集成到网页中，避免弹出独立的窗口，并且将检测结果展示在网页上。")
    print("Copy the Worker class from UI.py into website.py")
    print("model_path 和 image_path 为实际的模型文件路径和图片文件路径。")
    print("#############################################################################")

    app.run(debug=True)
