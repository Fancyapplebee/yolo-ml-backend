import os
from label_studio_ml.model import LabelStudioMLBase
from ultralytics import YOLO


class YOLOBackend(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        weights_path = os.environ.get("WEIGHTS_PATH", "weights/best.pt")
        self.model = YOLO(weights_path)

        # Map YOLO class names → Label Studio label names
        self.label_map = {
            'red': 'Red Alliance Robot',
            'blue': 'Blue Alliance Robot',
        }

    def predict(self, tasks, **kwargs):
        predictions = []
        for task in tasks:
            image_url = task["data"]["image"]
            local_path = self.get_local_path(image_url, task_id=task["id"])

            results = self.model(local_path)[0]
            regions = []

            img_h, img_w = results.orig_shape[0], results.orig_shape[1]

            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                score = float(box.conf[0])

                yolo_label = self.model.names[int(box.cls[0])]
                ls_label = self.label_map.get(yolo_label, yolo_label)

                regions.append({
                    "from_name": "label",
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "value": {
                        "rectanglelabels": [ls_label],
                        "x": x1 / img_w * 100,
                        "y": y1 / img_h * 100,
                        "width": (x2 - x1) / img_w * 100,
                        "height": (y2 - y1) / img_h * 100,
                    },
                    "score": score,
                })

            predictions.append({
                "result": regions,
                "score": sum(r["score"] for r in regions) / max(len(regions), 1) if regions else 0.0
            })

        return predictions