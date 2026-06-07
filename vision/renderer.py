import cv2


class Renderer:

    def draw_bounding_box(  # FIX: typo "boundering" → "bounding"
        self, frame, x, y, w, h, color=(255, 0, 0)
    ):
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    def draw_text(
        self, frame, text, position, color=(0, 255, 0)  # deve ser (int, int) — 2-tupla
    ):
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def draw_features(self, frame, features):
        x = features["x"]
        y = features["y"]
        area = features["area"]
        aspect_ratio = features["aspect_ratio"]
        circularity = features["circularity"]
        holes = features["holes"]
        is_hollow = features["is_hollow"]

        # FIX: (x, y, -10) era 3-tupla inválida → corrigido para (x, y - 10)
        self.draw_text(frame, f"Area: {int(area)}", (x, y - 10), (0, 255, 0))
        self.draw_text(frame, f"AR: {aspect_ratio:.2f}", (x, y - 35), (0, 255, 255))
        self.draw_text(frame, f"Circ: {circularity:.2f}", (x, y - 60), (255, 255, 0))
        # FIX: holes é int → removido :.2f
        self.draw_text(frame, f"Holes: {holes}", (x, y - 80), (255, 255, 0))
        self.draw_text(frame, f"Hollow: {is_hollow}", (x, y - 100), (255, 200, 0))

    def draw_complete_overlay(self, frame, features):
        x = features["x"]
        y = features["y"]
        w = features["width"]
        h = features["height"]

        self.draw_bounding_box(frame, x, y, w, h)
        self.draw_features(frame, features)
        return frame
