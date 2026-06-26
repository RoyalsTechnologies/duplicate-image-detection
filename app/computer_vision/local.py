import io

from PIL import Image, ImageStat

from app.computer_vision.base import BaseComputerVisionClient, DetectedObject, EMBEDDING_DIM


class LocalComputerVisionClient(BaseComputerVisionClient):
    """Heuristic local CV for development and offline testing."""

    def image_embedding(self, image_bytes: bytes) -> list[float]:
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            small = image.resize((16, 16))
            color_values: list[float] = []
            for r, g, b in small.getdata():
                color_values.extend([(r / 127.5) - 1.0, (g / 127.5) - 1.0, (b / 127.5) - 1.0])
            histogram = image.resize((64, 64)).histogram()
            total = max(1, sum(histogram))
            hist_values = [(value / total) * 2.0 - 1.0 for value in histogram]
            values = (color_values + hist_values)[:EMBEDDING_DIM]
            if len(values) < EMBEDDING_DIM:
                values.extend([0.0] * (EMBEDDING_DIM - len(values)))
            return values
        except Exception:
            return [0.0] * EMBEDDING_DIM

    def detect_objects(self, image_bytes: bytes) -> list[DetectedObject]:
        features = self._image_features(image_bytes)
        if not features:
            return []

        labels: list[DetectedObject] = []
        blue_ratio = features["blue_ratio"]
        green_ratio = features["green_ratio"]
        dark_ratio = features["dark_ratio"]
        gray_ratio = features["gray_ratio"]
        brown_ratio = features["brown_ratio"]
        edge_score = features["edge_score"]
        brightness = features["brightness"]

        if blue_ratio > 0.26 and brightness < 190:
            labels.append(DetectedObject("stagnant_water", min(0.95, blue_ratio + 0.45)))
        if blue_ratio > 0.34:
            labels.append(DetectedObject("flooding", min(0.95, blue_ratio + 0.40)))
        if brown_ratio > 0.18 and edge_score > 0.13:
            labels.append(DetectedObject("rubbish", min(0.9, brown_ratio + edge_score + 0.35)))
        if dark_ratio > 0.22 and edge_score > 0.10:
            labels.append(DetectedObject("pothole", min(0.88, dark_ratio + edge_score + 0.28)))
        if gray_ratio > 0.38 and brightness > 115:
            labels.append(DetectedObject("smoke", min(0.86, gray_ratio + 0.25)))
        if green_ratio > 0.28 and brown_ratio > 0.10:
            labels.append(
                DetectedObject("blocked_drain", min(0.82, green_ratio + brown_ratio + 0.25))
            )

        labels.sort(key=lambda item: item.confidence, reverse=True)
        return labels[:5]

    def _image_features(self, image_bytes: bytes) -> dict[str, float]:
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((96, 96))
        except Exception:
            return {}

        pixels = list(image.getdata())
        total = max(1, len(pixels))
        blue = green = dark = gray = brown = 0
        for r, g, b in pixels:
            brightness = (r + g + b) / 3
            if b > r * 1.15 and b > g * 1.05:
                blue += 1
            if g > r * 1.08 and g > b * 1.08:
                green += 1
            if brightness < 70:
                dark += 1
            if abs(r - g) < 18 and abs(g - b) < 18:
                gray += 1
            if r > 75 and g > 45 and b < 80 and r >= g >= b:
                brown += 1

        stat = ImageStat.Stat(image.convert("L"))
        brightness = float(stat.mean[0])
        edge_score = self._edge_score(image)
        return {
            "blue_ratio": blue / total,
            "green_ratio": green / total,
            "dark_ratio": dark / total,
            "gray_ratio": gray / total,
            "brown_ratio": brown / total,
            "brightness": brightness,
            "edge_score": edge_score,
        }

    def _edge_score(self, image: Image.Image) -> float:
        gray = image.convert("L").resize((48, 48))
        pixels = gray.load()
        width, height = gray.size
        total = 0.0
        count = 0
        for y in range(height - 1):
            for x in range(width - 1):
                total += abs(pixels[x, y] - pixels[x + 1, y]) + abs(pixels[x, y] - pixels[x, y + 1])
                count += 2
        return min(1.0, (total / max(1, count)) / 255.0)
