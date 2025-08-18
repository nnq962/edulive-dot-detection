from circle_detector import CircleDetector

model = CircleDetector()

model.predict(
    source="test_images/IMG_0844.jpeg",
    mode="contour",
    show=True,
    verbose=True,
    save=False
)